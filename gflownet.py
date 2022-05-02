import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from my_transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm

def create_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(-1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
    return position_ids

class GFlowNet_Z(nn.Module):
    def __init__(self, d_model):
        nn.Module.__init__(self)
        self.to_flow = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x, pad_mask):
        x = self.to_flow(x).squeeze(-1)
        masked_x = (x.view(-1) * pad_mask.exp().view(-1)).view(x.size())
        pooled_x = masked_x.sum(1) #/ pad_mask.exp().sum(dim=-1).view(-1)
        return pooled_x

class GFlowNet_shared_embedding(nn.Module):
    def __init__(self, n_vocab, d_model, seqlen=128, n_nts=0):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.seqlen = seqlen
        self.n_vocab = n_vocab
        self.n_nts = n_nts
        self.embedding_tgt = nn.Embedding(n_vocab+n_nts, d_model)
        self.embedding_pos = nn.Embedding(seqlen, d_model)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.embedding_tgt.weight, mean=0.0, std=1/128**0.5)
        nn.init.normal_(self.embedding_pos.weight, mean=0.0, std=1/128**0.5)

    def forward(self, x):
        x = x[:, x.sum(dim=0)!=0]
        assert x.max() < self.n_vocab+self.n_nts, f"input contains token id {x.max()}, but max is {self.n_vocab+self.n_nts-1}"
        encoded_tgt = self.embedding_tgt(x)
        pos_ids = create_position_ids(x)
        encoded_tgt += self.embedding_pos(pos_ids)
        tgt_key_padding_mask = encoded_tgt.new_zeros(x.shape)
        tgt_key_padding_mask[x==0] = -float('inf')
        return encoded_tgt, tgt_key_padding_mask

class GFlowNet_encoder(nn.Module):
    def __init__(self, n_vocab, d_model, nhead, dim_feedforward, dropout, norm_first, nlayers, n_nts=0,
                 seqlen=128, batch_first=True, activation="relu", shared_embedding=None):
        nn.Module.__init__(self)
        self.d_model = d_model
        if shared_embedding is None:
            self.embedding = GFlowNet_shared_embedding(n_vocab, d_model, seqlen, n_nts=n_nts)
        else:
            self.embedding = shared_embedding
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                                batch_first=batch_first, activation=activation, norm_first=norm_first)
        encoder_norm = LayerNorm(d_model, eps=1e-5)
        self.model_encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

    def forward(self, x):
        encoded_src, src_key_padding_mask = self.embedding(x)
        memory = self.model_encoder(encoded_src, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

class GFlowNet_forward_split(nn.Module):
    '''
    picking a position to insert a <split> symbol to create one more non-empty span
    '''
    def __init__(self, d_model):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.to_pos = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                    nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        return self.to_pos(x).squeeze(-1)

class GFlowNet_forward_tag(nn.Module):
    '''
    tag a span autoregressively
    '''
    def __init__(self, n_nts, d_model):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_nts = n_nts
        self.to_pos = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                    nn.ReLU(), nn.Linear(d_model, 1))
        self.to_tok = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                        nn.ReLU(), nn.Linear(d_model, n_nts))

    def forward(self, x):
        pos_logits = self.to_pos(x)
        tok_logits = self.to_tok(x)
        return torch.cat([pos_logits, tok_logits], dim=-1)

class GFlowNet_backward(nn.Module):
    '''
    We pick <split> token one at a time for deletion
    '''
    def __init__(self, d_model):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.to_pos = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                    nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        pos_logits = self.to_pos(x).squeeze(-1)
        return pos_logits

class ar_segmenter_controller():
    def __init__(self, device, args, n_vocab, pcfg, ar_model):
        self.device = device
        self.args = args
        self.n_vocab = n_vocab
        self.pcfg = pcfg
        self.ar_model = ar_model
        self.split_sym = n_vocab + args.t_states + args.nt_states

    def sample_forward(self,
                    action : str,
                    F_logits: torch.Tensor,
                    states: torch.Tensor,
                    greedy: bool = False,
                    temperature_pos : float = -1.,):
        F_actions = self._sample_forward_actions(action, F_logits, greedy, temperature_pos)
        P_F = self.calc_forward_prob(F_logits=F_logits,
                                            F_actions=F_actions,
                                            states=states)
        states = self.apply_forward_actions(states, F_actions)
        return states, F_actions, P_F
    

    def calc_forward_prob(self,
                        action : str,
                        F_logits: torch.Tensor,
                        states: torch.Tensor, 
                        F_actions: tuple = None,
                        greedy: bool = False,
                        temperature_pos: float = 1.0):
        raise NotImplementedError

    def _sample_forward_actions(self,
                                action : str,
                                F_logits: torch.Tensor,
                                greedy: bool = False,
                                temperature_pos : float = 1.,):
        raise NotImplementedError

    @torch.no_grad()
    def apply_forward_actions(self,
                            states: list,
                            F_actions: tuple):
        '''
        states is a list of tensors
        F_actions is a tuple of three things
        '''
        actions, positions, tokens = F_actions
        for i, (action, pos, tok) in enumerate(zip(actions, positions, tokens)):
            if action == "none":
                continue
            elif action == "split":
                # insert a split symbol at pos
                states[i] = torch.cat([states[i][:pos], torch.zeros(1).to(self.device)+self.split_sym, states[i][pos:]], dim=0)
            elif action == "tag":
                # change the pos-th split symbol to tok
                states[i][states[i]==self.split_sym][pos] = tok # need to verify that this actually modifies states
        return states

    @torch.no_grad()
    def reverse_forward_actions(self,
                                states: torch.Tensor,
                                F_actions: list):
        actions, positions, _ = F_actions
        B_actions = []
        B_positions = []
        for i, (action, pos) in enumerate(zip(actions, positions)):
            if action == "none":
                B_actions.append('none')
                B_positions.append(0)
            elif action == "split":
                B_actions.append('merge')
                # return the index of the split symbol at pos
                B_positions.append((states[i][:pos]==self.split_sym).sum().item())
            elif action == "tag":
                B_actions.append('untag')
                # return the index of the split symbol at pos
                B_positions.append((states[i][:pos]==self.split_sym).sum().item())
        return (B_actions, B_positions)

    def sample_backward(self, 
                        B_logits: torch.Tensor,
                        states: torch.Tensor,
                        greedy: bool = False,
                        temperature_pos : float = -1.):
        B_actions = self._sample_backward_actions(B_logits, greedy, temperature_pos)
        P_B = self.calc_backward_prob(B_logits=B_logits,
                                    B_actions=B_actions,
                                    states=states)
        states = self.apply_backward_actions(states, B_actions)
        return states, B_actions, P_B

    def calc_backward_prob(self,
                        B_logits: torch.Tensor,
                        states: torch.Tensor,
                        B_actions: torch.Tensor = None,
                        greedy: bool = False,
                        temperature_pos: float = 1.0):
        raise NotImplementedError

    def _sample_backward_actions(self,
                                B_logits: torch.Tensor,
                                greedy: bool = False,
                                temperature_pos : float = -1.,):
        raise NotImplementedError

    @torch.no_grad()
    def apply_backward_actions(self,
                            states: torch.Tensor,
                            B_actions: torch.Tensor):
        actions, positions = B_actions
        for i, (action, pos) in enumerate(zip(actions, positions)):
            if action == "none":
                continue
            elif action == "merge":
                # remove the split symbol at pos
                states[i] = torch.cat([states[i][:pos], states[i][pos+1:]], dim=0)
            elif action == "untag":
                # change the pos-th symbol to a split symbol
                states[i][states[i]>=self.split_sym][pos] = self.split_sym # need to verify that this actually modifies states
        return states

    @torch.no_grad()
    def reverse_backward_actions(self,
                                states: torch.Tensor,
                                B_actions: torch.Tensor):
        actions, positions = B_actions
        F_actions = []
        F_positions = []
        F_tokens = []
        for i, (action, pos) in enumerate(zip(actions, positions)):
            if action == "none":
                F_actions.append('none')
                F_positions.append(0)
                F_tokens.append(0)
            elif action == "merge":
                F_actions.append('split')
                # return the index of the pos-th split symbol minus 1
                F_positions.append(torch.nonzero(states[i]==self.split_sym, as_tuple=True)[0][pos]-1)
                F_tokens.append(0)
            elif action == "untag":
                F_actions.append('tag')
                # return the index of the pos-th split symbol minus 1
                F_positions.append(torch.nonzero(states[i]==self.split_sym, as_tuple=True)[0][pos]-1)
                F_tokens.append(states[i][F_positions[-1]+1])
        return (F_actions, F_positions, F_tokens)
    

    @torch.no_grad()
    def calc_reward(self, to_calc):
        return self.calc_log_reward(to_calc).exp()

    # @torch.no_grad()
    # removing torch.no_grad here, since we can reuse this function as a loss for training of the PCFG and of the AR model

    def calc_log_reward(self, seqs):
        spans = []
        tag_seqs = []

        for seq in seqs:
            nt_positions = torch.nonzero(seq >= self.n_vocab)
            p = [-1] + list(nt_positions.cpu().numpy())
            spans += [ seq[p[i]+1:p[i+1]] for i in range(len(p)) ]
            tag_seqs.append(seq[nt_positions])

        x = torch.nn.utils.rnn.pad_sequence(spans, batch_first=True)
        lengths = torch.Tensor(list(map(len, spans))).to(x.device)
        tree_lls = self.pcfg.batch_marginal_with_roots(self, x, lengths, torch.cat(tag_seqs, 0))

        x_tag = torch.nn.utils.rnn.pad_sequence(tag_seqs)
        ar_lls = None # score x_tag using the AR model

        lr = torch.zeros((len(seqs),), device=x.device)
        start = 0
        for i, ar_ll, tag_seq in zip(range(len(seqs)), ar_lls, tag_seqs):
            lr[i] = ar_ll + tree_lls[start:start+len(tag_seq)]
            start += len(tag_seq)
    
        return lr