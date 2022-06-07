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
        assert x.max() < self.n_vocab+self.n_nts, f"input contains token id {x.max()}, but max is {self.n_vocab+self.n_nts-1}"
        encoded_tgt = self.embedding_tgt(x)
        pos_ids = create_position_ids(x)
        encoded_tgt += self.embedding_pos(pos_ids)
        return encoded_tgt

class GFlowNet_encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first, nlayers, batch_first=True, activation="relu", shared_embedding=None):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.embedding = shared_embedding
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                                batch_first=batch_first, activation=activation, norm_first=norm_first)
        encoder_norm = LayerNorm(d_model, eps=1e-5)
        self.model_encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

    def forward(self, x, pad_mask):
        encoded_src = self.embedding(x)
        memory = self.model_encoder(encoded_src, src_key_padding_mask=pad_mask)
        return memory

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

class segmenter_controller():
    def __init__(self, device, args, n_vocab=None, pcfg=None, ar_model=None):
        self.device = device
        self.args = args
        self.n_vocab = n_vocab
        self.pcfg = pcfg
        self.ar_model = ar_model
        self.split_sym = n_vocab
        if type(args) is dict:
            self.pad_sym = n_vocab + args['t_states'] + args['nt_states'] + 1
        else:
            self.pad_sym = n_vocab + args.t_states + args.nt_states + 1

    def sample_forward(self,
                    action : str,
                    F_logits: torch.Tensor,
                    states: torch.Tensor,
                    temperature_pos : float = 1.,
                    temperature_tok : float = 1.,):
        if type(states) is not list:
            # convert states to a list and remove padding
            states = [sent[sent!=self.pad_sym] for sent in states]
            convert_to_tensor = True
        else:
            convert_to_tensor = False
        F_actions = self._sample_forward_actions(action, F_logits, states, temperature_pos, temperature_tok)
        P_F = self.calc_forward_prob(F_logits=F_logits,
                                    F_actions=F_actions,
                                    states=states)
        B_actions = self.reverse_forward_actions(states, F_actions)
        states = self.apply_forward_actions(states, F_actions)
        if convert_to_tensor:
            states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True, padding_value=self.pad_sym).long()
        return states, F_actions, B_actions, P_F
    

    def calc_forward_prob(self,
                        F_logits: torch.Tensor,
                        states: list,
                        F_actions: tuple = None,):
        if type(states) is not list:
            # convert states to a list and remove padding
            states = [sent[sent!=self.pad_sym] for sent in states]
        logF_prob = []
        for i, state in enumerate(states):
            # if the action is none, the associated log probability is 0
            if F_actions[0][i] == 'none':
                logF_prob.append(torch.tensor(0., device=self.device))
                continue
            elif F_actions[0][i] == "split":
                mask = torch.zeros(state.size()).to(self.device)
                # mask[state>=self.split_sym] = -100
                split_idx = (state==self.split_sym).nonzero()
                mask[split_idx] = -100
                left_one = split_idx - 1
                left_one = left_one[left_one >= 0]
                mask[left_one] = -100
                left_two = split_idx - 2
                left_two = left_two[left_two >= 0]
                mask[left_two] = -100
                right_one = split_idx + 1
                right_one = right_one[right_one < state.size(-1)]
                mask[right_one] = -100
                # left_one_mask = torch.cat([state[1:], torch.zeros(1).to(self.device)-100], dim=0)
                # left_two_mask = torch.cat([state[2:], torch.zeros(2).to(self.device)-100], dim=0)
                # right_one_mask = torch.cat([torch.zeros(1).to(self.device)-100, state[:-1]], dim=0)
                # mask += left_one_mask + left_two_mask + right_one_mask
                mask[-1] = 0
                mask[-2] = -100
                mask[0] = -100
                # take softmax
                logP_pos = F.log_softmax(F_logits[i, :state.size(-1)]+mask, dim=-1)
                logF_prob.append(logP_pos[F_actions[1][i]])
            elif F_actions[0][i] == "tag":
                mask = torch.zeros(state.size()).to(self.device)
                mask[state!=self.split_sym] = -100
                # take softmax
                logP_pos = F.log_softmax(F_logits[i, :state.size(-1), 0]+mask, dim=-1)
                logP_tok = F.log_softmax(F_logits[i, F_actions[1][i], 1:], dim=-1)
                logF_prob.append(logP_pos[F_actions[1][i]]+logP_tok[F_actions[2][i]])
        return torch.stack(logF_prob, dim=0)

    def _sample_forward_actions(self,
                                action : str,
                                F_logits: torch.Tensor,
                                states: list,
                                temperature_pos : float = 1.,
                                temperature_tok : float = 1.,):
        actions = []
        positions = []
        tokens = []
        for i, state in enumerate(states):
            if action == "split":
                # if sf, add none to actions
                if state[state!=self.pad_sym][-1] == self.split_sym:
                    actions.append('none')
                    positions.append(0)
                    tokens.append(0)
                    continue
                # otherwise, sample a split symbol to delete by first constructing a mask
                # There are three rules for masking
                # 1. mask every split symbol
                # 2. mask the two tokens before and the one tokens after every split symbol
                # 3. the last token is always unmasked
                mask = torch.zeros(state.size()).to(self.device)
                split_idx = (state==self.split_sym).nonzero()
                mask[split_idx] = -100
                left_one = split_idx - 1
                left_one = left_one[left_one >= 0]
                mask[left_one] = -100
                left_two = split_idx - 2
                left_two = left_two[left_two >= 0]
                mask[left_two] = -100
                right_one = split_idx + 1
                right_one = right_one[right_one < state.size(-1)]
                mask[right_one] = -100
                # left_one_mask = torch.cat([state[1:], torch.zeros(1).to(self.device)-100], dim=0)
                # left_two_mask = torch.cat([state[2:], torch.zeros(2).to(self.device)-100], dim=0)
                # right_one_mask = torch.cat([torch.zeros(1).to(self.device)-100, state[:-1]], dim=0)
                # mask += left_one_mask + left_two_mask + right_one_mask
                mask[-1] = 0
                mask[-2] = -100
                mask[0] = -100
                # mask[mask!=0] = -100
                # take softmax over positions
                P_pos = F.softmax(F_logits[i, :state.size(-1)]/temperature_pos+mask, dim=-1)
                F_pos = torch.multinomial(P_pos, 1).item()
                positions.append(F_pos)
                tokens.append(0)
            elif action == "tag":
                # if sf, add none to actions
                if (state[state!=self.pad_sym][-1] >= self.split_sym) and (state[state==self.split_sym].sum().item() == 0):
                    actions.append('none')
                    positions.append(0)
                    tokens.append(0)
                    continue
                # otherwise, pick a tag to reverse back to a split symbol
                mask = torch.zeros(state.size()).to(self.device)
                mask[state!=self.split_sym] = -100
                # take softmax
                P_pos = F.softmax(F_logits[i, :state.size(-1), 0]/temperature_pos+mask, dim=-1)
                F_pos = torch.multinomial(P_pos, 1).item()
                positions.append(F_pos)
                # take softmax over tokens
                P_tok = F.softmax(F_logits[i, F_pos, 1:]/temperature_tok, dim=-1)
                F_tok = torch.multinomial(P_tok, 1).item()
                tokens.append(F_tok)
            actions.append(action)
        return (actions, positions, tokens)

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
                states[i] = torch.cat([states[i][:pos+1], torch.zeros(1).to(self.device)+self.split_sym, states[i][pos+1:]], dim=0).long()
            elif action == "tag":
                # change the pos-th split symbol to tok
                states[i][pos] = tok+self.n_vocab+1 # need to verify that this actually modifies states
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
                B_positions.append(pos+1)
            elif action == "tag":
                B_actions.append('untag')
                # return the index of the split symbol at pos
                B_positions.append(pos)
        return (B_actions, B_positions)

    def sample_backward(self,
                        action : str,
                        B_logits: torch.Tensor,
                        states: torch.Tensor,
                        temperature_pos : float = 1.):
        if type(states) is not list:
            # convert states to a list and remove padding
            states = [sent[sent!=self.pad_sym] for sent in states]
            convert_to_tensor = True
        else:
            convert_to_tensor = False
        B_actions = self._sample_backward_actions(action, B_logits, states, temperature_pos)
        P_B = self.calc_backward_prob(B_logits=B_logits,
                                    B_actions=B_actions,
                                    states=states)
        F_actions = self.reverse_backward_actions(states, B_actions)
        states = self.apply_backward_actions(states, B_actions)
        if convert_to_tensor:
            states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True, padding_value=self.pad_sym).long()
        return states, B_actions, F_actions, P_B

    def calc_backward_prob(self,
                        B_logits: torch.Tensor,
                        states: list,
                        B_actions: tuple):
        if type(states) is not list:
            # convert states to a list and remove padding
            states = [sent[sent!=self.pad_sym] for sent in states]
        logB_prob = []
        for i, state in enumerate(states):
            # if the action is none, the associated log probability is 0
            if B_actions[0][i] == 'none':
                logB_prob.append(torch.tensor(0., device=self.device))
                continue
            elif B_actions[0][i] == "merge":
                if state[state!=self.pad_sym][-1] == self.split_sym:
                    logB_prob.append(torch.tensor(0., device=self.device))
                    continue
                mask = torch.zeros(state.size()).to(self.device)
                mask[state!=self.split_sym] = -100
            elif B_actions[0][i] == "untag":
                mask = torch.zeros(state.size()).to(self.device)
                mask[state<=self.split_sym] = -100
            # take softmax
            lpgP_pos = F.log_softmax(B_logits[i, :state.size(-1)]+mask, dim=-1)
            logB_prob.append(lpgP_pos[B_actions[1][i]])
        return torch.stack(logB_prob, dim=0)

    def _sample_backward_actions(self,
                                action : str,
                                B_logits: torch.Tensor,
                                states: list,
                                temperature_pos : float = 1.,):
        actions = []
        positions = []
        for i, state in enumerate(states):
            if action == "merge":
                # if s0, add none to actions
                if state[state>=self.split_sym].sum().item() == 0:
                    actions.append('none')
                    positions.append(0)
                    continue
                # otherwise, sample a split symbol to delete by first constructing a mask
                mask = torch.zeros(state.size()).to(self.device)
                mask[state!=self.split_sym] = -100
                # take softmax
                P_pos = F.softmax(B_logits[i, :state.size(-1)]/temperature_pos+mask, dim=-1)
                B_pos = torch.multinomial(P_pos, 1).item()
                positions.append(B_pos)
            elif action == "untag":
                # if s0, add none to actions
                if state[state>self.split_sym].sum().item() == 0:
                    actions.append('none')
                    positions.append(0)
                    continue
                # otherwise, pick a tag to reverse back to a split symbol
                mask = torch.zeros(state.size()).to(self.device)
                mask[state<=self.split_sym] = -100
                # take softmax
                P_pos = F.softmax(B_logits[i, :state.size(-1)]/temperature_pos+mask, dim=-1)
                B_pos = torch.multinomial(P_pos, 1).item()
                positions.append(B_pos)
            actions.append(action)
        return (actions, positions)

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
                states[i][pos] = self.split_sym
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
                # return the index of the pos-th split symbol
                F_positions.append(pos-1)
                F_tokens.append(0)
            elif action == "untag":
                F_actions.append('tag')
                # return the index of the pos-th split symbol
                F_positions.append(pos)
                F_tokens.append(states[i][pos].item()-self.n_vocab-1)
        return (F_actions, F_positions, F_tokens)
    

    @torch.no_grad()
    def calc_reward(self, to_calc):
        return self.calc_log_reward(to_calc).exp()

    # @torch.no_grad()
    # removing torch.no_grad here, since we can reuse this function as a loss for training of the PCFG and of the AR model

    def calc_log_reward(self, seqs):
        spans = []
        tag_seqs = []
        seqs = [sent[sent!=self.pad_sym] for sent in seqs]
        for seq in seqs:
            nt_positions = torch.nonzero(seq > self.n_vocab)
            p = [-1] + nt_positions.cpu().numpy().flatten().tolist()
            spans += [ seq[p[i]+1:p[i+1]] for i in range(len(p) - 1) ]
            # import pdb; pdb.set_trace();
            tag_seqs.append(seq[nt_positions].flatten() - self.n_vocab - 1)

        x = torch.nn.utils.rnn.pad_sequence(spans, batch_first=True)
        lengths = torch.Tensor(list(map(len, spans))).to(x.device).long()
        # import pdb;pdb.set_trace();
        # tree_lls = self.pcfg.batch_marginal_with_roots(x, lengths, torch.cat(tag_seqs, 0))

        # pad_value = self.args.nt_states+1
        # start_end = torch.LongTensor([self.args.nt_states]).to(x.device)
        # x_tag = torch.nn.utils.rnn.pad_sequence([torch.cat([start_end,seq,start_end],0) for seq in tag_seqs], batch_first=True, padding_value=pad_value)
        # outs = self.ar_model(x_tag).log_softmax(-1)
        # token_lls = outs[:,:-1].gather(2, x_tag[:,1:].unsqueeze(2)).squeeze(2)
        # ar_lls = (token_lls * (x_tag[:,1:]!=pad_value).float()).sum(1)

        # print('ts', seqs[0], x_tag[0])
        # import pdb;pdb.set_trace()

        lr = torch.zeros((len(seqs),), device=x.device)
        start = 0
        for i, tag_seq in zip(range(len(seqs)), tag_seqs):
            lr[i] = lengths[start:start+len(tag_seq)].float().std()
            start += len(tag_seq)
        # import pdb; pdb.set_trace();
        return -torch.abs(torch.nan_to_num(lr)) # stdev for a single number returns nan so set it to 0.