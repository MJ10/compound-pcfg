import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PCFG import PCFG
from random import shuffle
from torch_struct import SentCFG
from utils import *

class ResidualLayer(nn.Module):
  def __init__(self, in_dim = 100,
               out_dim = 100):
    super(ResidualLayer, self).__init__()
    self.lin1 = nn.Linear(in_dim, out_dim)
    self.lin2 = nn.Linear(out_dim, out_dim)

  def forward(self, x):
    return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class CompPCFG(nn.Module):
  def __init__(self, vocab = 100,
               h_dim = 512, 
               w_dim = 512,
               z_dim = 64,
               state_dim = 256, 
               t_states = 10,
               nt_states = 10):
    super(CompPCFG, self).__init__()
    self.state_dim = state_dim
    self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))
    self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
    self.root_emb = nn.Parameter(torch.randn(1, state_dim))
    self.pcfg = PCFG(nt_states, t_states)
    self.nt_states = nt_states
    self.t_states = t_states
    self.all_states = nt_states + t_states
    self.dim = state_dim
    self.register_parameter('t_emb', self.t_emb)
    self.register_parameter('nt_emb', self.nt_emb)
    self.register_parameter('root_emb', self.root_emb)
    self.rule_mlp = nn.Linear(state_dim+z_dim, self.all_states**2)
    self.root_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),                         
                                  nn.Linear(state_dim, self.nt_states))
    if z_dim > 0:
      self.enc_emb = nn.Embedding(vocab, w_dim)
      self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers = 1, batch_first = True)
      self.enc_params = nn.Linear(h_dim*2, z_dim*2)
    self.z_dim = z_dim
    self.vocab_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   nn.Linear(state_dim, vocab))
      
  def enc(self, x):
    emb = self.enc_emb(x)
    h, _ = self.enc_rnn(emb)    
    params = self.enc_params(h.max(1)[0])
    mean = params[:, :self.z_dim]
    logvar = params[:, self.z_dim:]    
    return mean, logvar

  def kl(self, mean, logvar):
    result =  -0.5 * (logvar - torch.pow(mean, 2)- torch.exp(logvar) + 1)
    return result

  def forward(self, x, lengths, argmax=False, use_mean=False):
    #x : batch x n
    n = x.size(1)
    batch_size = x.size(0)
    if self.z_dim > 0:
      mean, logvar = self.enc(x)
      kl = self.kl(mean, logvar).sum(1) 
      z = mean.new(batch_size, mean.size(1)).normal_(0, 1)
      z = (0.5*logvar).exp()*z + mean    
      kl = self.kl(mean, logvar).sum(1) 
      if use_mean:
        z = mean
      self.z = z
    else:
      self.z = torch.zeros(batch_size, 1).cuda()

    t_emb = self.t_emb
    nt_emb = self.nt_emb
    root_emb = self.root_emb

    root_emb = root_emb.expand(batch_size, self.state_dim)
    t_emb = t_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, n, self.t_states, self.state_dim)
    nt_emb = nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim)

    if self.z_dim > 0:
      root_emb = torch.cat([root_emb, z], 1)
      z_expand = z.unsqueeze(1).expand(batch_size, n, self.z_dim)
      z_expand = z_expand.unsqueeze(2).expand(batch_size, n, self.t_states, self.z_dim)
      t_emb = torch.cat([t_emb, z_expand], 3)
      nt_emb = torch.cat([nt_emb, z.unsqueeze(1).expand(batch_size, self.nt_states, 
                                                         self.z_dim)], 2)
    root_scores = F.log_softmax(self.root_mlp(root_emb), 1)
    unary_scores = F.log_softmax(self.vocab_mlp(t_emb), 3)
    x_expand = x.unsqueeze(2).expand(batch_size, x.size(1), self.t_states).unsqueeze(3)
    unary = torch.gather(unary_scores, 3, x_expand).squeeze(3)
    rule_score = F.log_softmax(self.rule_mlp(nt_emb), 2) # nt x t**2
    rule_scores = rule_score.view(batch_size, self.nt_states, self.all_states, self.all_states)
    params = (unary, rule_scores, root_scores)
    dist = SentCFG(params, lengths=torch.tensor(lengths))
    log_Z = dist.partition
    if self.z_dim == 0:
      kl = torch.zeros_like(log_Z)
    if argmax:
      with torch.no_grad():
        spans = dist.argmax[-1]
        argmax_spans, argmax_trees = extract_parses(spans, lengths, inc=1)
      return -log_Z, kl, (argmax_trees, argmax_spans)
    else:
      return -log_Z, kl

  def batch_marginal_with_roots(self, x, lengths, tags):
    # x : (bs,l) the padded spans
    # lengths : (l,) their lengths
    # tags : (l,) the root NTs

    batch_size, l = x.shape

    t_emb = self.t_emb
    nt_emb = self.nt_emb

    t_emb = t_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, n, self.t_states, self.state_dim)
    nt_emb = nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim)

    root_scores = torch.nn.functional.one_hot(tags, self.nt_states).float().clamp(min=1e-32).log()
    unary_scores = self.vocab_mlp(t_emb).log_softmax(1)
    x_expand = x.unsqueeze(2).expand(batch_size, x.size(1), self.t_states).unsqueeze(3)
    unary = unary_scores.gather(3, x_expand).squeeze(3)

    rule_score = self.rule_mlp(nt_emb).log_softmax(2)
    rule_scores = rule_score.view(batch_size, self.nt_states, self.all_states, self.all_states)
    params = (unary, rule_scores, root_scores)
    dist = SentCFG(params, lengths=lengths)
    log_Z = dist.partition

    return log_Z

class ARModel(nn.Module):
  def __init__(self, V, num_layers, hidden_dim):
    super(ARModel, self).__init__()

    self.embedding = nn.Embedding(V, hidden_dim)
    self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers)
    self.unembedding = nn.Linear(hidden_dim, V)

  def forward(self,x):
    x = self.embedding(x)
    x, _ = self.lstm(x)
    return self.unembedding(x)