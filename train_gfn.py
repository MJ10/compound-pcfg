#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset, MinimalDataset
from utils import *
from models import CompPCFG, ARModel
from torch.nn.init import xavier_uniform_
from gflownet import *

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--data', default='data/pcfg_1')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')

# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--batch_size', default=4, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')
parser.add_argument('--minimal_dataloader', action="store_true")

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.minimal_dataloader:
    corpus = MinimalDataset(args.data, 1000, batch_size=args.batch_size,
                              batch_group_size=999999, add_master_token=False)
    train_data = corpus.train
    train_lens = corpus.train_lens
    # print(train_data[0])
    # import pdb;pdb.set_trace();
    val_data = corpus.valid
    val_lens = corpus.valid_lens
    vocab_size = len(corpus.dict.idx2word)  
    max_len = train_data.shape[1]
    print('Train: %d sents / %d batches, Val: %d sents / %d batches' % 
        (train_data.size(0), len(train_data), val_data.size(0), len(val_data)))
  else:
    train_data = Dataset(args.train_file)
    val_data = Dataset(args.val_file)  
    train_sents = train_data.batch_size.sum()
    vocab_size = int(train_data.vocab_size)
    max_len = max(val_data.sents.size(1), train_data.sents.size(1))
  
    print('Train: %d sents / %d batches, Val: %d sents / %d batches' % 
          (train_data.sents.size(0), len(train_data), val_data.sents.size(0), len(val_data)))
  print('Vocab size: %d, Max Sent Len: %d' % (vocab_size, max_len))
  print('Save Path', args.save_path)
  cuda.set_device(args.gpu)
  model = CompPCFG(vocab = vocab_size,
                   state_dim = args.state_dim,
                   t_states = args.t_states,
                   nt_states = args.nt_states,
                   h_dim = args.h_dim,
                   w_dim = args.w_dim,
                   z_dim = args.z_dim)
  ar_model = ARModel(V = args.t_states + args.nt_states + 2,
                     num_layers = 2,
                     hidden_dim = 128)
  gfn_Z = GFlowNet_Z(args.state_dim)
  gfn_emb = GFlowNet_shared_embedding(vocab_size, args.state_dim, 60, args.nt_states)
  gfn_encoder = GFlowNet_encoder(args.state_dim, 4, 4*args.state_dim, 0.1, True, 4, shared_embedding=gfn_emb)
  gfn_forward_split = GFlowNet_forward_split(args.state_dim)
  gfn_forward_tag = GFlowNet_forward_tag(args.nt_states, args.state_dim)
  gfn_backward = GFlowNet_backward(args.state_dim)
  controller = segmenter_controller(args.gpu, args, vocab_size, model, ar_model)
  for name, param in model.named_parameters():    
    if param.dim() > 1:
      xavier_uniform_(param)
  print("model architecture")
  print(model)
  model.train()
  model.cuda()
  ar_model.train()
  ar_model.cuda()
  optimizer = torch.optim.Adam({'params':model.parameters(), 'lr':args.lr, 'betas':(args.beta1, args.beta2)},
                               {'params':ar_model.parameters(), 'lr':args.lr, 'betas':(args.beta1, args.beta2)})
  gfn_Z.train()
  gfn_Z.cuda()
  gfn_encoder.train()
  gfn_encoder.cuda()
  gfn_forward_split.train()
  gfn_forward_split.cuda()
  gfn_forward_tag.train()
  gfn_forward_tag.cuda()
  gfn_backward.train()
  gfn_backward.cuda()
  gfn_optimizer = torch.optim.Adam({'params':gfn_Z.parameters(), 'lr':args.lr, 'betas':(args.beta1, args.beta2)},
                                  {'params':gfn_encoder.parameters(), 'lr':args.lr, 'betas':(args.beta1, args.beta2)},
                                  {'params':gfn_forward_split.parameters(), 'lr':args.lr, 'betas':(args.beta1, args.beta2)},
                                  {'params':gfn_forward_tag.parameters(), 'lr':args.lr, 'betas':(args.beta1, args.beta2)},
                                  {'params':gfn_backward.parameters(), 'lr':args.lr, 'betas':(args.beta1, args.beta2)})

  best_val_ppl = 1e5
  best_val_f1 = 0
  epoch = 0
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    train_kl = 0.
    train_gfn_logR = 0.
    train_gfn_tb = 0.
    train_gfn_logZ = 0.
    num_sents = 0.
    num_words = 0.
    all_stats = [[0., 0., 0.]]
    b = 0
    for i in np.random.permutation(len(train_data)):
      b += 1
      if args.minimal_dataloader:
        sents = train_data[i]
        lengths = train_lens[i]
        batch_size = args.batch_size

      else:
        sents, length, batch_size, _, gold_spans, gold_binary_trees, _ = train_data[i]      
        
        if length > args.max_length or length == 1: #length filter based on curriculum 
          continue

        lengths = [length]*batch_size

      sents = sents.cuda()
      num_sents += batch_size
      num_words += sum(lengths)+len(lengths)

      # sample GFlowNet for sequence
      logZ, logPF, logPB, sents = sample_gfn(sents)
      state = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True, padding_value=vocab_size+args.t_states+args.nt_states)
      logR = controller.calc_log_reward(state)
      tb_loss = (logZ + logPF - logPB - logR.detach()**2).mean()

      gfn_optimizer.zero_grad()
      tb_loss.backward(retain_graph=True)
      gfn_optimizer.step()

      optimizer.zero_grad()
      (-logR.sum()).backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)    
      optimizer.step()

      train_gfn_tb += tb_loss.item()
      train_gfn_logR += logR.sum().item()
      train_gfn_logZ += logZ.sum(0).item()

      # if not args.minimal_dataloader:
      #   for bb in range(batch_size):
      #     span_b = [(a[0], a[1]) for a in argmax_spans[bb]] #ignore labels
      #     span_b_set = set(span_b[:-1])
      #     update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
      if b % args.print_every == 0:
        # if not args.minimal_dataloader:
        #   all_f1 = get_f1(all_stats)
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        gparam_norm = sum([p.grad.norm()**2 for p in model.parameters() 
                           if p.grad is not None]).item()**0.5

        log_str = 'Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                  'GFN TB: %.4f, logR: %.4f, logZ: %.4f' + \
                  'Throughput: %.2f examples/sec'
        print(log_str %
              (epoch, b, len(train_data), param_norm, gparam_norm, args.lr, 
              train_gfn_tb / num_sents, train_gfn_logR / num_sents, train_gfn_logZ / num_sents,
              num_sents / (time.time() - start_time)))
       
        # # print an example parse
        # # tree = get_tree_from_binary_matrix(binary_matrix[0], length)
        # tree = trees[0]
        # action = get_actions(tree)
        # sent_str = [corpus.dict.idx2word[word_idx] for word_idx in list(sents[0].cpu().numpy())]
        # print("Pred Tree: %s" % get_tree(action, sent_str))
        # if not args.minimal_dataloader:
        #   print("Gold Tree: %s" % get_tree(gold_binary_trees[0], sent_str))

    args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
    print('--------------------------------')
    print('Checking validation perf...')    
    val_ppl, val_f1 = eval(val_data, val_lens, model)
    print('--------------------------------')
    if val_ppl < best_val_ppl:
      best_val_ppl = val_ppl
      # best_val_f1 = val_f1
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
        'word2idx': corpus.dict.word2idx,
        'idx2word': corpus.dict.idx2word
      }
      print('Saving checkpoint to %s' % args.save_path)
      torch.save(checkpoint, args.save_path)
      model.cuda()

def eval(data, data_lens, model):
  model.eval()
  ar_model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  total_kl = 0.
  corpus_f1 = [0., 0., 0.] 
  sent_f1 = [] 
  with torch.no_grad():
    for i in range(len(data)):
      if args.minimal_dataloader:
        sents = data[i]
        lengths = data_lens[i]
        batch_size = args.batch_size
       
      else:
        sents, length, batch_size, _, gold_spans, gold_binary_trees, other_data = data[i] 
        if length == 1:
          continue
        lengths = [lengths] * batch_size
      
      # note that for unsuperised parsing, we should do model(sents, argmax=True, use_mean = True)
      # but we don't for eval since we want a valid upper bound on PPL for early stopping
      # see eval.py for proper MAP inference
      num_sents += batch_size
      num_words += sum(lengths)+len(lengths)

      state = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True, padding_value=vocab_size+args.t_states+args.nt_states)
      # sample GFlowNet for sequence
      logZ, logPF, logPB, state = sample_gfn(state)
      logR = gfn.calc_log_reward(state)

      total_nll += logR.sum().item()
      # if not args.minimal_dataloader:
      #   for b in range(batch_size):
      #     span_b = [(a[0], a[1]) for a in argmax_spans[b]] #ignore labels
      #     span_b_set = set(span_b[:-1])        
      #     gold_b_set = set(gold_spans[b][:-1])
      #     tp, fp, fn = get_stats(span_b_set, gold_b_set) 
      #     corpus_f1[0] += tp
      #     corpus_f1[1] += fp
      #     corpus_f1[2] += fn
      #     # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py

      #     model_out = span_b_set
      #     std_out = gold_b_set
      #     overlap = model_out.intersection(std_out)
      #     prec = float(len(overlap)) / (len(model_out) + 1e-8)
      #     reca = float(len(overlap)) / (len(std_out) + 1e-8)
      #     if len(std_out) == 0:
      #       reca = 1. 
      #       if len(model_out) == 0:
      #         prec = 1.
      #     f1 = 2 * prec * reca / (prec + reca + 1e-8)
      #     sent_f1.append(f1)
  # if not args.minimal_dataloader:
  #   tp, fp, fn = corpus_f1  
  #   prec = tp / (tp + fp)
  #   recall = tp / (tp + fn)
  #   corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  #   sent_f1 = np.mean(np.array(sent_f1))
  recon_ppl = np.exp(total_nll / num_words)
  ppl_elbo = np.exp((total_nll + total_kl)/num_words) 
  # kl = total_kl /num_sents
  print('ReconPPL: %.2f, PPL (Upper Bound): %.2f' %
        (recon_ppl, ppl_elbo))
  # if not args.minimal_dataloader:
  #   print('Corpus F1: %.2f, Sentence F1: %.2f' %
  #         (corpus_f1*100, sent_f1*100))
  model.train()
  ar_model.train()
  return ppl_elbo, sent_f1*100 if not args.minimal_dataloader else 0

def sample_gfn(sents, controller, gfn_Z, gfn_encoder, gfn_forward_split, gfn_forward_tag, gfn_backward):
  def done_splitting(sent):
    if sent[-1] == args.vocab_size:
      return True
    return False
  def done_tagging(sent):
    if sent[sent==args.vocab_size].sum() == 0:
      return True
    return False
  logPF = torch.zeros(sents.shape[0], device=args.device)
  logPB = torch.zeros(sents.shape[0], device=args.device)
  # Phase I:
  #  - get logZ
  logZ = gfn_Z(sents)
  # Phase II:
  #  - split the seqs until the last token is a split symbol
  encoded_sents = gfn_encoder(sents)
  while not all([done_splitting(sent) for sent in sents]):
    _, _, B_actions, _logPF = \
                controller.sample_forward('split',
                                          gfn_forward_split(encoded_sents),
                                          sents)
    encoded_sents = gfn_encoder(sents)
    _logPB = controller.calc_backward_prob(gfn_backward(encoded_sents),
                                          new_sents,
                                          B_actions)
    encoded_sents = gfn_encoder(sents)
    logPF += _logPF
    logPB += _logPB
  # Phase III:
  #  - tag all the split symbols until there are none left
  encoded_sents = gfn_encoder(sents)
  while not all([done_tagging(sent) for sent in sents]):
    new_sents, _, B_actions, _logPF = \
                controller.sample_forward('tag',
                                          gfn_forward_tag(encoded_sents),
                                          sents)
    encoded_sents = gfn_encoder(sents)
    _logPB = controller.calc_backward_prob(gfn_backward(encoded_sents),
                                          new_sents,
                                          B_actions)
    encoded_sents = gfn_encoder(sents)
    logPF += _logPF
    logPB += _logPB
  return logZ, logPF, logPB, sents

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
