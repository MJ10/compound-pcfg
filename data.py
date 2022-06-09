#!/usr/bin/env python3
import numpy as np
import torch
import os
from io import open
import random
import pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

# [corpus.dictionary.idx2word[idx] for idx in data[:40].tolist()]
from collections import defaultdict

class HFTokenizedDataset(object):
  def __init__(self, path, HF_tokenizer, seqlen=-1, batch_size=1, batch_group_size=100, device='cuda', pad_value=0):
    self.train = None
    self.valid = None
    self.test = None
    self.device = device
    self.seqlen = seqlen
    self.batch_size = batch_size
    self.batch_group_size = batch_group_size
    self.pad_value = pad_value
    self.HF_tokenizer = HF_tokenizer
    if not self.load_cache(path):
      self.train, self.train_lens = self.tokenize(os.path.join(path, 'train'))
      self.valid, self.valid_lens = self.tokenize(os.path.join(path, 'valid'))
      self.test, self.test_lens = self.tokenize(os.path.join(path, 'test'))
      self.train = torch.nn.utils.rnn.pad_sequence(self.train, batch_first=True, padding_value=self.pad_value)
      self.valid = torch.nn.utils.rnn.pad_sequence(self.valid, batch_first=True, padding_value=self.pad_value)
      self.test = torch.nn.utils.rnn.pad_sequence(self.test, batch_first=True, padding_value=self.pad_value)
      self.save_cache(path)
    for loader in ['train', 'valid', 'test']:
      tmp_copy = getattr(self, loader)
      tmp_copy = tmp_copy[:tmp_copy.size(0)//self.batch_size*self.batch_size]
      setattr(self, loader, tmp_copy.reshape(tmp_copy.size(0) // self.batch_size, self.batch_size, tmp_copy.size(-1)))

      tmp_copy = getattr(self, loader+"_lens")
      tmp_copy = tmp_copy[:tmp_copy.size(0)//self.batch_size*self.batch_size]
      setattr(self, loader + "_lens", tmp_copy.reshape(tmp_copy.size(0) // self.batch_size, self.batch_size))
    # import pdb; pdb.set_trace();

  def sort_n_shuffle(self, dataloader):
    dataloader = sorted(dataloader, key=lambda x:len(x))
    groups = []
    for i, sample in enumerate(dataloader, 0):
      if i % (self.batch_size * self.batch_group_size) == 0:
        groups.append([])
      groups[-1].append(sample)
    for group in groups:
      random.shuffle(group)
    dataloader = [ele for group in groups for ele in group]
    lens = [len(ele) for group in groups for ele in group]
    return dataloader, lens
    
  def load_cache(self, path):
    suffix = f'.{self.seqlen}' if self.seqlen > 0 else ''
    suffix += '.no_MT'
    for cache in ['train.pt', 'valid.pt', 'test.pt']:
      cache_path = os.path.join(path, cache+suffix)
      if not os.path.exists(cache_path):
        return False
    self.train = torch.load(os.path.join(path, f'train.pt{suffix}'))
    self.train_lens = torch.load(os.path.join(path, f'train_lens.pt{suffix}'))
    self.valid = torch.load(os.path.join(path, f'valid.pt{suffix}'))
    self.valid_lens = torch.load(os.path.join(path, f'valid_lens.pt{suffix}'))
    self.test = torch.load(os.path.join(path, f'test.pt{suffix}'))
    self.test_lens = torch.load(os.path.join(path, f'test_lens.pt{suffix}'))
    return True

  def save_cache(self, path):
    suffix = f'.{self.seqlen}' if self.seqlen > 0 else ''
    suffix += '.no_MT'
    torch.save(self.train, os.path.join(path, f'train.pt{suffix}'))
    torch.save(self.train_lens, os.path.join(path, f'train_lens.pt{suffix}'))
    torch.save(self.valid, os.path.join(path, f'valid.pt{suffix}'))
    torch.save(self.valid_lens, os.path.join(path, f'valid_lens.pt{suffix}'))
    torch.save(self.test, os.path.join(path, f'test.pt{suffix}'))        
    torch.save(self.test_lens, os.path.join(path, f'test_lens.pt{suffix}'))        

  def tokenize(self, path):
    """Tokenizes a text file."""
    src_path = path + '.src'
    assert os.path.exists(src_path)
    with open(src_path, 'r', encoding="utf8") as src_f:
      src_idss = []
      for src_line in src_f:
        src_line = src_line.strip('\n')
        src_words = src_line.split()
        if self.seqlen > 0 and (len(src_words) > self.seqlen):
          continue
        src_ids = self.HF_tokenizer(src_line, return_tensors='pt',
                                    add_special_tokens=True)['input_ids'][0]
        src_ids[src_ids==2] = 1
        src_idss.append(src_ids)
    src_idss, lengths = self.sort_n_shuffle(src_idss)
    lengths = torch.tensor(lengths)
    return src_idss, lengths

class MinimalDataset(object):
  def __init__(self, path, seqlen=-1, batch_size=1, batch_group_size=100, add_master_token=False, device='cuda', pad_value=0):
    self.dict = Dictionary()
    self.train = None
    self.valid = None
    self.test = None
    self.device = device
    self.seqlen = seqlen
    self.batch_size = batch_size
    self.batch_group_size = batch_group_size
    self.add_master_token = add_master_token
    self.pad_value = pad_value
    if not self.load_cache(path):
      self.train, self.train_lens = self.tokenize(os.path.join(path, 'train'))
      self.valid, self.valid_lens = self.tokenize(os.path.join(path, 'valid'))
      self.test, self.test_lens = self.tokenize(os.path.join(path, 'test'))
      self.train = torch.nn.utils.rnn.pad_sequence(self.train, batch_first=True, padding_value=self.pad_value + len(self.dict.idx2word)+1)
      self.valid = torch.nn.utils.rnn.pad_sequence(self.valid, batch_first=True, padding_value=self.pad_value + len(self.dict.idx2word)+1)
      self.test = torch.nn.utils.rnn.pad_sequence(self.test, batch_first=True, padding_value=self.pad_value + len(self.dict.idx2word)+1)
      self.save_cache(path)
    for loader in ['train', 'valid', 'test']:
      tmp_copy = getattr(self, loader)
      tmp_copy = tmp_copy[:tmp_copy.size(0)//self.batch_size*self.batch_size]
      setattr(self, loader, tmp_copy.reshape(tmp_copy.size(0) // self.batch_size, self.batch_size, tmp_copy.size(-1)))

      tmp_copy = getattr(self, loader+"_lens")
      tmp_copy = tmp_copy[:tmp_copy.size(0)//self.batch_size*self.batch_size]
      setattr(self, loader + "_lens", tmp_copy.reshape(tmp_copy.size(0) // self.batch_size, self.batch_size))
    # import pdb; pdb.set_trace();

  def sort_n_shuffle(self, dataloader):
    dataloader = sorted(dataloader, key=lambda x:len(x))
    groups = []
    for i, sample in enumerate(dataloader, 0):
      if i % (self.batch_size * self.batch_group_size) == 0:
        groups.append([])
      groups[-1].append(sample)
    for group in groups:
      random.shuffle(group)
    dataloader = [ele for group in groups for ele in group]
    lens = [len(ele) for group in groups for ele in group]
    return dataloader, lens
    
  def load_cache(self, path):
    suffix = f'.{self.seqlen}' if self.seqlen > 0 else ''
    suffix += '.no_MT' if not self.add_master_token else ''
    for cache in ['train.pt', 'valid.pt', 'test.pt', 'dict.pt']:
      cache_path = os.path.join(path, cache+suffix)
      if not os.path.exists(cache_path):
        return False
    self.dict = torch.load(os.path.join(path, f'dict.pt{suffix}'))
    self.train = torch.load(os.path.join(path, f'train.pt{suffix}'))
    self.train_lens = torch.load(os.path.join(path, f'train_lens.pt{suffix}'))
    self.valid = torch.load(os.path.join(path, f'valid.pt{suffix}'))
    self.valid_lens = torch.load(os.path.join(path, f'valid_lens.pt{suffix}'))
    self.test = torch.load(os.path.join(path, f'test.pt{suffix}'))
    self.test_lens = torch.load(os.path.join(path, f'test_lens.pt{suffix}'))
    return True

  def save_cache(self, path):
    suffix = f'.{self.seqlen}' if self.seqlen > 0 else ''
    suffix += '.no_MT' if not self.add_master_token else ''
    torch.save(self.dict, os.path.join(path, f'dict.pt{suffix}'))
    torch.save(self.train, os.path.join(path, f'train.pt{suffix}'))
    torch.save(self.train_lens, os.path.join(path, f'train_lens.pt{suffix}'))
    torch.save(self.valid, os.path.join(path, f'valid.pt{suffix}'))
    torch.save(self.valid_lens, os.path.join(path, f'valid_lens.pt{suffix}'))
    torch.save(self.test, os.path.join(path, f'test.pt{suffix}'))        
    torch.save(self.test_lens, os.path.join(path, f'test_lens.pt{suffix}'))        

  def tokenize(self, path):
    """Tokenizes a text file."""
    src_path = path + '.src'
    trees_path = path + '-trees.pkl'
    assert os.path.exists(src_path)
    # Add words to the dictionary
    with open(src_path, 'r', encoding="utf8") as src_f:
      src_idss = []
      for src_line in src_f:
        src_ids = []
        if self.add_master_token:
          src_words = src_line.split()
        else:
          src_words = src_line.split()
        if self.seqlen > 0 and (len(src_words) > self.seqlen):
          continue
        for word in src_words:
          self.dict.add_word(word)
          src_ids.append(self.dict.word2idx[word])
        src_idss.append(torch.tensor(src_ids, device=self.device).type(torch.int64))
    src_idss, lengths = self.sort_n_shuffle(src_idss)
    print(len(self.dict.idx2word), self.pad_value, self.pad_value + len(self.dict.idx2word)+1)
    # src_idss = torch.nn.utils.rnn.pad_sequence(src_idss, batch_first=True, padding_value=self.pad_value + len(self.dict.idx2word)+1)
    lengths = torch.tensor(lengths)
    return src_idss, lengths




class Dataset(object):
  def __init__(self, data_file):
    data = pickle.load(open(data_file, 'rb')) #get text data
    self.sents = self._convert(data['source']).long()
    self.other_data = data['other_data']
    self.sent_lengths = self._convert(data['source_l']).long()
    self.batch_size = self._convert(data['batch_l']).long()
    self.batch_idx = self._convert(data['batch_idx']).long()
    self.vocab_size = data['vocab_size'][0]
    self.num_batches = self.batch_idx.size(0)
    self.word2idx = data['word2idx']
    self.idx2word = data['idx2word']

  def _convert(self, x):
    return torch.from_numpy(np.asarray(x))

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    assert(idx < self.num_batches and idx >= 0)
    start_idx = self.batch_idx[idx]
    end_idx = start_idx + self.batch_size[idx]
    length = self.sent_lengths[idx].item()
    sents = self.sents[start_idx:end_idx]
    other_data = self.other_data[start_idx:end_idx]
    sent_str = [d[0] for d in other_data]
    tags = [d[1] for d in other_data]
    actions = [d[2] for d in other_data]
    binary_tree = [d[3] for d in other_data]
    spans = [d[5] for d in other_data]
    batch_size = self.batch_size[idx].item()
    # original data includes </s>, which we don't need
    data_batch = [sents[:, 1:length-1], length-2, batch_size, actions, 
                  spans, binary_tree, other_data]
    return data_batch
