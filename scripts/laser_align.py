# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import pandas as pd
import sys
import os

from laserembeddings import Laser


BATCH_SIZE = 500

def convert_to_np(lst, dtype='float'):
  count = len(lst)
  dim = len(lst[0])

  matrix = np.empty((count, dim), dtype=dtype)
  for i in range(count):
    matrix[i] = np.asarray(lst[i], dtype=dtype)

  return matrix

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
  xp = get_array_module(m)
  n = m.shape[0]
  ans = xp.zeros(n, dtype=m.dtype)
  if k <= 0:
    return ans
  if not inplace:
    m = xp.array(m)
  ind0 = xp.arange(n)
  ind1 = xp.empty(n, dtype=int)
  minimum = m.min()
  for i in range(k):
    m.argmax(axis=1, out=ind1)
    ans += m[ind0, ind1]
    m[ind0, ind1] = minimum
  return ans / k


def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Select candidate translations giving sentences in two languages')
  parser.add_argument('-s', '--src_sentences', default=sys.stdin.fileno(), help='the file containing source sentences.')
  parser.add_argument('-t', '--trg_sentences', default=sys.stdin.fileno(), help='the file containing target sentences.')
  parser.add_argument('--src_lang', required=True, type=str, help='source language code.')
  parser.add_argument('--trg_lang', required=True, type=str, help='target language code.')
  parser.add_argument('-o', '--output', default='', help='path to save the translations.')
  parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
  parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
  parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
  parser.add_argument('-n', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
  parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
  parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
  parser.add_argument('--seed', type=int, default=0, help='the random seed')
  parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
  parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
  args = parser.parse_args()

  # Choose the right dtype for the desired precision
  if args.precision == 'fp16':
    dtype = 'float16'
  elif args.precision == 'fp32':
    dtype = 'float32'
  elif args.precision == 'fp64':
    dtype = 'float64'

  if not os.path.isdir(args.output):
    os.makedirs(args.output)
    print('creating output directory: done')

  laser = Laser()

  # Get source embeddings
  with open(args.src_sentences, 'r') as f:  
    src_sents = f.readlines()
    src_sents = [line.strip() for line in src_sents]

  x = laser.embed_sentences(src_sents, lang=args.src_lang)
  x = convert_to_np(x)

  # Get target embeddings
  with open(args.trg_sentences, 'r') as f:  
    trg_sents = f.readlines()
    trg_sents = [line.strip() for line in trg_sents]
  
  z = laser.embed_sentences(trg_sents, lang=args.trg_lang)
  z = convert_to_np(z)

  # NumPy/CuPy management
  if args.cuda:
    if not supports_cupy():
      print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
      sys.exit(-1)
    xp = get_cupy()
    src_embeddings = xp.asarray(x)
    trg_embeddings = xp.asarray(z)
  else:
    print('cuda not provided, using cpu.')
    xp = np
  xp.random.seed(args.seed)

  # Length normalize embeddings so their dot product effectively computes the cosine similarity
  if not args.dot:
    embeddings.length_normalize(x)
    embeddings.length_normalize(z)

    print('normarlize embeddings: done')

  # Build sent to index map
  src_sent2ind = {sent: i for i, sent in enumerate(src_sents)}
  print('build source sent to index map: done')
  print('length of source embedding', len(src_sent2ind))
  
  trg_sent2ind = {sent: i for i, sent in enumerate(trg_sents)}
  print('build target word to index map: done')
  print('length of target embedding', len(trg_sent2ind))

  src = [ind for ind in src_sent2ind.values()]

  # Find translations
  translation = collections.defaultdict(int)

  # Standard nearest neighbor
  if args.retrieval == 'nn':
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      similarities = x[src[i:j]].dot(z.T)
      nn = similarities.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]
  
  # Inverted nearest neighbor
  elif args.retrieval == 'invnn':
    best_rank = np.full(len(src), x.shape[0], dtype=int)
    best_sim = np.full(len(src), -100, dtype=dtype)
    for i in range(0, z.shape[0], BATCH_SIZE):
      j = min(i + BATCH_SIZE, z.shape[0])
      similarities = z[i:j].dot(x.T)
      ind = (-similarities).argsort(axis=1)
      ranks = asnumpy(ind.argsort(axis=1)[:, src])
      sims = asnumpy(similarities[:, src])
      for k in range(i, j):
        for l in range(len(src)):
          rank = ranks[k-i, l]
          sim = sims[k-i, l]
          if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
            best_rank[l] = rank
            best_sim[l] = sim
            translation[src[l]] = k
  
  # Inverted softmax
  elif args.retrieval == 'invsoftmax':
    sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
    partition = xp.zeros(z.shape[0])
    for i in range(0, len(sample), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(sample))
      partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
      nn = p.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]
  
  # Cross-domain similarity local scaling
  elif args.retrieval == 'csls':
    knn_sim_bwd = xp.zeros(z.shape[0])
    for i in range(0, z.shape[0], BATCH_SIZE):
      j = min(i + BATCH_SIZE, z.shape[0])
      knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
      nn = similarities.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]

  # save translations
  trans_src = [src_sents[s] for s in translation.keys()]
  trans_trg = [trg_sents[t] for t in translation.values()]

  df = pd.DataFrame({'source sentences': trans_src, 'translations': trans_trg})
  #print(df)
  df.to_csv(os.path.join(args.output, 'laser_translations.csv'), index=False)

if __name__ == '__main__':
  main()