import argparse
import time
import os
import random
import collections
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from meter import AverageMeter
from model_TAE import TAE, AAE
from vocab import Vocab
from batchify import get_batches
from utils import *
from train import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints', metavar='DIR',
                    help='checkpoint directory')
parser.add_argument('--output', default='sample', metavar='FILE',
                    help='output file name (in checkpoint directory)')
parser.add_argument('--data', default='data/ccsplit', metavar='FILE',
                    help='path to data file')
parser.add_argument('--vocab_loc', default='vocab', metavar='DIR',
                    help='path to load vocab')

parser.add_argument('--enc', default='mu', metavar='M',
                    choices=['mu', 'z'],
                    help='encode to mean of q(z|x) or sample z from q(z|x)')
parser.add_argument('--dec', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='decoding algorithm')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--max-len', type=int, default=64, metavar='N',
                    help='max sequence length')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate on data file')
parser.add_argument('--ppl', action='store_true',
                    help='compute ppl by importance sampling')
parser.add_argument('--reconstruct', action='store_true',
                    help='reconstruct data file')
parser.add_argument('--sample', action='store_true',
                    help='sample sentences from prior')
parser.add_argument('--m', type=int, default=100, metavar='N',
                    help='num of samples for importance sampling estimate')
parser.add_argument('--n', type=int, default=5, metavar='N',
                    help='num of sentences to generate for sample/interpolate')

parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')

def get_model(path):
    ckpt = torch.load(path, map_location = device)
    train_args = ckpt['args']
    model = {'aae': AAE, 'tae' : TAE }[train_args.model_type](
        vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    vocab = Vocab(os.path.join(args.vocab_loc, 'vocab.txt'))
    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, 'model.pt'))

    if args.reconstruct:
        f = open(os.path.join(args.checkpoint, args.output, "test_inference.txt"), 'w')
        test_sents = load_sent(args.data)
        batches = get_batches(test_sents, vocab, args.batch_size, device)
            
        for inputs, _ in tqdm(batches):
            mu, logvar, _ = model.encode(inputs)
            zi = mu

            pred_sentence = torch.ones(zi.shape[0], zi.shape[1]).long().to(device)*vocab.go
            for t in range(1, zi.shape[0]):
                _input = model.position_encoding(model.embed(pred_sentence))
                _input = _input[:t,:]
                memory = model.drop(model.z2emb(zi))
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t).to(device)
                output = model.Decoder(_input, memory, tgt_mask)
                output = model.drop(output)
                logits = model.proj(output.view(-1, output.size(-1)))
                pred_next = logits.view(output.size(0), output.size(1), -1).argmax(dim=-1)
                pred_sentence[1:t+1, ] = pred_next

            inputs = inputs.t()
            pred_sentence = pred_sentence.t()
            
            for pair in zip(inputs[:, 1 :], pred_sentence[:, 1:]):
                for sen, stop in zip(pair, [vocab.pad, vocab.eos]):
                    a = (sen == stop).nonzero(as_tuple=False)
                    if a.numel() == 0:
                        pass
                    else : 
                        index = a[0,0]
                        sen = torch.split(sen, index)[0]
                    result = ' '.join(map(lambda x: vocab.idx2word[x], sen)) + '\n'
                    f.write(result)
                f.write('\n')
        f.close()
        print('END')

