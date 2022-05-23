import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

from hgraph import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)
parser.add_argument('--data_folder', type=str, required=True)
parser.add_argument('--association', type=str, help='Whether or not to store indexes of smile strings in tensor files')
parser.add_argument('--out_file', type=str, required=True)

parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--nsample', type=int, default=10000)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()


model_state, _, _, beta = torch.load(args.model)
model.load_state_dict(model_state)
model.eval()

torch.manual_seed(args.seed)
random.seed(args.seed)

dataset = DataFolder(args.data_folder, args.batch_size, shuffle=False)
print(f'Data length {len(dataset)}')
embeddings = [] 
with torch.no_grad(): 

    for i, batch in tqdm(enumerate(dataset)):
        try: 
            root_vecs = model.generate_embeddings(*batch, beta=beta)
            embeddings.append(root_vecs.detach().cpu().numpy())
        except RuntimeError as e :
            print(f'error on batch {i}') 
        # print(root_vecs.shape)
        

embeddings = np.concatenate(embeddings)
print(embeddings.shape)

np.save(args.out_file, embeddings)



