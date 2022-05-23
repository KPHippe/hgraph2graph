import json
import torch
import numpy
import pickle
import argparse
from pathlib import Path
import math, random, sys
from functools import partial
from multiprocessing import Pool

from hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    # Hacky solution... Set batch size to 1 and throw them out if they are none
    # See line 120 (ish) for how we handle them 
    try: 
        x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    except KeyError: 
        return None
    except AssertionError as e: 
        return None 
    except RecursionError as r:
        return None

    return to_numpy(x)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tensors_per_file', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='single')
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--out_dir', type=Path, default=Path("."))
    parser.add_argument('--save_association', action='store_true', default=False)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    args.vocab = PairVocab(vocab, cuda=False)

    pool = Pool(args.ncpu) 
    random.seed(1)

    if args.mode == 'pair':
        #dataset contains molecule pairs
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_pair, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open(args.out_dir / ('tensors-%d.pkl' % split_id), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'cond_pair':
        #dataset contains molecule pairs with conditions
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:3] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_cond, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open(args.out_dir / ('tensors-%d.pkl' % split_id), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        #dataset contains single molecules
        with open(args.train) as f:
            smiles_data = [line.strip("\r\n ").split()[0] for line in f]


        batches = [smiles_data[i : i + args.batch_size] for i in range(0, len(smiles_data), args.batch_size)]
        func = partial(tensorize, vocab = args.vocab)
        # all_data = pool.map(func, batches) original function 
        raw_tensor_data = pool.map(func, batches)
        tensor_data = [] 
        bad_count = 0 
        #handling if the graphs for some smiles don't get created correctly.
        data_association = {}  
        counter = 0 
        for smile, elem in zip(smiles_data, raw_tensor_data): 
            if elem is not None: 
                tensor_data.append(elem)
                data_association[smile] = counter
                counter += 1  
            else: 
                bad_count += 1 
            
        

        print(f"Bad smiles strings: {bad_count}")

        num_splits = len(tensor_data) // args.tensors_per_file
        le = (len(tensor_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = tensor_data[st : st + le]

            with open(args.out_dir / ('tensors-%d.pkl' % split_id), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

        if args.save_association: 
            data_association['metadata'] = {'num_splits': num_splits, 'le': le, 'tensors_per_file': args.tensors_per_file}
            json.dump(data_association, open(args.out_dir / 'tensor_association.json', 'w'))  
            

