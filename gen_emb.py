# coding=utf-8
import torch
import gc
import numpy as np
from propagation import InstantGNN
import argparse
import pickle
import os
import copy
import scipy.sparse as sp
import pdb
import uuid
from datetime import datetime

np.random.seed(0)

## init: empty graph
def load_data_init(path, datastr, rmax, alpha, randomize_features=False, neg=False):
    if datastr == 'wikipedia_init':
        m = 9227; n = 9227
    if datastr == 'reddit_init':
        m = 10984; n = 10984
    if datastr == 'CollegeMsg_init':
        m = 1899; n = 1899
    if datastr == 'bitcoinotc_init':
        m = 5881; n = 5881
    if datastr == 'bitcoinalpha_init':
        m = 3783; n = 3783
    if datastr == 'GDELT_init':
        m = 16682; n = 16682
    if datastr == 'MAG_init':
        m = 72508661; n = 72508661

    print("Load %s!" % datastr)

    py_alg = InstantGNN()
    dataset = datastr.split('_')[0]
    if randomize_features:
        if dataset in ['wikipedia', 'reddit']:
            features = np.random.rand(n, 172)
        else:
            features = np.random.rand(n, 128)
    else:
        features = torch.load('/data/yanping_zheng/data/{}/node_features.pt'.format(dataset))
        if features.dtype == torch.bool:
            features = features.type(torch.float64)
        features=features.numpy()

    print('features:', features)
    memory_dataset = py_alg.initial_operation(path, datastr, m, n, rmax, alpha, features)
    return features, py_alg, n

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/wikipedia/',
                        help='graph data path')
parser.add_argument('--data', default='wikipedia',
                        help='graph name, e.g. wikipedia, reddit, CollegeMsg, bitcoinotc, bitcoinalpha, GDELT, MAG')
parser.add_argument('--rmax', type=float, default=1e-7, help='rmax')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--save_num', type=int, default=100000, help='save num')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--split', action='store_true', help='Whether split users sequence')
parser.add_argument('--disperse', action='store_true', help='Whether disperse graph')
parser.add_argument('--undirect', action='store_true', help='Undirect graph')
args = parser.parse_args()
print(args)

features, py_alg, node_num = load_data_init(args.path, args.data+'_init', rmax=args.rmax, alpha=args.alpha, randomize_features=args.randomize_features)

feat_dim = features.shape[1]
print('feat_dim:', feat_dim)
nodes_seq_lst = []

rand_str = ''
if args.randomize_features:
    rand_str = '_randomize'

disperse_str = ''
if args.disperse:
    disperse_str = '_disperse'

out_file = args.path+args.data+'_nodes_seq_lst'+ rand_str +'_mul' + disperse_str

time_edge_dict_lst = []
seq_len = 0

if args.split:
    for ss in ['train', 'valid', 'test']:
        time_edge_file = args.path+args.data+'_time_edge_map_' + ss + '.pkl'
        with open(time_edge_file, 'rb') as f:
            time_edge_dict = pickle.load(f)
        time_edge_dict_lst.append(time_edge_dict)
        seq_len += len(time_edge_dict)
else:
    time_edge_file = args.path+args.data+'_time_edge_map'+ disperse_str +'.pkl'
    if args.disperse:
        time_edge_file = args.path+args.data+'_time_edge_map_disperse.pkl'
    with open(time_edge_file, 'rb') as f:
        time_edge_dict = pickle.load(f)
    time_edge_dict_lst.append(time_edge_dict)
    seq_len += len(time_edge_dict)

nodes_seq_lst=[np.zeros((seq_len + 1, feat_dim)) for i in range(node_num)]


for node in range(node_num):
    nodes_seq_lst[node][0] = features[node]
print('init feat append......')

if args.split:
    splits = ['train', 'valid', 'test']
else:
    splits = ['full']

pdb.set_trace()
count = 0
history = 0
tmp_file = 'tmp_'+args.data+'.txt'
for it, ss in enumerate(splits):
    print('---- %s ----' % ss)
    time_edge_dict = time_edge_dict_lst[it]

    for idx, time in enumerate(time_edge_dict):
        old_feat = copy.deepcopy(features)
        edges = time_edge_dict[time]
        print('idx: ',idx+history+1,'/',seq_len+1,', time: ', time, ', edges: ', edges.shape)

        ##reverse edges
        if args.undirect:
            ss, tt = edges[:,0], edges[:,1]
            ss=ss.reshape(-1,1)
            tt=tt.reshape(-1,1)
            re_edges=np.concatenate([tt,ss], axis=1)
            edges = np.concatenate([edges, re_edges])

        np.savetxt(tmp_file, edges, fmt='%d', delimiter=' ')

        py_alg.snapshot_operation(tmp_file, args.rmax, args.alpha, features)
        os.remove(tmp_file)

        delta_feat = features - old_feat

        affacted_nodes, pos = np.where(delta_feat!=0)
        for cur_node, cur_pos in zip(affacted_nodes, pos):
            nodes_seq_lst[cur_node][idx+1+history, cur_pos] = delta_feat[cur_node, cur_pos]
    history += len(time_edge_dict)

pdb.set_trace()
out_file += '.pkl'
for i in range(node_num): nodes_seq_lst[i] = sp.csr_matrix(nodes_seq_lst[i])
ttf = open(out_file,'wb')
pickle.dump(nodes_seq_lst, ttf, pickle.HIGHEST_PROTOCOL)
ttf.close()
print('get embeddings finish..')

