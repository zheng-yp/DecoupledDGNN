# coding=utf-8
'''
    Following ROLAND and EvolveGCN, snapshots were obtained at fixed intervals：
    UCI-MSG: 190080 s
    Bitcoin-Alpha, Bitcoin-OTC: 1200000s
'''
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
import pdb
import time

def generate_init_graph(file_name, node_num):
    with open(file_name, 'w+') as f:
        for node_id in range(node_num):
            f.write('%d %d\n'%(node_id, node_id))
    print('init graph generate finish!!')

def reindex(df):
    new_df = df.copy()

    node_list = list(new_df.u.unique())+list(new_df.i.unique())
    node_list = list(set(node_list))
    node_list.sort()
    if max(new_df.u.max(), new_df.i.max()) - 0 + 1 == len(node_list): ## The ids of u and i are consecutive and start from 0.
        return new_df
    min_id = min(new_df.u.min(),new_df.i.min())
    if min_id == 1 and max(new_df.u.max(), new_df.i.max()) - min_id + 1 == len(node_list): ## The ids of u and i are consecutive and start from 1.
        new_df.u -= 1
        new_df.i -= 1
        return new_df

    node_id_map = {}
    for idx, node in enumerate(node_list):
        node_id_map[node] = idx

    from_list, to_list = [], []
    for u, i in zip(df.u, df.i):
        from_id = node_id_map[u]
        to_id = node_id_map[i]
        from_list.append(from_id)
        to_list.append(to_id)
    new_df.u = from_list
    new_df.i = to_list

    return new_df

def disperse_dataset(data, graph_df, snapshot_freq='S'):
    new_df = graph_df.copy()
    timestamps = graph_df.ts.values
    if snapshot_freq in ['S']:
        if data == 'CollegeMsg':
            ts_str = timestamps // 190080
            snap_num = len(set(ts_str))
            print('snapshot_freq is 190080s, snap_num = ', snap_num)
        if data in ['bitcoinotc', 'bitcoinalpha']:
            ts_str = timestamps // 1200000
            snap_num = len(set(ts_str))
            print('snapshot_freq is 1200000s, snap_num = ', snap_num)
    elif snapshot_freq in ['W']:
        pass
    new_df.insert(3, 'ts_str', list(ts_str))
    return new_df

def build_time_edge_map(graph_df, file_name, disperse=True):
    sources = graph_df.u.values
    destinations = graph_df.i.values
    if disperse:
        timestamps = graph_df.ts_str.values
    else:
        timestamps = graph_df.ts.values
    time_set = list(set(timestamps))
    time_set.sort()
    print('len of time_set: ', len(time_set))

    time_edge_map = {}
    for i,tts in enumerate(time_set):
        if i % 20 == 0:
            print(i, ', tts: ', tts)
        from_nodes = sources[timestamps == tts]
        to_nodes = destinations[timestamps == tts]
        edges = np.array([from_nodes, to_nodes])

        edges = edges.transpose(1,0)
        if edges.shape[0]>=2: 
            time_edge_map[tts] = edges

    print(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(time_edge_map, f, pickle.HIGHEST_PROTOCOL)
    print('save time_edge_map finish!!!')

def run(data_name, snapshot_freq=True, disperse=True):
    save_path = "../data/" + data_name
    Path(save_path).mkdir(parents=True, exist_ok=True)

    OUT_DF = save_path + 'ml_{}_disperse.csv'.format(data_name)
    OUT_TIME_EDGE_MAP = save_path + '{}_time_edge_map_disperse.pkl'.format(data_name)
    OUT_INIT_GRAPH = save_path + '{}_init.txt'.format(data_name)

    if data_name =='CollegeMsg':
        PATH = './{}.txt'.format(data_name)
        df = pd.read_csv(PATH, sep=' ', header=None, index_col=None)
        df.columns = ['u', 'i', 'ts']
    else:
        PATH = './{}.csv'.format(data_name)
        df = pd.read_csv(PATH, sep=',', header=None, index_col=None)
        df.columns = ['u', 'i', 'rating', 'ts']
    new_df = reindex(df)
    max_idx = max(new_df.u.max(), new_df.i.max())
    print('num of nodes: ', max_idx+1)

    newnew_df = disperse_dataset(data_name, new_df) ## split into snapshots
    newnew_df.to_csv(OUT_DF)

    generate_init_graph(OUT_INIT_GRAPH, node_num = max_idx+1) # generate empty graph that contains only self-loop edges
    build_time_edge_map(newnew_df,OUT_TIME_EDGE_MAP, disperse=disperse)

parser = argparse.ArgumentParser('Data preprocessing for Bitcoins')
parser.add_argument('--data', type=str, help='Dataset name (eg. CollegeMsg, bitcoinotc, bitcoinalpha)',
                    default='CollegeMsg')
parser.add_argument('--snapshot_freq', type=str, help='How to generate snapshots? [W]eakly or [S]econdly',
                    default='S')

args = parser.parse_args()

run(args.data, snapshot_freq=args.snapshot_freq)


