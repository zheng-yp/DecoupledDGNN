# coding=utf-8
import numpy as np
import random
import pandas as pd
import pickle
import torch
import pdb
import random
import os
import pdb

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels, shuffle=False):
        temp = list(zip(sources, destinations, timestamps, edge_idxs, labels))
        if shuffle:
            random.shuffle(temp)
        else:
            temp.sort(key=lambda x: x[2], reverse=False) ##按timestamp正序排序
        sources[:], destinations[:], timestamps[:], edge_idxs[:], labels[:] = zip(*temp)
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = np.concatenate((sources, destinations))
        self.unique_nodes = np.unique(self.unique_nodes)
        self.n_unique_nodes = len(self.unique_nodes)
        self.unique_times = np.unique(timestamps)

class EdgeHelper():
    def __init__(self, dataset_name, randomize=True, disperse=False, use_neg=False, inductive=False, split=False):
        self.dataset_name = dataset_name
        self.split = split
        self.use_neg = use_neg
        self.time_edge_dict = dict()
        self.nodes_seq_lst = []
        self.node_num = 0
        self.get_time_edges(disperse,inductive)
        self.get_nodes_seq_lst(randomize,disperse,inductive)

    def get_time_edges(self,disperse=False, inductive=False):
        disperse_str = ''
        if disperse:
            disperse_str = '_disperse'
        if self.split:
            history = 1
            for i, ss in enumerate(['train', 'valid', 'test']):
                time_edge_file = './data/{0}/{0}_time_edge_map_{1}.pkl'.format(self.dataset_name, ss)
                with open(time_edge_file, 'rb') as f:
                    time_edge_dict = pickle.load(f)
                for idx, time in enumerate(time_edge_dict):
                    edges = time_edge_dict[time]
                    self.time_edge_dict[ss+str(time)] = {'idx': idx+history, 'edges': edges}
                print('%s has %d time step'%(ss, len(time_edge_dict)))
                history += len(time_edge_dict)
        else:
            if inductive:
                time_edge_file = './data/{0}/{0}_time_edge_map_inductive{1}.pkl'.format(self.dataset_name, disperse_str)
            else:
                time_edge_file = './data/{0}/{0}_time_edge_map{1}.pkl'.format(self.dataset_name, disperse_str)
            with open(time_edge_file, 'rb') as f:
                time_edge_dict = pickle.load(f)

            self.time_lst = np.zeros(len(time_edge_dict) + 1)
            self.time_lst[0] = -1
            for idx, time in enumerate(time_edge_dict):
                edges = time_edge_dict[time]
                self.time_edge_dict[time] = {'idx': idx+1, 'edges': edges}
                self.time_lst[idx+1] = time

    def get_nodes_seq_lst(self, randomize=True, disperse=False, inductive=False):
        rand_str = ''
        if randomize:
            rand_str = '_randomize'
        disperse_str = ''
        if disperse:
            disperse_str = '_disperse'
        if self.dataset_name in ['CollegeMsg', 'bitcoinalpha', 'bitcoinotc'] and self.use_neg:
            file1 = './data/{0}/{0}_nodes_seq_lst{1}_mul{2}.pkl'.format(self.dataset_name, rand_str, disperse_str) ## 1-alpha
            file2 = './data/{0}/{0}_nodes_seq_lst{1}_mul{2}_alpha-1.pkl'.format(self.dataset_name, rand_str, disperse_str) ## alpha-1
            with open(file1, 'rb') as f:
                aaalpha = pickle.load(f)
            with open(file2, 'rb') as f:
                alphaaa = pickle.load(f)
            self.node_num = len(aaalpha)
            self.nodes_seq_lst = []
            for i in range(len(aaalpha)):
                self.nodes_seq_lst.append(scipy.sparse.hstack((aaalpha[i], alphaaa[i]),format='csr'))
        else:
            if inductive:
                nodes_seq_lst_file = './data/{0}/{0}_nodes_seq_lst{1}_mul_inductive{2}.pkl'.format(self.dataset_name, rand_str, disperse_str)
            else:
                nodes_seq_lst_file = './data/{0}/{0}_nodes_seq_lst{1}_mul{2}.pkl'.format(self.dataset_name, rand_str, disperse_str)
            with open(nodes_seq_lst_file, 'rb') as f:
                self.nodes_seq_lst = pickle.load(f)
            self.node_num = len(self.nodes_seq_lst)

    def get_edges_feats(self, sources, destinations, timestamps, window_size=5, concat=True):
        src_feat_lst = dst_feat_lst = torch.zeros((len(sources), window_size, self.nodes_seq_lst[0].shape[1]))
        src_dts_lst = dts_dts_lst = torch.zeros((len(sources), window_size))
        for i, (src, dst, ts) in enumerate(zip(sources, destinations, timestamps)):
            ts = round(ts, 3)
            ts_id = self.time_edge_dict[ts]['idx']

            src_feat, src_prev_ts = self.get_node_feats(src, ts_id, window_size)
            dst_feat, dst_prev_ts = self.get_node_feats(dst, ts_id, window_size)
            src_feat_lst[i][-len(src_feat):] = src_feat
            dst_feat_lst[i][-len(dst_feat):] = dst_feat

            src_dts = ts - src_prev_ts
            dst_dts = ts - dst_prev_ts
            src_dts_lst[i][-len(src_dts):] = torch.tensor(src_dts, dtype=torch.float32)
            dts_dts_lst[i][-len(dst_dts):] = torch.tensor(dst_dts, dtype=torch.float32)
        if concat:
            edges_feats = torch.cat((src_feat_lst, dst_feat_lst), dim=-1)
            return edges_feats, src_dts_lst, dts_dts_lst
        else:
            return src_feat_lst, dst_feat_lst, src_dts_lst, dts_dts_lst

    def get_node_feats(self, node, tsid, window_size):
        mask = np.array(self.nodes_seq_lst[node].sum(-1)) != 0
        mask = mask.reshape(-1)
        mask *= np.arange(self.nodes_seq_lst[node].shape[0]) < tsid
        node_feat = self.nodes_seq_lst[node][mask]
        node_feat = node_feat[-window_size:, :]
        node_feat_tensor = torch.tensor(node_feat.toarray(), dtype=torch.float32)
        node_time = self.time_lst[mask][-window_size:]
        return node_feat_tensor, node_time


def get_data_transductive(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, shuffle=False,disperse=False):
    disperse_str = ''
    if disperse:
        disperse_str = '_disperse'
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/{0}/ml_{0}{1}.csv'.format(dataset_name, disperse_str))

    timestamps = graph_df.ts.values
    if disperse:
        timestamps = graph_df.ts_str.values

    if dataset_name in ['bitcoinotc', 'bitcoinalpha']:
        splits = [0.70, 0.80]
        ts_lst=np.unique(timestamps)
        train_end = int(len(ts_lst) * splits[0])
        valid_end = int(len(ts_lst) * splits[1])
        val_time=ts_lst[train_end]
        test_time=ts_lst[valid_end]
    elif dataset_name in ['CollegeMsg']:
        splits = [0.71, 0.80]
        ts_lst=np.unique(timestamps)
        train_end = int(len(ts_lst) * splits[0])
        valid_end = int(len(ts_lst) * splits[1])
        valid_end += 2 ##
        val_time=ts_lst[train_end]
        test_time=ts_lst[valid_end]
    else:
        val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    if {'label'}.issubset(graph_df.columns):
        labels = graph_df.label.values
    else:
        labels = np.ones(sources.shape)

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
      set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_mask = timestamps <= val_time #transductive

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask], shuffle=shuffle)

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
          [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
          [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


    else:
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])
    print('split finish')
    pdb.set_trace()

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

    print("--------- Get {} data: Transductive ---------".format(dataset_name))
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
    return full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data

def get_data_inductive(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, shuffle=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/{0}/ml_{0}.csv'.format(dataset_name))
    split = np.load('./data/{0}/{0}_split_mask.npz'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    if {'label'}.issubset(graph_df.columns):
        labels = graph_df.label.values
    else:
        labels = np.ones(sources.shape)
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_mask = split['train_mask']
    val_mask = split['val_mask']
    test_mask = split['test_mask']
    new_node_val_mask = split['new_node_val_mask']
    new_node_test_mask = split['new_node_test_mask']
    
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask], shuffle=shuffle)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

    new_test_source_mask = split['new_test_source_mask']
    new_test_destination_mask = split['new_test_destination_mask']
    new_test_node_set = set(sources[new_test_source_mask]) | set(destinations[new_test_destination_mask])

    print("--------- Get data: Inductive ---------")
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

    return full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data
