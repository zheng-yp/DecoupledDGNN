# coding=utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from src import Preprocessing
#from src import MyLinkPrediction, MLP

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import parameter_parser
import pdb

import pickle
from data_loader import Data
import math
import random
import pandas as pd
import os

def get_data_nodecla(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, shuffle=True):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/{0}/ml_{0}.csv'.format(dataset_name))
    edge_features = np.load('./data/{0}/ml_{0}.npy'.format(dataset_name))
    node_features = np.load('./data/{0}/ml_{0}_node.npy'.format(dataset_name))

    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    if {'label'}.issubset(graph_df.columns):
        labels = graph_df.label.values
    else:
        labels = np.ones(sources.shape)
    print('labels: ', labels)
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_mask = timestamps <= val_time #transductive

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask], shuffle=shuffle)

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask], shuffle=shuffle)

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask], shuffle=shuffle)

    pos_mask = (train_data.labels == 1)
    neg_mask = (train_data.labels == 0)

    train_pos_data = Data(train_data.sources[pos_mask], train_data.destinations[pos_mask], train_data.timestamps[pos_mask],
                          train_data.edge_idxs[pos_mask], train_data.labels[pos_mask], shuffle=shuffle)
    train_neg_data = Data(train_data.sources[neg_mask], train_data.destinations[neg_mask], train_data.timestamps[neg_mask],
                          train_data.edge_idxs[neg_mask], train_data.labels[neg_mask], shuffle=shuffle)

    pos_mask = (val_data.labels == 1)
    neg_mask = (val_data.labels == 0)
    
    valid_pos_data = Data(val_data.sources[pos_mask], val_data.destinations[pos_mask], val_data.timestamps[pos_mask],
                          val_data.edge_idxs[pos_mask], val_data.labels[pos_mask], shuffle=shuffle)
    valid_neg_data = Data(val_data.sources[neg_mask], val_data.destinations[neg_mask], val_data.timestamps[neg_mask],
                          val_data.edge_idxs[neg_mask], val_data.labels[neg_mask], shuffle=shuffle)

    pos_mask = (test_data.labels == 1)
    neg_mask = (test_data.labels == 0)
    
    test_pos_data = Data(test_data.sources[pos_mask], test_data.destinations[pos_mask], test_data.timestamps[pos_mask],
                          test_data.edge_idxs[pos_mask], test_data.labels[pos_mask], shuffle=shuffle)
    test_neg_data = Data(test_data.sources[neg_mask], test_data.destinations[neg_mask], test_data.timestamps[neg_mask],
                          test_data.edge_idxs[neg_mask], test_data.labels[neg_mask], shuffle=shuffle)

    print("--------- Get data for node classification: Transductive ---------")
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))

    return full_data, train_pos_data, train_neg_data, valid_pos_data, valid_neg_data, test_pos_data, test_neg_data

class EdgeHelper():
    def __init__(self, dataset_name, inductive=False):
        self.dataset_name = dataset_name
        self.time_edge_dict = dict()
        self.nodes_seq_lst = []
        self.get_time_edges(inductive)
        self.get_nodes_seq_lst(inductive)

    def get_time_edges(self, inductive=False):
        if inductive:
            time_edge_file = './data/{0}/{0}_time_edge_map_inductive.pkl'.format(self.dataset_name)
        else:
            time_edge_file = './data/{0}/{0}_time_edge_map.pkl'.format(self.dataset_name)
        with open(time_edge_file, 'rb') as f:
            time_edge_dict = pickle.load(f)
        for idx, time in enumerate(time_edge_dict):
            edges = time_edge_dict[time]
            self.time_edge_dict[time] = {'idx': idx+1, 'edges': edges}

    def get_nodes_seq_lst(self, inductive=False):
        if inductive:
            nodes_seq_lst_file = './data/{0}/{0}_nodes_seq_lst_randomize_mul_inductive.pkl'.format(self.dataset_name)
        else:
            nodes_seq_lst_file = './data/{0}/{0}_nodes_seq_lst_randomize_mul.pkl'.format(self.dataset_name)
        with open(nodes_seq_lst_file, 'rb') as f:
            self.nodes_seq_lst = pickle.load(f)

    def cal_node_temporal_feat(self, data):
        records_features = []
        use_edge_feat = False
        nodes, timestamps = data.sources, data.timestamps
        if self.dataset_name in ['reddit', 'wikipedia']:
            use_edge_feat = True
            edge_features = np.load('data/{0}/ml_{0}.npy'.format(self.dataset_name))
            assert ((edge_features.shape[0]-1) == len(nodes))
        for idx, (src, ts) in enumerate(zip(nodes, timestamps)):
            if idx % 10000 == 0:
                print('idx: ', idx)
            ts = round(ts, 3)
            ts_id = self.time_edge_dict[ts]['idx']
            src_feat = self.nodes_seq_lst[src][1:ts_id+1, :].sum(axis=0) # 不加0时刻的随机特征 = 0时刻赋0向量
            if use_edge_feat:
                edge_feat = edge_features[idx + 1] #第一个是无意义的0向量
                src_feat = np.array(src_feat).squeeze()
                src_feat = np.concatenate((src_feat, edge_feat))
            src_feat = torch.tensor(src_feat, dtype=torch.float32).squeeze()
            records_features.append(src_feat)
        records_features=torch.stack(records_features)
        return records_features

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc_2 = torch.nn.Linear(hidden_dim, out_dim)
        #self.fc_3 = torch.nn.Linear(out_dim, out_dim)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = torch.nn.Sigmoid()(self.fc_2(x))
        return x

class Execute:
    def __init__(self, args):
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.args = args

        # Set device
        if args.gpu>=0:
            device_string = 'cuda:{}'.format(args.gpu) # if torch.cuda.is_available() else 'cpu'
        else:
            device_string = 'cpu'
        self.device = torch.device(device_string)

        self.batch_size = args.batch_size
        self.data = args.data

        self.window_size = args.window_size
        self.inductive = args.inductive
        self.shuffle = args.shuffle
        self.checkpt_path = args.checkpt_path

        ##params for focalloss
        self.alpha = args.alpha
        self.gamma = args.gamma

        self.checkpt_file = self.checkpt_path + '/' + self.data + '_ws'+str(self.window_size)+'_nodecla_best.pt'
        print('Decoder model will save at: ', self.checkpt_file)

        self.patience = args.patience

    def train(self):
        full_data, train_data, train_neg_data, val_data, val_neg_data, test_data, test_neg_data = get_data_nodecla(self.data, shuffle=True)
        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / self.batch_size)
        print('num of training instances: {}'.format(num_instance))
        print('num of batches per epoch: {}'.format(num_batch))
 
        emb_file = 'data/{}/records_features.pt'.format(self.data)
        if os.path.exists(emb_file):
            print('load emb at ', emb_file)
            self.all_feats = torch.load(emb_file)
        else:
            self.edge_helper = EdgeHelper(self.data, self.inductive)
            self.all_feats = self.edge_helper.cal_node_temporal_feat(full_data)
            torch.save(self.all_feats, emb_file)
            print('emb sava at ', emb_file)

        self.decoder = MLP(self.all_feats.shape[1], hidden_dim=self.args.hidden_dim, out_dim=1).to(self.device)

        self.criterion = nn.BCELoss()
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=args.learning_rate)
        best_auc, best_epoch = 0, 0
        for epoch in range(args.epochs):
            self.decoder.train()
            correct = 0
            num_samples = 0
            all_loss = []
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_instance, start_idx + self.batch_size)
                #print('batch_idx: ', batch_idx,', start_idx: ', start_idx, ', end_idx: ', end_idx)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                labels_batch = train_data.labels[start_idx:end_idx]
                size = len(sources_batch)
                ### get pos features
                pos_features = self.all_feats[edge_idxs_batch]
                pos_labels_torch = torch.tensor(labels_batch).float()
                assert torch.sum(pos_labels_torch) == size

                neg_idxs = np.random.randint(len(train_neg_data.edge_idxs), size=size)
                neg_features = self.all_feats[train_neg_data.edge_idxs[neg_idxs]]
                neg_labels_batch = torch.zeros(size)
                labels_batch_torch = torch.cat((pos_labels_torch, neg_labels_batch))

                decoder_optimizer.zero_grad()
                temp = torch.cat((pos_features, neg_features))
                index = [i for i in range(len(temp))]
                random.shuffle(index)
                preds = self.decoder(temp[index].to(self.device))

                labels_batch_torch = labels_batch_torch[index]
                loss = self.criterion(preds.squeeze(dim=1), labels_batch_torch.to(self.device))

                all_loss.append(loss.item())

                loss.backward()
                decoder_optimizer.step()

                pred_labels = torch.zeros(preds.shape[0])
                pred_labels[preds.squeeze(dim=1) > 0.5] = 1
                correct += torch.sum(pred_labels == labels_batch_torch).item()
                num_samples += (preds.shape[0])
            train_loss = np.mean(all_loss)
            train_acc = correct / num_samples

            valid_loss, valid_acc, valid_auc, valid_ap = self.valid(val_data, val_neg_data)
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Valid loss: %.5f, Valid accuracy: %.5f, Valid AUC: %.5f, Valid AP: %.5f" % (epoch+1, train_loss, train_acc, valid_loss, valid_acc, valid_auc, valid_ap))
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_epoch = epoch
                torch.save(self.decoder.state_dict(), self.checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == self.patience:
                break
        print('begin test...load model at epoch: %d' % (best_epoch+1))
        test_loss, test_acc, test_auc, test_ap = self.test(test_data, test_neg_data)
        print("Test loss: %.5f, Test accuracy: %.5f, Test AUC: %.5f, Test AP: %.5f" % (test_loss, test_acc, test_auc, test_ap))

    def valid(self, val_data, val_neg_data):
        self.decoder.eval()
        correct = 0
        num_samples = 0
        all_loss = []

        val_batchsize = 2*self.batch_size
        num_instance = len(val_data.sources)
        num_batch = math.ceil(num_instance / val_batchsize)
        y_true, y_pred = [], []
        for batch_idx in range(num_batch):
            start_idx = batch_idx * val_batchsize
            end_idx = min(num_instance, start_idx + val_batchsize)
            sources_batch, destinations_batch = val_data.sources[start_idx:end_idx], \
                                                val_data.destinations[start_idx:end_idx]
            edge_idxs_batch = val_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = val_data.timestamps[start_idx:end_idx]
            labels_batch = val_data.labels[start_idx:end_idx]
            size = len(sources_batch)
            ### get pos features
            pos_features = self.all_feats[edge_idxs_batch]
            pos_labels_torch = torch.tensor(labels_batch).float()
            assert torch.sum(pos_labels_torch) == size

            neg_idxs = np.random.randint(len(val_neg_data.edge_idxs), size=size)
            neg_features = self.all_feats[val_neg_data.edge_idxs[neg_idxs]]
            neg_labels_batch = torch.zeros(size)
            labels_batch_torch = torch.cat((pos_labels_torch, neg_labels_batch))

            preds = self.decoder(torch.cat((pos_features, neg_features)).to(self.device))

            loss = self.criterion(preds.squeeze(dim=1), labels_batch_torch.to(self.device))
            all_loss.append(loss.item())

            pred_labels = torch.zeros(preds.shape[0])
            pred_labels[preds.squeeze(dim=1) > 0.5] = 1
            correct += torch.sum(pred_labels == labels_batch_torch).item()
            num_samples += (preds.shape[0])

            y_pred += list(preds.detach().cpu().numpy().squeeze())
            y_true += list(labels_batch_torch.numpy())

        auc = self.auc_score(y_true, y_pred)
        ap = self.ap_score(y_true, y_pred)
        valid_loss = np.mean(all_loss)
        valid_acc = correct / num_samples
        return valid_loss, valid_acc, auc, ap

    def auc_score(self, y_true, y_score):
        ''' use sklearn roc_auc_score API
            y_true & y_score; array-like, shape = [n_samples]
        '''
        from sklearn.metrics import roc_auc_score
        roc = roc_auc_score(y_true=y_true, y_score=y_score)
        return roc

    def ap_score(self, y_true, y_score):
        ''' use sklearn roc_auc_score API
            y_true & y_score; array-like, shape = [n_samples]
        '''
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_score)
        return ap

    def test(self, test_data, test_neg_data):
        self.decoder.load_state_dict(torch.load(self.checkpt_file))
        self.decoder.eval()
        correct = 0
        num_samples = 0
        all_loss = []

        num_instance = len(test_data.sources)
        test_batchsize = 2*self.batch_size
        num_batch = math.ceil(num_instance / test_batchsize)
        y_true, y_pred = [], []
        for batch_idx in range(num_batch):
            start_idx = batch_idx * test_batchsize
            end_idx = min(num_instance, start_idx + test_batchsize)
            sources_batch, destinations_batch = test_data.sources[start_idx:end_idx], \
                                                test_data.destinations[start_idx:end_idx]
            edge_idxs_batch = test_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = test_data.timestamps[start_idx:end_idx]
            labels_batch = test_data.labels[start_idx:end_idx]
            size = len(sources_batch)
            ### get pos features
            pos_features = self.all_feats[edge_idxs_batch]
            pos_labels_torch = torch.tensor(labels_batch).float()
            assert torch.sum(pos_labels_torch) == size

            neg_idxs = np.random.randint(len(test_neg_data.edge_idxs), size=size)
            neg_features = self.all_feats[test_neg_data.edge_idxs[neg_idxs]]
            neg_labels_batch = torch.zeros(size)
            labels_batch_torch = torch.cat((pos_labels_torch, neg_labels_batch))

            preds = self.decoder(torch.cat((pos_features, neg_features)).to(self.device))

            loss = self.criterion(preds.squeeze(dim=1), labels_batch_torch.to(self.device))
            all_loss.append(loss.item())

            pred_labels = torch.zeros(preds.shape[0])
            pred_labels[preds.squeeze(dim=1) > 0.5] = 1
            correct += torch.sum(pred_labels == labels_batch_torch).item()
            num_samples += (preds.shape[0])

            y_pred += list(preds.detach().cpu().numpy().squeeze())
            y_true += list(labels_batch_torch.numpy())

        auc = self.auc_score(y_true, y_pred)
        ap = self.ap_score(y_true, y_pred)
        test_loss = np.mean(all_loss)
        test_acc = correct / num_samples
        return test_loss, test_acc, auc, ap

if __name__ == "__main__":
    args = parameter_parser()
    print(args)
    execute = Execute(args)
    execute.train()

