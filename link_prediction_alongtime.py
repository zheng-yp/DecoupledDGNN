# coding=utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from src import Preprocessing
from src import MyLinkPrediction
from src.transformer import linkPredictor ##Transformer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import parameter_parser
import pdb

import pickle
from data_loader import get_data_inductive, get_data_transductive, EdgeHelper
import math
import random

def edge_index_difference(edge_all, edge_except, num_nodes):
    """Set difference operator, return edges in edge_all but not
        in edge_except.
    """
    idx_all = edge_all[0] * num_nodes + edge_all[1]
    idx_except = edge_except[0] * num_nodes + edge_except[1]
    mask=np.isin(idx_all, idx_except)
    idx_kept = idx_all[~mask]
    ii = idx_kept // num_nodes
    jj = idx_kept % num_nodes
    return np.vstack((ii,jj)).astype(np.int64)

def gen_negative_edges(sources, destinations, num_nodes, num_neg_per_node):
    """Generates a fixed number of negative edges for each node.

    Args:
        sources: (E) array of positive edges' sources.
        destinations: (E) array of positive edges' destinations.
        num_nodes: total number of nodes.
        num_neg_per_node: approximate number of negative edges generated for
            each source node in edge_index.
    """
    src_lst = np.unique(sources) # get unique senders.
    pos_edge_index = np.vstack((sources, destinations))
    num_neg_per_node = int(1.5 * num_neg_per_node)  # add some redundancy.
    ii = src_lst.repeat(num_neg_per_node)
    jj = np.random.choice(num_nodes, len(ii), replace=True)
    candidates = np.vstack((ii, jj)).astype(np.int64)
    neg_edge_index = edge_index_difference(candidates, pos_edge_index, num_nodes)
    return neg_edge_index[0], neg_edge_index[1]

class Execute:
    def __init__(self, args):
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        self.args = args
        GPU = args.gpu
        # Set device
        device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_string)
        
        self.batch_size = args.batch_size
        self.data = args.data
        self.use_neg = args.use_neg
        if self.use_neg:
            args.emb_size *= 2

        if args.seq_model in ['lstm', 'gru']:
            self.model = MyLinkPrediction(args, self.device)
        elif args.seq_model == 'transformer':
            self.model = linkPredictor(self.device, args.emb_size * 2, args.hidden_dim, 4, 4, args.dropout)

        self.model = self.model.to(self.device)
        self.window_size = args.window_size
        self.inductive = args.inductive
        
        self.shuffle = args.shuffle
        self.checkpt_path = args.checkpt_path
        self.checkpt_file = self.checkpt_path + '/' + self.data + '_ws'+str(self.window_size)+'_best.pt'
        self.patience = args.patience

    def train(self):
        full_data, train_data, val_data, test_data, _, _ = get_data_transductive(self.data, shuffle=self.shuffle, disperse=True)
        print('window_size: ', self.window_size)
        self.edge_helper = EdgeHelper(self.data,randomize=True, disperse=True, use_neg=self.use_neg, inductive=self.inductive, split=False)

        criterion = nn.BCELoss()
        optimizer = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)
        best_ap, best_epoch, best_loss = 0, 0, 10000.
        bad_counter = 0
        for epoch in range(args.epochs):
            self.model.train()
            correct = 0
            num_samples = 0
            all_loss = []

            for time in train_data.unique_times:
                if self.edge_helper.time_edge_dict[time]['idx'] < self.window_size:
                    continue
                edges_snap = self.edge_helper.time_edge_dict[time]['edges']
                num_instance = edges_snap.shape[0]
                num_batch = math.ceil(num_instance / self.batch_size)
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_instance, start_idx + self.batch_size)
                    sources_batch, destinations_batch = edges_snap[start_idx:end_idx, 0], edges_snap[start_idx:end_idx, 1]
                    size = len(sources_batch)
                    timestamps_batch = np.repeat(time, size)

                    if args.seq_model in ['lstm', 'gru']:
                        ### get pos features    (sources_batch, destinations_batch, timestamps_batch)
                        src_features, dst_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
                        pos_preds = self.model.get_edges_embedding(src_features.to(self.device), dst_features.to(self.device))
                    elif args.seq_model == 'transformer':
                        pos_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=True)
                        pos_preds = self.model(pos_features.to(self.device))
                    pos_labels = torch.ones(pos_preds.shape[0], dtype=torch.float, device=self.device)
                    pos_loss = criterion(pos_preds.squeeze(dim=1), pos_labels) ##pos_loss = -torch.log(pos_preds[:, 1])

                    ### get negtive sample and features
                    neg_destinations_batch = np.random.randint(0, self.edge_helper.node_num, size)
                    if args.seq_model in ['lstm', 'gru']:
                        neg_src_features, neg_dst_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
                        neg_preds = self.model.get_edges_embedding(neg_src_features.to(self.device), neg_dst_features.to(self.device))
                    elif args.seq_model == 'transformer':
                        neg_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=True)
                        neg_preds = self.model(neg_features.to(self.device))

                    neg_labels = torch.zeros(neg_preds.shape[0], dtype=torch.float, device=self.device)
                    neg_loss = criterion(neg_preds.squeeze(dim=1), neg_labels) ##neg_loss = -torch.log(neg_preds[:, 0])

                    # backward
                    loss = pos_loss + neg_loss ##loss = torch.mean(pos_loss + neg_loss)
                    all_loss.append(loss.item())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    TP = torch.sum(pos_preds>=0.5)
                    TN = torch.sum(neg_preds<0.5)
                    correct += (TP + TN).item()
                    num_samples += (pos_preds.shape[0] + neg_preds.shape[0])

            train_loss = np.mean(all_loss)
            train_acc = correct / num_samples
            if epoch > 999:
                valid_loss, valid_acc, valid_auc, valid_ap, valid_mrr = self.valid(val_data, cal_mrr=True)
                print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Valid loss: %.5f, Valid accuracy: %.5f, Valid AUC: %.5f, Valid AP: %.5f, MRR: %.5f" % (epoch+1, train_loss, train_acc, valid_loss, valid_acc, valid_auc, valid_ap, valid_mrr))
            else:
                valid_loss, valid_acc, valid_auc, valid_ap = self.valid(val_data)
                print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Valid loss: %.5f, Valid accuracy: %.5f, Valid AUC: %.5f, Valid AP: %.5f" % (epoch+1, train_loss, train_acc, valid_loss, valid_acc, valid_auc, valid_ap))

            if valid_ap > best_ap:
                best_ap = valid_ap
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == self.patience:
                break

        print('begin test...')
        print(best_epoch)
        test_loss, test_acc, test_auc, test_ap, test_mrr = self.test(test_data)
        print("Test loss: %.5f, Test accuracy: %.5f, Test AUC: %.5f, Test AP: %.5f, MRR: %.5f" % (test_loss, test_acc, test_auc, test_ap, test_mrr))

    def valid(self, val_data, cal_mrr=False):
        criterion = nn.BCELoss()
        self.model.eval()
        correct = 0
        num_samples = 0
        all_loss = []

        val_batchsize = 2*self.batch_size
        y_true, y_pred = [], []
        mrr_hist = []
        for time in val_data.unique_times:
            edges_snap = self.edge_helper.time_edge_dict[time]['edges']
            num_instance = edges_snap.shape[0]
            num_batch = math.ceil(num_instance / val_batchsize)
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_instance, start_idx + self.batch_size)
                sources_batch, destinations_batch = edges_snap[start_idx:end_idx, 0], edges_snap[start_idx:end_idx, 1]
                size = len(sources_batch)
                timestamps_batch = np.repeat(time, size)

                if args.seq_model in ['lstm', 'gru']:
                    ### get pos features    (sources_batch, destinations_batch, timestamps_batch)
                    src_features, dst_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
                    pos_preds = self.model.get_edges_embedding(src_features.to(self.device), dst_features.to(self.device))
                elif args.seq_model == 'transformer':
                    pos_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=True)
                    pos_preds = self.model(pos_features.to(self.device))
                pos_labels = torch.ones(pos_preds.shape[0], dtype=torch.float, device=self.device)
                pos_loss = criterion(pos_preds.squeeze(dim=1), pos_labels) ##pos_loss = -torch.log(pos_preds[:, 1])

                ### get negtive sample and features
                neg_destinations_batch = np.random.randint(0, self.edge_helper.node_num, size)
                if args.seq_model in ['lstm', 'gru']:
                    neg_src_features, neg_dst_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
                    neg_preds = self.model.get_edges_embedding(neg_src_features.to(self.device), neg_dst_features.to(self.device))
                elif args.seq_model == 'transformer':
                    neg_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=True)
                    neg_preds = self.model(neg_features.to(self.device))

                neg_labels = torch.zeros(neg_preds.shape[0], dtype=torch.float, device=self.device)
                neg_loss = criterion(neg_preds.squeeze(dim=1), neg_labels)

                loss = pos_loss + neg_loss ##loss = torch.mean(pos_loss + neg_loss)
                all_loss.append(loss.item())
               
                TP = torch.sum(pos_preds>=0.5)
                TN = torch.sum(neg_preds<0.5)

                correct += (TP + TN).item()
                num_samples += (pos_preds.shape[0] + neg_preds.shape[0])
                for i in range(pos_preds.shape[0]):
                    y_pred.append(pos_preds[i].item())
                    y_true.append(1)
                for i in range(neg_preds.shape[0]):
                    y_pred.append(neg_preds[i].item())
                    y_true.append(0)

            if cal_mrr:
                # calculate MRR for each snap
                mrr, recall_at = self.eval_mrr_and_recall(edges_snap, np.repeat(time, edges_snap.shape[0]), self.edge_helper.node_num)
                mrr_hist.append(mrr)
        auc = self.auc_score(y_true, y_pred)
        ap = self.ap_score(y_true, y_pred)
        valid_loss = np.mean(all_loss)
        valid_acc = correct / num_samples
        if cal_mrr:
            valid_mrr = np.mean(mrr_hist)
            return valid_loss, valid_acc, auc, ap, valid_mrr
        else:
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

    def eval_mrr_and_recall(self, eval_edges, eval_timestamps, num_nodes, num_neg_per_node=1000):
        from datetime import datetime
        start = datetime.now()
        eval_sources, eval_destinations = eval_edges[:, 0], eval_edges[:, 1]

        # A list of source nodes to consider.
        src_lst = np.unique(eval_sources) # get unique senders.
        num_users = len(src_lst)

        src_features, dst_features = self.edge_helper.get_edges_feats(eval_sources, eval_destinations, eval_timestamps, window_size=self.window_size, concat=False)
        pos_preds = self.model.get_edges_embedding(src_features.to(self.device), dst_features.to(self.device))
        pos_preds = pos_preds.squeeze(dim=1)
        pos_labels = torch.ones(pos_preds.shape[0], dtype=torch.float, device=self.device)

        # generate negtive samples
        neg_sources, neg_destinations = gen_negative_edges(eval_sources, eval_destinations, num_nodes, num_neg_per_node)
        neg_timestamps = np.resize(eval_timestamps, neg_sources.shape)
        neg_src_features, neg_dst_features = self.edge_helper.get_edges_feats(neg_sources, neg_destinations, neg_timestamps, window_size=self.window_size, concat=False)
        neg_preds = self.model.get_edges_embedding(neg_src_features.to(self.device), neg_dst_features.to(self.device))
        neg_preds = neg_preds.squeeze(dim=1)
        neg_labels = torch.zeros(neg_preds.shape[0], dtype=torch.float, device=self.device)

        # The default setting, consider the rank of the most confident edge.
        from torch_scatter import scatter_max
        best_p_pos, _ = scatter_max(src=pos_preds, index=torch.from_numpy(eval_sources).to(self.device), dim_size=num_nodes)
        # best_p_pos has shape (num_nodes), for nodes not in src_lst has value 0.
        best_p_pos_by_user = best_p_pos[src_lst]

        uni, counts = np.unique(neg_sources,return_counts=True)
        # find index of first occurrence of each src in neg_sources 
        first_occ_idx = np.cumsum(counts,axis=0) - counts
        add = np.arange(num_neg_per_node)
        # take the first $num_neg_per_node$ negative edges from each src.
        score_idx = first_occ_idx.reshape(-1, 1) + add.reshape(1, -1)
        score_idx = torch.from_numpy(score_idx).long()
        p_neg_by_user = neg_preds[score_idx] # (num_users, num_neg_per_node)

        compare = (p_neg_by_user >= best_p_pos_by_user.reshape(num_users, 1)).float()
        assert compare.shape == (num_users, num_neg_per_node)
        # compare[i, j], for node i, the j-th negative edge's score > p_best.

        # counts 1 + how many negative edge from src has higher score than p_best.
        # if there's no such negative edge, rank is 1.
        rank_by_user = compare.sum(axis=1) + 1  # (num_users,)
        assert rank_by_user.shape == (num_users,)

        mrr = float(torch.mean(1 / rank_by_user))
        print(f'MRR={mrr}, time taken: {(datetime.now() - start).seconds} s')

        recall_at = dict()
        for k in [1, 3, 10]:
            recall_at[k] = float((rank_by_user <= k).float().mean())
        return mrr, recall_at

    def test(self, test_data):
        self.model.load_state_dict(torch.load(self.checkpt_file))
        self.model.eval()
        criterion = nn.BCELoss()
        correct = 0
        num_samples = 0
        all_loss = []
        test_batchsize = 2*self.batch_size
        y_true, y_pred = [], []
        mrr_hist = []
        for time in test_data.unique_times:
            edges_snap = self.edge_helper.time_edge_dict[time]['edges']
            num_instance = edges_snap.shape[0]
            num_batch = math.ceil(num_instance / test_batchsize)

            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_instance, start_idx + self.batch_size)
                sources_batch, destinations_batch = edges_snap[start_idx:end_idx, 0], edges_snap[start_idx:end_idx, 1]
                size = len(sources_batch)
                timestamps_batch = np.repeat(time, size)
                if args.seq_model in ['lstm', 'gru']:
                    ### get pos features    (sources_batch, destinations_batch, timestamps_batch)
                    src_features, dst_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
                    pos_preds = self.model.get_edges_embedding(src_features.to(self.device), dst_features.to(self.device))
                elif args.seq_model == 'transformer':
                    pos_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=True)
                    pos_preds = self.model(pos_features.to(self.device))
                pos_labels = torch.ones(pos_preds.shape[0], dtype=torch.float, device=self.device)
                pos_loss = criterion(pos_preds.squeeze(dim=1), pos_labels) ##pos_loss = -torch.log(pos_preds[:, 1])

                ### get negtive sample and features
                neg_destinations_batch = np.random.randint(0, self.edge_helper.node_num, size)
                if args.seq_model in ['lstm', 'gru']:
                    neg_src_features, neg_dst_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
                    neg_preds = self.model.get_edges_embedding(neg_src_features.to(self.device), neg_dst_features.to(self.device))
                elif args.seq_model == 'transformer':
                    neg_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=True)
                    neg_preds = self.model(neg_features.to(self.device))
                neg_labels = torch.zeros(neg_preds.shape[0], dtype=torch.float, device=self.device)
                neg_loss = criterion(neg_preds.squeeze(dim=1), neg_labels)
                loss = pos_loss + neg_loss ##loss = torch.mean(pos_loss + neg_loss)
                all_loss.append(loss.item())
               
                TP = torch.sum(pos_preds>=0.5)
                TN = torch.sum(neg_preds<0.5)

                correct += (TP + TN).item()
                num_samples += (pos_preds.shape[0] + neg_preds.shape[0])
                for i in range(pos_preds.shape[0]):
                    y_pred.append(pos_preds[i].item())
                    y_true.append(1)
                for i in range(neg_preds.shape[0]):
                    y_pred.append(neg_preds[i].item())
                    y_true.append(0)

            # calculate MRR for each snap
            mrr, recall_at = self.eval_mrr_and_recall(edges_snap, np.repeat(time, edges_snap.shape[0]), self.edge_helper.node_num)
            mrr_hist.append(mrr)

        test_auc = self.auc_score(y_true, y_pred)
        test_ap = self.ap_score(y_true, y_pred)
        test_loss = np.mean(all_loss)
        test_acc = correct / num_samples
        test_mrr = np.mean(mrr_hist)
        return test_loss, test_acc, test_auc, test_ap, test_mrr


if __name__ == "__main__":
    
    args = parameter_parser()
    print(args)
    execute = Execute(args)
    execute.train()
