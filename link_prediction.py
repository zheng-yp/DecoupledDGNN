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
        self.model = MyLinkPrediction(args, self.device)
        self.model = self.model.to(self.device)

        self.window_size = args.window_size
        self.inductive = args.inductive

        self.shuffle = args.shuffle
        self.checkpt_path = args.checkpt_path
        self.checkpt_file = self.checkpt_path + '/' + self.data + '_ws'+str(self.window_size)+'_best.pt'
        self.patience = args.patience

    def train(self):
        if self.inductive:
            full_data, train_data, _, _, val_data, test_data = get_data_inductive(self.data, shuffle=self.shuffle)
        else:
            full_data, train_data, val_data, test_data, _, _ = get_data_transductive(self.data, shuffle=self.shuffle)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / self.batch_size)
        print('num of training instances: {}'.format(num_instance))
        print('num of batches per epoch: {}'.format(num_batch))
        print('window_size: ', self.window_size)
        self.edge_helper = EdgeHelper(self.data, inductive=self.inductive)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        best_ap, best_epoch = 0, 0
        for epoch in range(args.epochs):
            self.model.train()
            correct = 0
            num_samples = 0
            all_loss = []
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_instance, start_idx + self.batch_size)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]

                size = len(sources_batch)
                ### get pos features    (sources_batch, destinations_batch, timestamps_batch)
                src_features, dst_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)

                pos_preds = self.model.get_edges_embedding(src_features.to(self.device), dst_features.to(self.device))
                
                pos_labels = torch.ones(pos_preds.shape[0], dtype=torch.float, device=self.device)
                pos_loss = criterion(pos_preds.squeeze(), pos_labels) ##pos_loss = -torch.log(pos_preds[:, 1])

                ### get negtive sample and features
                neg_destinations_batch = np.random.randint(0, self.edge_helper.node_num, size)
                neg_src_features, neg_dst_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)

                neg_preds = self.model.get_edges_embedding(neg_src_features.to(self.device), neg_dst_features.to(self.device))
                
                neg_labels = torch.zeros(neg_preds.shape[0], dtype=torch.float, device=self.device)
                neg_loss = criterion(neg_preds.squeeze(), neg_labels) ##neg_loss = -torch.log(neg_preds[:, 0])

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
        test_loss, test_acc, test_auc, test_ap = self.test(test_data)
        print("Test loss: %.5f, Test accuracy: %.5f, Test AUC: %.5f, Test AP: %.5f" % (test_loss, test_acc, test_auc, test_ap))
        
    def valid(self, val_data):
        criterion = nn.BCELoss()
        self.model.eval()
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
            size = len(sources_batch)
            ### get pos features    (sources_batch, destinations_batch, timestamps_batch)
            src_features, dst_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
            pos_preds = self.model.get_edges_embedding(src_features.to(self.device), dst_features.to(self.device))
            pos_labels = torch.ones(pos_preds.shape[0], dtype=torch.float, device=self.device)
            pos_loss = criterion(pos_preds.squeeze(), pos_labels) ##pos_loss = -torch.log(pos_preds[:, 1])

            ### get negtive sample and features
            neg_destinations_batch = np.random.randint(0, self.edge_helper.node_num, size)
            neg_src_features, neg_dst_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
            neg_preds = self.model.get_edges_embedding(neg_src_features.to(self.device), neg_dst_features.to(self.device))
            neg_labels = torch.zeros(neg_preds.shape[0], dtype=torch.float, device=self.device)
            neg_loss = criterion(neg_preds.squeeze(), neg_labels)
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

    def test(self, test_data):
        self.model.load_state_dict(torch.load(self.checkpt_file))
        self.model.eval()
        criterion = nn.BCELoss()
        correct = 0
        num_samples = 0
        all_loss = []

        num_instance = len(test_data.sources)
        # full batch for test
        test_batchsize = num_instance
        num_batch = math.ceil(num_instance / test_batchsize)
        y_true, y_pred = [], []
        for batch_idx in range(num_batch):
            start_idx = batch_idx * test_batchsize
            end_idx = min(num_instance, start_idx + test_batchsize)
            sources_batch, destinations_batch = test_data.sources[start_idx:end_idx], \
                                                test_data.destinations[start_idx:end_idx]
            edge_idxs_batch = test_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = test_data.timestamps[start_idx:end_idx]
            size = len(sources_batch)
            ### get pos features    (sources_batch, destinations_batch, timestamps_batch)
            src_features, dst_features = self.edge_helper.get_edges_feats(sources_batch, destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
            pos_preds = self.model.get_edges_embedding(src_features.to(self.device), dst_features.to(self.device))
            pos_labels = torch.ones(pos_preds.shape[0], dtype=torch.float, device=self.device)
            pos_loss = criterion(pos_preds.squeeze(), pos_labels) ##pos_loss = -torch.log(pos_preds[:, 1])

            ### get negtive sample and features
            neg_destinations_batch = np.random.randint(0, self.edge_helper.node_num, size)
            neg_src_features, neg_dst_features = self.edge_helper.get_edges_feats(sources_batch, neg_destinations_batch, timestamps_batch, window_size=self.window_size, concat=False)
            neg_preds = self.model.get_edges_embedding(neg_src_features.to(self.device), neg_dst_features.to(self.device))
            neg_labels = torch.zeros(neg_preds.shape[0], dtype=torch.float, device=self.device)
            neg_loss = criterion(neg_preds.squeeze(), neg_labels)
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
        test_auc = self.auc_score(y_true, y_pred)
        test_ap = self.ap_score(y_true, y_pred)
        test_loss = np.mean(all_loss)
        test_acc = correct / num_samples
        return test_loss, test_acc, test_auc, test_ap

if __name__ == "__main__":
    
    args = parameter_parser()
    print(args)    
    execute = Execute(args)
    execute.train()

