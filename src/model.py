import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class LSTM_Emb(nn.ModuleList):
    def __init__(self, batch_size, hidden_dim, lstm_layers, emb_size, dropout, device):
        super(LSTM_Emb, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = emb_size

        self.dropout = nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc0 = nn.Linear(in_features=self.input_size, out_features=self.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
        self.device = device

    def forward(self, x):
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim), device=x.device)
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim), device=x.device)

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = torch.tanh(self.fc0(x))
        out = self.dropout(out)
        out, (hidden, cell) = self.lstm(out, (h,c))
        return out

class GRU_Emb(nn.ModuleList):
    def __init__(self, batch_size, hidden_dim, lstm_layers, emb_size, dropout, device):
        super(GRU_Emb, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = emb_size # embedding dimention

        self.dropout = nn.Dropout(dropout)##nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.fc0 = nn.Linear(in_features=self.input_size, out_features=self.hidden_dim)
        self.gru = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
        self.device = device

    def forward(self, x):
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim), device=self.device)

        torch.nn.init.xavier_normal_(h)

        out = torch.tanh(self.fc0(x))
        out = self.dropout(out)
        out, hidden = self.gru(out, h)
        return out


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        #print("MergeLayer/forward/")
        #pdb.set_trace()
        h = self.act(self.fc1(x[:,-1,:])) #x[:,-1,:] for lstm(with out shape: [128, 20, 64])
        out =  torch.nn.Sigmoid()(self.fc2(h))
        return out

class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, dim)
        self.fc_2 = torch.nn.Linear(dim, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x[:,-1,:])) #x[:,-1,:] for lstm(with out shape: [128, 20, 64])
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        #return self.fc_3(x).squeeze(dim=1)
        return torch.nn.Sigmoid()(self.fc_3(x))

class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]

        # output has shape [batch_size, seq_len, dimension]
        '''if isinstance(t, PackedSequence):
            output = torch.cos(self.w(t.data))

            return PackedSequence(output, t.batch_sizes).to(t.data.device)
        if isinstance(t, torch.Tensor):'''
        t = t.unsqueeze(-1)
        output = torch.cos(self.w(t))
        return output

class TimeEncodeMixer(torch.nn.Module):
    """
    # Time Encoding proposed by Mixer
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncodeMixer, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        t = t.unsqueeze(-1)
        output = torch.cos(self.w(t))
        return output

'''class TimeEncodeNoParam(torch.nn.Module): -- ld 实现
    # Time Encoding proposed by TGAT
    def __init__(self, dimension=100, alpha=10., beta=10.):
        super(TimeEncodeNoParam, self).__init__()

        self.dimension = dimension
        self.alpha = alpha
        self.beta = beta
        self.w = self.alpha ** (-torch.arange(self.dimension) / self.beta)

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        t = t.unsqueeze(-1)
        output = torch.cos(self.w * t)
        return output'''

class MyLinkPrediction(nn.ModuleList):
    def __init__(self, args, device):
        super(MyLinkPrediction, self).__init__()

        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.LSTM_layers = args.lstm_layers
        self.nodes_embedding_size = args.emb_size
        self.dropout = args.dropout
        self.embedding_module_type = args.seq_model
        print('self.embedding_module_type: ', self.embedding_module_type)
        if self.embedding_module_type=='lstm':
            self.embedding_model = LSTM_Emb(self.batch_size, self.hidden_dim, self.LSTM_layers, self.nodes_embedding_size, self.dropout, device)
        elif self.embedding_module_type=='gru':
            self.embedding_model = GRU_Emb(self.batch_size, self.hidden_dim, self.LSTM_layers, self.nodes_embedding_size, self.dropout, device)
        self.decoder = MergeLayer(self.hidden_dim, self.hidden_dim, self.hidden_dim, 1)

    def get_nodes_embedding(self, source_feats, destination_feats):
        src_emb = self.embedding_model(source_feats)
        dst_emb = self.embedding_model(destination_feats)
        return src_emb, dst_emb
    
    def get_edges_embedding(self, source_feats, destination_feats):
        src_emb, dst_emb = self.get_nodes_embedding(source_feats, destination_feats)
        edge_prob = self.decoder(src_emb, dst_emb)
        return edge_prob

