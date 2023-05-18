import torch.nn as nn
import torch
import torch.nn.functional as F

from .transformer import TransformerBlock


class linkPredictor(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, device, in_dim, hidden=256, n_layers=8, attn_heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.device = device
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4
        
        self.fc = nn.Linear(in_features=in_dim, out_features=hidden)
        self.fc1 = nn.Linear(in_features=hidden, out_features=hidden)
        self.fc2 = nn.Linear(in_features=hidden, out_features=1)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        seq_len = x.shape[1]
        bs = x.shape[0]
        mask = torch.ones([bs, 1, seq_len, seq_len], device=self.device)
        
        x = torch.tanh(self.fc(x))
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x = F.relu(self.fc1(x))
        #x = F.sigmoid(self.fc2(x))
        x = torch.nn.Sigmoid()(self.fc2(x))
        return x[:, -1, :]
