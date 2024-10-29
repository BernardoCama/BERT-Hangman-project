
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import math
from torchprofile import profile_macs

from Classes.utils.utils import clone
from Classes.dataset.dataset import Vocabulary


class Bert(nn.Module):
    def __init__(self, params):

        super().__init__()
        self.params = params
        self.vocabulary = Vocabulary()

        # Define the embedding layer (word embeddings + positional encodings) 
        position = PositionalEncoding(d_model=params['hidden_size'], dropout=0.1)
        self.embed = nn.Sequential(Embeddings(d_model=params['hidden_size'], vocab=self.vocabulary.vocab_size), position)

        # Define the components of the encoder
        self_attn = MultiHeadedAttention(h=params['n_heads'], d_model=params['hidden_size'], d_k=params['hidden_size'] // params['n_heads'], d_v=params['hidden_size'] // params['n_heads'], dropout=0.1)
        feed_forward = FullyConnectedFeedForward(d_model=params['hidden_size'], d_ff=params['d_ff'], dropout=0.1)
        # Define the encoder and the layers
        self.encoder = Encoder(self_attn=self_attn, feed_forward=feed_forward, size=params['hidden_size'], dropout=0.1)
        self.layers = clone(self.encoder, params['n_encoders'])

        # Define the generator (output layer)
        self.generator = Generator(d_model=params['hidden_size'], vocab_size=self.vocabulary.vocab_size)

        # Define the layer normalization (along the embedding dimension)
        self.layer_norm = LayerNorm(self.encoder.size)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor, not_present: torch.Tensor):
        """
        :param x: shape (batch_size, max_word_length)
        :param src_mask: shape (batch_size, 1, max_word_length)
        :param not_present: shape (batch_size, vocab_size)
        :return:
        """
        x = self.embed(x)

        # Modify embeddings based on not_present information
        # (batch_size, max_word_length, embedding_dim) = (batch_size, 1, vocab_size)   (1, vocab_size, embedding_dim)
        # torch.matmul (do not consider batch size) is used to compute a weighted sum of the embedding vectors based on the not_present tensor,
        # which is then added to the original embeddings x.
        not_present_emb = torch.matmul(not_present.unsqueeze(1).float(), self.embed[0].lut.weight.unsqueeze(0))
        x = x + not_present_emb

        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)

    @property
    def device(self):
        return self.generator.linear.weight.device

    def print_MACs_FLOPs(self):
    
        num_macs = profile_macs(self, (torch.zeros(self.params['batch_size'], 17, dtype=torch.long), torch.zeros(self.params['batch_size'], 1, 17), torch.zeros(self.params['batch_size'], 28))) 
        print("#MACs:", num_macs)
        print("#FLOPs:", num_macs*2)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(in_features=d_model,
                                out_features=vocab_size)

    def forward(self, x, exist_mask = None):
        result, _ = torch.max(self.linear(x), dim=1)
        if exist_mask is not None:
            result = result.masked_fill_(exist_mask == 1, -1e9)
        return F.log_softmax(result, dim=1)



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.query_linear = nn.Linear(in_features=d_k * h,
                                      out_features=d_model,
                                      bias=False)
        self.key_linear = nn.Linear(in_features=d_k * h,
                                    out_features=d_model,
                                    bias=False)
        self.value_linear = nn.Linear(in_features=d_v * h,
                                      out_features=d_model,
                                      bias=False)

        self.attn = None  # not used for computation, only for visualization
        self.dropout = nn.Dropout(p=dropout)

        self.output_linear = nn.Linear(in_features=d_model,
                                       out_features=h * d_v)

    def forward(self, query, key, value, mask=None):
        """
        d_k * h = d_model

        query: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size,
        key: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size
        value: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size,

        output: shape (batch_size, max_sent_length, embedding_size)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        d_k = self.d_model // self.h

        n_batches = query.size(0)
        max_sent_length = query.size(1)

        query = self.query_linear(query).view(n_batches, max_sent_length, self.h, d_k).transpose(1, 2)
        key = self.key_linear(key).view(n_batches, key.size(1), self.h, d_k).transpose(1, 2)
        value = self.value_linear(value).view(n_batches, value.size(1), self.h, d_k).transpose(1, 2)

        # scores shape: (batch_size, h, max_sent_length, d_k)
        scores, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concat attention scores over multiple heads
        # (batch_size, max_sent_length, d_model)
        scores = scores.transpose(1, 2).contiguous().view(n_batches, max_sent_length, self.h * d_k)

        return self.output_linear(scores)

def attention(query, key, value, mask=None, dropout=None):
    """
    query: shape (*, n_queries, d_k) n_queries is the maximum sentence length / max_sent_length - 1 if key from decoder
    key: (*, K, d_k) , K is the maximum sentence length / max_sent_length - 1 if key from decoder
    value: (*, K, d_v)

    scores: (n_quires, K)
    output: (n_queries, d_v)
    """

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -np.inf)
    p = F.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)

    return torch.matmul(p, value), p



class FullyConnectedFeedForward(nn.Module):
    """
    A fully connected neural network with Relu activation
    input: d_model
    hidden: d_ff
    output: d_model

    Implements FFN equation.
    FFN(x) = max(0, xW_1 + b)W_2 + b

    It consist of two linear layer and a Relu activation in between

    Linear_2(Relu(Linear_1(x))))
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FullyConnectedFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: shape (batch_size, max_sent_len, embedding_size/d_model)
        :return: output: shape (batch_size, max_sent_len, embedding_size/d_model)
        """
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
    

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape: (batch_size, max_word_length, d_model = embedding_size)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x.shape: (batch_size, max_word_length)
        return self.lut(x) * math.sqrt(self.d_model)
    

class Encoder(nn.Module):
    def __init__(self, self_attn, feed_forward, size, dropout):
        super(Encoder, self).__init__()
        self.sub_layers = clone(SublayerSkipConnection(size, dropout), 2)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x, mask):
        # x.shape: (batch_size, max_word_length, d_model = embedding_size)
        x = self.sub_layers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sub_layers[1](x, self.feed_forward)
    

class SublayerSkipConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerSkipConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2