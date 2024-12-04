import collections
import torch
from torch import nn, optim
from torch.nn import functional as F
import math
from args import args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embed_dim)

    def forward(self, x):
        x = self.embed(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(max_len, num_hidden)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, num_hidden, 2).float() * (-math.log(10000.0) / num_hidden))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        PE = PE.unsqueeze(0).transpose(0, 1)
        self.register_buffer('PE', PE)

    def forward(self, X):
        X = X + self.PE[:X.size(0), :]
        return X


class FFN(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.dense = nn.Linear(num_hidden, num_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, num_hidden, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(normalized_shape=num_hidden)

    def forward(self, x, y):
        y = self.dropout(y) + x
        return self.normalize(y)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_head, desc='enc'):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.desc = desc

        self.Wq = nn.Linear(self.num_hidden, self.num_hidden)
        self.Wk = nn.Linear(self.num_hidden, self.num_hidden)
        self.Wv = nn.Linear(self.num_hidden, self.num_hidden)

        self.relu = nn.ReLU()
        self.Q = nn.Sequential(self.Wq, self.relu)
        self.K = nn.Sequential(self.Wk, self.relu)
        self.V = nn.Sequential(self.Wv, self.relu)

    def forward(self, queries, keys, values, attention_mask):
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        q_split = torch.chunk(q, self.num_head, dim=-1)
        k_split = torch.chunk(k, self.num_head, dim=-1)
        v_split = torch.chunk(v, self.num_head, dim=-1)
        q_stack = torch.stack(q_split, dim=1)
        k_stack = torch.stack(k_split, dim=1)
        v_stack = torch.stack(v_split, dim=1)
        score = torch.matmul(q_stack, k_stack.permute(0, 1, 3, 2))
        score = score / (k_stack.size()[-1] ** 0.5)

        if self.desc == 'enc':
            enc_attention_mask = (attention_mask.unsqueeze(1)).unsqueeze(3).repeat(1,
                                                                                   self.num_head,
                                                                                   1,
                                                                                   1)
            enc_attention_mask = (enc_attention_mask == 0)
            score.masked_fill_(enc_attention_mask, -1e9)

        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_stack)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1), q.size(2)))
        a += queries
        return a


class EncoderBlock(nn.Module):
    def __init__(self, num_hidden, num_head, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_hidden=num_hidden,
                                            num_head=num_head,
                                            desc='enc')
        self.AddNorm1 = AddNorm(num_hidden, dropout)
        self.AddNorm2 = AddNorm(num_hidden, dropout)
        self.FFN = FFN(num_hidden=num_hidden)

    def forward(self, X, attention_mask):
        Y = self.AddNorm1(X, self.attention(X, X, X, attention_mask))
        outputs = self.AddNorm2(Y, self.FFN(Y))
        return outputs

    def load_parameters(self, params):
        for name, param in params.items():
            setattr(self, name, param)


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_hidden, num_head, num_block, num_classes, max_len, dropout):
        super().__init__()
        self.embedding = Embedding(vocab_size, num_hidden)
        self.pe = PositionalEncoding(num_hidden,
                                     dropout,
                                     max_len)
        self.layers = nn.ModuleList([EncoderBlock(num_hidden,
                                                  num_head,
                                                  dropout) for _ in range(num_block)])

        self.classifier = nn.Linear(num_hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        embedding = self.embedding(input_ids)
        hidden_state = self.pe(embedding.transpose(0, 1)).transpose(0, 1)

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)

        output = self.classifier(torch.mean(hidden_state, dim=1))
        return hidden_state, output


class EncoderMeta(nn.Module):
    def __init__(self, vocab_size, num_hidden, num_head, num_block, num_classes, max_len, dropout):
        super().__init__()
        self.embedding = Embedding(vocab_size, num_hidden)
        self.pe = PositionalEncoding(num_hidden,
                                     dropout,
                                     max_len)
        self.layers = nn.ModuleList([EncoderBlock(num_hidden,
                                                  num_head,
                                                  dropout) for _ in range(num_block)])

        self.classifier = nn.Linear(num_hidden, num_classes)
        self.num_head = num_head

    def forward(self, input_ids, attention_mask):
        embedding = self.embedding(input_ids)
        hidden_state = self.pe(embedding.transpose(0, 1)).transpose(0, 1)

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)

        output = self.classifier(torch.mean(hidden_state, dim=1))
        return hidden_state, output

    def functional_attention(self, hidden_state, attention_mask, params, layer_ids):
        q = F.linear(hidden_state,
                     weight=params[f'layers.{layer_ids}.attention.Wq.weight'],
                     bias=params[f'layers.{layer_ids}.attention.Wq.bias'])
        k = F.linear(hidden_state,
                     weight=params[f'layers.{layer_ids}.attention.Wk.weight'],
                     bias=params[f'layers.{layer_ids}.attention.Wk.bias'])
        v = F.linear(hidden_state,
                     weight=params[f'layers.{layer_ids}.attention.Wv.weight'],
                     bias=params[f'layers.{layer_ids}.attention.Wv.bias'])

        q_split = torch.chunk(q, self.num_head, dim=-1)
        k_split = torch.chunk(k, self.num_head, dim=-1)
        v_split = torch.chunk(v, self.num_head, dim=-1)
        q_stack = torch.stack(q_split, dim=1)
        k_stack = torch.stack(k_split, dim=1)
        v_stack = torch.stack(v_split, dim=1)
        score = torch.matmul(q_stack, k_stack.permute(0, 1, 3, 2))
        score = score / (k_stack.size()[-1] ** 0.5)
        enc_attention_mask = (attention_mask.unsqueeze(1)).unsqueeze(3).repeat(1,
                                                                               self.num_head,
                                                                               1,
                                                                               1)
        enc_attention_mask = (enc_attention_mask == 0)
        score.masked_fill_(enc_attention_mask, -1e9)
        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_stack)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1), q.size(2)))
        a += hidden_state
        return a

    def functional_encoder_block(self, hidden_state, attention_mask, params, layer_ids):
        attention_output = self.functional_attention(hidden_state, attention_mask, params, layer_ids)
        norm1_output = F.normalize(attention_output + hidden_state)
        ffn_output = F.linear(norm1_output,
                              weight=params[f'layers.{layer_ids}.FFN.dense.weight'],
                              bias=params[f'layers.{layer_ids}.FFN.dense.bias'])
        norm2_output = F.normalize(ffn_output + norm1_output)
        return norm2_output

    def functional_forward(self, input_ids, attention_mask, params):
        embedding_output = F.embedding(input_ids, params['embedding.embed.weight'])
        hidden_state = self.pe(embedding_output.transpose(0, 1)).transpose(0, 1)
        for layer_ids, layer in enumerate(self.layers):
            hidden_state = self.functional_encoder_block(hidden_state, attention_mask, params, layer_ids)

        output = torch.mean(hidden_state, dim=1)
        logit = F.linear(output, params['classifier.weight'], params['classifier.bias'])

        return logit

    def bottleneck(self, input_ids, attention_mask):
        embedding = self.embedding(input_ids)
        hidden_state = self.pe(embedding.transpose(0, 1)).transpose(0, 1)

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)

        output = self.classifier(torch.mean(hidden_state, dim=1))
        hidden_state = torch.mean(hidden_state, dim=0)
        loc = hidden_state.mean(dim=0)
        scale = hidden_state.std(dim=0)
        return torch.distributions.Normal(loc=loc, scale=scale), output


class Attenuator(nn.Module):
    def __init__(self, model):
        super().__init__()
        fast_weight = collections.OrderedDict(model.named_parameters())
        for i in range(args.num_block):
            fast_weight.pop(f'layers.{i}.AddNorm1.normalize.weight', None)
            fast_weight.pop(f'layers.{i}.AddNorm2.normalize.weight', None)
            fast_weight.pop(f'layers.{i}.AddNorm1.normalize.bias', None)
            fast_weight.pop(f'layers.{i}.AddNorm2.normalize.bias', None)
        num_layers = len(fast_weight.keys())
        self.attenuator = nn.Sequential(
            nn.Linear(num_layers, num_layers),
            nn.ReLU(),
            nn.Linear(num_layers, num_layers),
            nn.Sigmoid()
        ).to(device)

    def forward(self, grad):
        attenuated_grad = self.attenuator(grad)
        return attenuated_grad
