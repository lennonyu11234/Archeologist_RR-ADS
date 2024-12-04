import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MultiGRU(nn.Module):
    def __init__(self, voc_size=24):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, 512)
        self.gru_1 = nn.GRUCell(512, 1024)
        self.gru_2 = nn.GRUCell(1024, 1024)
        self.gru_3 = nn.GRUCell(1024, 1024)
        self.linear = nn.Linear(1024, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = torch.tensor(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        return torch.tensor(torch.zeros(3, batch_size, 1024))


class RNN(nn.Module):
    def __init__(self, voc):
        super().__init__()
        self.rnn = MultiGRU()
        self.voc = voc

    def likelihood(self, target):
        batch_size, seq_length = target.size()

        start_token = torch.tensor(torch.zeros(batch_size, 1).long()).to(device)
        start_token[:] = self.voc['<s>']
        start_token.to(device)
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        log_probs = torch.tensor(torch.zeros(batch_size)).to(device)

        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h.to(device))
            log_prob = F.log_softmax(logits).to(device)
            prob = F.softmax(logits)

            log_probs += NLLLoss(log_prob, target[:, step])

        return log_probs

    def sample(self, batch_size, max_length=15):
        start_token = torch.tensor(torch.zeros(batch_size).long())
        start_token[:] = self.voc['<s>']
        start_token.to(device)
        h = self.rnn.init_h(batch_size).to(device)
        x = start_token.to(device)

        sequences = []
        log_probs = torch.tensor(torch.zeros(batch_size)).to(device)

        finished = torch.zeros(batch_size).byte().to(device)
        entropy = torch.tensor(torch.zeros(batch_size)).to(device)
        for step in range(max_length):
            logits, h = self.rnn(x, h.to(device))
            prob = F.softmax(logits)
            log_prob = F.log_softmax(logits)
            x = torch.multinomial(prob, 1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs += NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = torch.tensor(x.data)
            EOS_sampled = (x == self.voc['</s>']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break
        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy


def NLLLoss(inputs, targets):
    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).to(device)
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = torch.tensor(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss



