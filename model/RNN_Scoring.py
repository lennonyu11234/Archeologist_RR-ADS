import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics
from args import args
from Meta_dataset import DatasetAMPs
from Transformer_module import Encoder, EncoderMeta
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FewShotScoring(nn.Module):
    def __init__(self, scoring_idx):
        super().__init__()
        self.model = EncoderMeta(vocab_size=25,
                                 num_hidden=args.num_hidden,
                                 num_head=args.num_head,
                                 num_block=args.num_block,
                                 num_classes=args.num_classes,
                                 max_len=42,
                                 dropout=args.dropout).to(device)
        self.idx = scoring_idx

    def forward(self, input_ids, input_mask):
        loaded_fast_weight = torch.load(f'model/1-shot/AB{self.idx}.pth')
        logit = self.model.functional_forward(input_ids,
                                              input_mask,
                                              loaded_fast_weight)
        _, predict = torch.max(logit, dim=1)
        value0, value1 = -0.1, 0.1
        if self.idx == 0:
            value0, value1 = -0.3, 0.3
        elif self.idx in [1, 2, 3, 4]:
            value0, value1 = -0.2, 0.2
        elif self.idx in [5, 6, 7]:
            value0, value1 = -0.25, 0.25
        final_output = torch.tensor(predict)
        final_output = torch.where(final_output == 0, value0, final_output)
        final_output = torch.where(final_output == 1, value1, final_output)
        return final_output











