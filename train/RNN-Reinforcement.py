import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from Meta_dataset import DatasetAMPs
import torch
from model.RNN import RNN
from transformers import AutoTokenizer
from model.RNN_Scoring import FewShotScoring
from utils.RNN_utils import unique, idx_to_seqs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("../Pep",
                                          ignore_mismatched_sizes=True)


class Customize(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.restore_prior_from = 'model/RNN/RNN-Prior.ckpt'
        self.restore_agent_from = 'model/RNN/RNN-Prior.ckpt'
        self.agent_save = 'model/RNN/Agent-RNN.ckpt'
        self.sigma = 60
        self.save_dir = 'Result/Generation/Prior/'
        self.dataset = DatasetAMPs(path='dataset/AB.csv')
        vocab = self.dataset.amino_acid_vocab

        self.Prior = RNN(vocab)
        self.Agent = RNN(vocab)
        self.Prior.rnn.load_state_dict(torch.load(self.restore_prior_from))
        self.Agent.rnn.load_state_dict(torch.load(self.restore_agent_from))
        for param in self.Prior.rnn.parameters():
            param.requires_grad = False

        self.scoring_model_list = [FewShotScoring(scoring_idx=0),
                                   FewShotScoring(scoring_idx=1),
                                   FewShotScoring(scoring_idx=2),
                                   FewShotScoring(scoring_idx=3),
                                   FewShotScoring(scoring_idx=4),
                                   FewShotScoring(scoring_idx=5),
                                   FewShotScoring(scoring_idx=6),
                                   FewShotScoring(scoring_idx=7),
                                   FewShotScoring(scoring_idx=8),
                                   FewShotScoring(scoring_idx=9)]
        for model in self.scoring_model_list:
            for param in model.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        seq_save = []
        seqs, agent_likelihood, entropy = self.Agent.sample(batch_size=256)
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]

        prior_likelihood = self.Prior.likelihood(torch.tensor(seqs))
        sequences = idx_to_seqs(seqs)
        padded_sequences = []
        for seq in sequences:
            tokens = [tokenizer.bos_token] + list(seq) + [tokenizer.eos_token]
            if len(tokens) < self.dataset.sequence_max_length:
                tokens += [tokenizer.pad_token] * (self.dataset.sequence_max_length - len(tokens))
            padded_sequences.append(tokens)
        input_ids, input_mask = [], []
        for pad_seq in padded_sequences:
            token_indices = tokenizer.convert_tokens_to_ids(pad_seq)
            input_indices = torch.tensor(token_indices, dtype=torch.long)
            attention_mask = (input_indices != tokenizer.vocab[tokenizer.pad_token]).long()
            input_ids.append(input_indices)
            input_mask.append(attention_mask)
        input_ids = torch.stack(input_ids).to(device)
        input_mask = torch.stack(input_mask).to(device)

        seq_save.extend(sequences)

        # ===========score===============
        scoring_list = []
        for model in self.scoring_model_list:
            score = model(input_ids, input_mask)
            scoring_list.append(score.list())

        all_rows = []
        for sequence in seq_save:
            all_rows.append({
                'SEQUENCE': str(sequence)
            })
        for idx, score in enumerate(scoring_list):
            for s in score:
                all_rows.append({
                    f'Score{idx}': s.item()
                })

        # ============save====================
        df = pd.DataFrame(all_rows)
        if batch_idx % 5 == 0 and batch_idx != 0:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            df.to_csv(self.save_dir + f'sampled_{batch_idx}.csv', index=False, header=True)

        if batch_idx % 100 == 0 and batch_idx != 0:
            torch.save(self.Agent.rnn.state_dict(), self.agent_save)

        # ================cal loss=======================
        scoring_tensor = torch.tensor(scoring_list)
        score_final = scoring_tensor.mean(dim=0)

        augmented_likelihood = prior_likelihood + self.sigma * torch.tensor(score_final)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        loss = loss.mean()
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        self.log("train_loss", loss.clone().detach(),
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=True, drop_last=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.Agent.rnn.parameters(), lr=1e-5)


if __name__ == "__main__":
    # ====================================== Reinforce ========================================
    logger = CSVLogger("logs", name="Reinforce Learning", version='version0')
    KD_model = Customize()
    trainer = pl.Trainer(max_epochs=1, logger=logger)
    trainer.fit(KD_model)