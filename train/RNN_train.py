import csv
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from Meta_dataset import DatasetAMPs
from transformers import AutoTokenizer
import torch
from model.RNN import RNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("../Pep",
                                          ignore_mismatched_sizes=True)


class TrainPriorRNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dataset = DatasetAMPs(path='dataset/Generator/AMP.csv')
        self.voc = self.dataset.amino_acid_vocab
        self.model = RNN(self.voc)

    def training_step(self, batch, batch_idx):
        seqs = batch['input_ids']
        log_p = self.model.likelihood(seqs)
        loss = - log_p.mean()
        self.log("train_loss", loss.clone().detach(),
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            save_model = 'model/RNN/RNN-Prior.ckpt'
            torch.save(self.model.rnn.state_dict(), save_model)

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=True, batch_size=256)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.rnn.parameters(), lr=1e-4)


class Sample(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dataset = DatasetAMPs(path='dataset/count.csv')
        self.voc = self.dataset.amino_acid_vocab
        self.model = RNN(self.voc)
        self.model.rnn.load_state_dict(torch.load('model/RNN/RNN-Prior.ckpt'))
        for param in self.model.rnn.parameters():
            param.requires_grad = False

    def predict_step(self, batch, batch_idx):
        seqs, likelihood, entropy = self.model.sample(batch_size=128)
        preds = []
        for seq in seqs:
            pred = [tokenizer.decode(g,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True) for g in seq]
            letters_only = [char for char in pred if char.isalpha()]
            letters_string = ''.join(letters_only)
            preds.append(letters_string)
        csv_file_name = 'Result/RNN-Generation.csv'
        with open(csv_file_name, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['SEQUENCE'])
            for letters_only in preds:
                writer.writerow([letters_only])

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=50, shuffle=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.rnn.parameters(), lr=1e-5)


if __name__ == "__main__":
    # ====================================== Train RNN ========================================
    logger = CSVLogger("logs", name="RNN-Train", version='Version0')
    model = TrainPriorRNN()
    trainer = pl.Trainer(max_epochs=51, logger=logger)
    trainer.fit(model)
    # ====================================== Sample ========================================
    # prediction_model = Sample()
    # trainer = pl.Trainer()
    # trainer.predict(prediction_model)














