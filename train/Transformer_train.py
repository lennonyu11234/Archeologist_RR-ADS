import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn import metrics
from args import args
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoTokenizer
import pandas as pd
from Meta_dataset import DatasetHemo
from model.Transformer_module import EncoderMeta
tokenizer = AutoTokenizer.from_pretrained("../Pep",
                                          ignore_mismatched_sizes=True)


class TrainTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dataset = DatasetHemo(path='Dataset/Hemolysis.csv')
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        self.model = EncoderMeta(vocab_size=25,
                                 num_hidden=args.num_hidden,
                                 num_head=args.num_head,
                                 num_classes=1,
                                 num_block=args.num_blocks,
                                 max_len=42,
                                 dropout=args.dropout)

        self.save_dir = 'Result/Generation/Final/Reg.csv'

    def training_step(self, batch, batch_ids):
        input_ids, input_mask, label = batch['input_ids'], batch['attention_mask'],\
            batch['label']

        criterion = nn.MSELoss()
        last_hidden_state, logit = self.model(input_ids, input_mask)

        loss = criterion(logit, label)
        self.log('Train_loss', loss.clone().detach(),
                 on_epoch=True, prog_bar=True,
                 on_step=False)

        return loss

    def validation_step(self, batch, batch_ids):
        input_ids, input_mask, label = batch['input_ids'], batch['attention_mask'], batch['label']
        last_hidden_state, logit = self.model(input_ids, input_mask)
        prediction = logit.squeeze(dim=1)
        r2 = metrics.r2_score(label.cpu().numpy(), prediction.cpu().numpy())
        self.log('Val_R2', r2,
                 on_epoch=True, prog_bar=True, on_step=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def predict_step(self, batch, batch_idx):
        input_ids, input_mask, label = batch['input_ids'], batch['attention_mask'], batch['label']
        last_hidden_state, logit = self.model(input_ids, input_mask)
        preds = logit.squeeze(dim=1)

        indices = input_ids.squeeze(dim=0)
        mask = indices != 1
        trimmed_tensor = indices[mask]
        sequence = tokenizer.decode(trimmed_tensor.tolist())
        sequence = sequence[3:-4]

        all_rows = []
        all_rows.append({
            'SEQUENCE': str(sequence),
            'Prediction': preds.item()
        })
        df = pd.DataFrame(all_rows)
        df.to_csv(self.save_dir, index=False, header=False, mode='a')

    def predict_dataloader(self):
        dataset = DatasetHemo(path='Dataset/Hemolysis_test.csv')
        return DataLoader(dataset, batch_size=1, shuffle=False)


if __name__ == '__main__':
    # ====================================== train ==========================================
    checkpoint_callback = ModelCheckpoint(
                monitor='Val_R2',
                mode='max',
                save_top_k=1,
                save_last=False
            )
    logger = CSVLogger("logs", name="Score Function", version='HEMO')
    model = TrainTransformer()
    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback],
                         logger=logger)
    trainer.fit(model)
    # ====================================== Predict ==========================================
    # predict_model = TrainTransformer.load_from_checkpoint(
    #     checkpoint_path='logs/Score Function/Regression/checkpoints/epoch=499-step=1500.ckpt'
    # )
    # trainer = pl.Trainer()
    # trainer.predict(predict_model)


