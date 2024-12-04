from torch import optim
import torch
from tqdm import tqdm
from Meta_dataset import MetaDataset, DataLoader
from model.Transformer_module import EncoderMeta, Attenuator
from model.Meta_loop import meta_loop
from args import args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = MetaDataset(data_path='dataset/Train',
                            task_num=args.num_task,
                            k_shot=args.k_shot,
                            q_query=args.q_query)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

meta = EncoderMeta(vocab_size=25,
                   num_hidden=args.num_hidden,
                   num_head=args.num_head,
                   num_block=args.num_block,
                   num_classes=args.num_classes,
                   max_len=42,
                   dropout=args.dropout).to(device)
attenuator = Attenuator(meta)
maml_parameters = meta.parameters()
attenuator_parameters = attenuator.parameters()
all_params = [*maml_parameters, *attenuator_parameters]
optimizer = optim.Adam(all_params, args.outer_lr)
best_acc, best_precision = 0, 0

meta.train()
train_bar = tqdm(range(args.epochs), total=args.epochs, desc='Outer Loop')
for i in train_bar:
    train_loss = meta_loop(loader=train_dataset,
                           model=meta,
                           inner_step=args.inner_step,
                           inner_lr=args.inner_lr,
                           optimizer=optimizer,
                           attenuator=attenuator)
    train_bar.set_postfix(loss="{:.4f}".format(train_loss.item()))
