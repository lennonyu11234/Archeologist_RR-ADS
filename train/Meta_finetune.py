import numpy as np
import torch
from Meta_dataset import MetaDataset, DataLoader
from model.Transformer_module import EncoderMeta
from model.Meta_loop import meta_loop_test
from args import args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acc_list, precision_list, recall_list, f1_list, mcc_list = [], [], [], [], []
acc_mean_list, precision_mean_list, recall_mean_list, f1_mean_list, mcc_mean_list = [], [], [], [], []

test_dataset = MetaDataset(data_path=f'dataset/Test/AB',
                           task_num=1,
                           k_shot=args.k_shot,
                           q_query=200)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

meta = EncoderMeta(vocab_size=args.vocab_size,
                   num_hidden=args.num_hidden,
                   num_head=args.num_head,
                   num_block=args.num_block,
                   num_classes=args.num_classes,
                   max_len=args.max_len,
                   dropout=args.dropout).to(device)
meta.load_state_dict(torch.load('model/RR-ADS.pth'))
meta.eval()

loss, acc, precision, recall, f1, mcc = meta_loop_test(dataloader=test_loader,
                                                       model=meta,
                                                       inner_step=args.inner_step,
                                                       inner_lr=args.inner_lr,
                                                       is_save=True)

acc_list.append(acc)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)
mcc_list.append(mcc)

acc_mean = np.mean(acc_list)
precision_mean = np.mean(precision_list)
recall_mean = np.mean(recall_list)
f1_mean = np.mean(f1_list)
mcc_mean = np.mean(mcc_list)

acc_mean_list.append(acc_mean)
precision_mean_list.append(precision_mean)
recall_mean_list.append(recall_mean)
f1_mean_list.append(f1_mean)
mcc_mean_list.append(mcc_mean)

print(f'\n Accuracy: {np.mean(acc_list):.4f}'
      f'\n Precision: {np.mean(precision_list):.4f}'
      f'\n Recall: {np.mean(recall_list):.4f}'
      f'\n F1-Score: {np.mean(f1_list):.4f}'
      f'\n MCC: {np.mean(mcc_list):.4f}')











