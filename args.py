import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--task_num', type=int, default=23,
                    help='Number of task per train batch.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='The outer training epochs.')
parser.add_argument('--inner_step', type=int, default=10,
                    help='The inner training epochs.')
parser.add_argument('--inner_lr', type=float, default=1e-2,
                    help='The learning rate of of the support set.')
parser.add_argument('--outer_lr', type=float, default=1e-2,
                    help='The learning rate of of the query set.')
parser.add_argument('--k_shot', type=int, default=5,
                    help='The number of support set image for every task.')
parser.add_argument('--q_query', type=int, default=10,
                    help='The number of query set image for every task.')
parser.add_argument('--num_hidden', type=int, default=512,
                    help='number of hidden states')
parser.add_argument('--num_head', type=int, default=8,
                    help='number of heads')
parser.add_argument('--num_block', type=int, default=4,
                    help='number of blocks')
parser.add_argument('--num_classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout')
parser.add_argument('--epsilon', type=float, default=1e-6,
                    help='epsilon')
parser.add_argument('--eta', type=float, default=5.0,
                    help='eta')
parser.add_argument('--beta', type=float, default=5.0,
                    help='beta')
args = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




