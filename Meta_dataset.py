import os
import random
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Pep",
                                          ignore_mismatched_sizes=True)


class MetaDataset(Dataset):
    def __init__(self, data_path, task_num, k_shot, q_query):
        # a[0]['support_pairs'][0]['sequence'] a[task number][][pair number][]
        self.file_list = self.get_file_list(data_path)
        self.task_num = task_num
        self.k_shot = k_shot
        self.q_query = q_query
        self.max_length = 42
        self.data_path = data_path

        self.all_data = self.get_whole_data()

    def __len__(self):
        return self.task_num

    def __getitem__(self, task_idx):
        task = self.all_data[task_idx]
        return task

    def get_file_list(self, data_path):
        items = os.listdir(data_path)
        file_names = [item for item in items if os.path.isfile(os.path.join(data_path, item))]
        return file_names

    def read_csv_file(self, file_name):
        df = pd.read_csv(file_name, encoding_errors='replace')
        sequences = df['SEQUENCE'].tolist()
        labels = df['MIC'].tolist()

        processed_labels = []
        for label in labels:
            if label <= 16:
                label = 1
            else:
                label = 0
            processed_labels.append(torch.tensor(label))

        processed_sequences = []
        attention_masks = []
        for sequence in sequences:
            tokens = [tokenizer.bos_token] + list(sequence) + [tokenizer.eos_token]
            if len(tokens) < self.max_length:
                tokens += [tokenizer.pad_token] * (self.max_length - len(tokens))
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            processed_sequences.append(input_ids)
            attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long()
            attention_masks.append(attention_mask)

        return processed_sequences, attention_masks, processed_labels

    def get_one_task_data(self, csv_file):
        support_pairs, query_pairs = [], []
        positive_pairs = []
        negative_pairs = []

        input_ids, attention_masks, labels = self.read_csv_file(csv_file)
        for sequence, attention_mask, label in zip(input_ids, attention_masks, labels):
            pair = {
                'sequence': sequence,
                'attention_mask': attention_mask,
                'label': label}
            if label <= 16:
                positive_pairs.append(pair)
            else:
                negative_pairs.append(pair)

        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)
        support_positive_pairs = positive_pairs[:self.k_shot]
        support_negative_pairs = negative_pairs[:self.k_shot]

        query_positive_pairs = positive_pairs[self.k_shot: self.k_shot+self.q_query]
        query_negative_pairs = negative_pairs[self.k_shot: self.k_shot+self.q_query]

        support_pairs.extend(support_positive_pairs)
        support_pairs.extend(support_negative_pairs)
        query_pairs.extend(query_positive_pairs)
        query_pairs.extend(query_negative_pairs)
        random.shuffle(support_pairs)
        random.shuffle(query_pairs)

        return support_pairs, query_pairs

    def get_whole_data(self):
        tasks = []
        for idx, csv_file in enumerate(self.file_list):
            support_pairs, query_pairs = self.get_one_task_data(f'{self.data_path}/{csv_file}')
            task = {
                'idx': idx,
                'support_set': support_pairs,
                'query_set': query_pairs
            }
            tasks.append(task)
        return tasks


class DatasetAMPs(Dataset):
    def __init__(self, path):
        with open('Pep/voc.json', 'r', encoding='utf-8') as f:
            self.amino_acid_vocab = json.load(f)
        self.sequences, self.label = self.read_file_csv(path)
        self.sequence_max_length = self.get_max_length(self.sequences) + 2
        self.padded_sequences = self.pad_sequences(self.sequences, self.sequence_max_length)

    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, idx):
        tokens = self.padded_sequences[idx]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.tensor(self.label[idx])
        if labels >= 16:
            labels = 0
        else:
            labels = 1

        attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }

    def read_file_csv(self, path):
        df = pd.read_csv(path, encoding='ISO-8859-1')
        sequences = df['SEQUENCE'].tolist()
        label = df['MIC'].tolist()
        return sequences, label

    def get_max_length(self, sequences):
        return max(len(seq) for seq in sequences)

    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            # Add SOS and EOS tokens
            tokens = [tokenizer.bos_token] + list(seq) + [tokenizer.eos_token]
            # Pad with <pad> tokens
            if len(tokens) < max_length:
                tokens += [tokenizer.pad_token] * (max_length - len(tokens))
            padded_sequences.append(tokens)
        return padded_sequences


class DatasetHemo(Dataset):
    def __init__(self, path):
        with open('F:/Project/Meta/Pep/voc.json', 'r', encoding='utf-8') as f:
            self.amino_acid_vocab = json.load(f)
        self.sequences, self.labels = self.read_file_csv(path)
        self.sequence_max_length = 42
        self.padded_sequences = self.pad_sequences(self.sequences, self.sequence_max_length)

    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, idx):
        tokens = self.padded_sequences[idx]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        labels = torch.tensor(self.labels[idx])
        attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels.clone().detach().float()
        }

    def read_file_csv(self, path):
        df = pd.read_csv(path)
        sequences = df['sequence'].tolist()
        hemolysis = df['Hemolysis'].tolist()
        concentration = df['Concentration'].tolist()
        label = df['H/log(C)_avg']
        return sequences, label

    def get_max_length(self, sequences):
        return max(len(seq) for seq in sequences)

    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            # Add SOS and EOS tokens
            tokens = [tokenizer.bos_token] + list(seq) + [tokenizer.eos_token]
            # Pad with <pad> tokens
            if len(tokens) < max_length:
                tokens += [tokenizer.pad_token] * (max_length - len(tokens))
            padded_sequences.append(tokens)
        return padded_sequences









