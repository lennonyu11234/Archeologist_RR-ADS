import numpy as np
from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("pep",
                                          ignore_mismatched_sizes=True)


def unique(arr):
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def idx_to_seqs(seqs):
    sequences = []
    for i in seqs:
        seq = [tokenizer.decode(g,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True) for g in i]
        letters_only = [char for char in seq if char.isalpha()]
        letters_string = ''.join(letters_only)
        sequences.append(letters_string)

    return sequences