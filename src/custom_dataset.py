import numpy as np
import torch
from itertools import chain

def batch_maker(data, targets, user_names, utterance_ids, user_tree, user_tag, user_history, user_hismask, batch_size, shuffle=True):
    sequences = list(zip(data, targets, user_names, utterance_ids, user_tree, user_tag, user_history, user_hismask))
    sequences.sort(key=lambda x: len(x[0]), reverse=True)
    # return mini-batched data and targets
    ret = list(chunks(sequences, batch_size))
    if shuffle:
        np.random.shuffle(ret)
    return ret 
def chunks(l, n):
    head = 0
    for i in range(0, len(l)):
        if i == len(l) -1 or len(l[i][0]) != len(l[i+1][0]) or i - head == n - 1:
            src_seqs = np.array(list(list(zip(*l[head:i + 1]))[0]))
            trg_seqs = np.array(list(list(zip(*l[head:i + 1]))[1]))
            usr_names = list(list(zip(*l[head:i + 1]))[2])
            utt_ids = list(list(zip(*l[head:i + 1]))[3])
            usr_tree = list(list(zip(*l[head:i + 1]))[4])
            usr_tag = np.array(list(list(zip(*l[head:i + 1]))[5]))
            usr_his = np.array(list(list(zip(*l[head:i + 1]))[6]))
            usr_hismask = np.array(list(list(zip(*l[head:i + 1]))[7]))
            src_seqs = torch.from_numpy(src_seqs).long()
            trg_seqs = torch.from_numpy(trg_seqs).float()
            usr_tag = torch.from_numpy(usr_tag).float()
            usr_his = torch.from_numpy(usr_his).long()
            usr_hismask = torch.from_numpy(usr_hismask).float()
            yield (src_seqs, trg_seqs, usr_names, utt_ids, usr_tree, usr_tag, usr_his, usr_hismask)
            head = i + 1

def flattern_result(list_of_lists):
    return list(chain.from_iterable(list_of_lists))
