import torchtext
from macros import *
import os

def get_iters(ftrain, fvalid, bsz, device, min_freq):

    def tokenizer(txt):
        res = [BOS] + txt.split(' ')
        return res

    SEQ = torchtext.data.Field(sequential=True,
                               tokenize=tokenizer,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)

    train = torchtext.data.TabularDataset(path=os.path.join(DATA, ftrain),
                                          format='tsv',
                                          fields=[('seq', SEQ)])

    SEQ.build_vocab(train, min_freq=min_freq)
    valid = torchtext.data.TabularDataset(path=os.path.join(DATA, fvalid),
                                          format='tsv',
                                          fields=[('seq', SEQ)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         device=device)

    return train_iter, valid_iter, SEQ