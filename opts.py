import argparse

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=100)
    group.add_argument('-hdim', type=int, default=100)
    group.add_argument('-stack_len', type=int, default=15)
    group.add_argument('-nsteps', type=int, default=100)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-ftrain', type=str, default='ptb.train.txt')
    group.add_argument('-fvalid', type=str, default='ptb.valid.txt')
    # group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-bsz', type=int, default=1)
    group.add_argument('-min_freq', type=int, default=1)
    group.add_argument('-nepoch', type=int, default=10)
    group.add_argument('-save_per', type=int, default=5)
    group.add_argument('-name', type=str, default='uuparsing')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-lr', type=float, default=1e-3)
    group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-wdecay', type=float, default=0)

