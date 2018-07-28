import nets
import preproc
from macros import *
import argparse
import opts
import torch
import utils
from torch import optim
from torch import nn

def train(model, iters, opt, criterion_lm, optim):
    train_iter = iters['train']
    valid_iter = iters['valid']

    # valid(model, valid_iter, opt)


    for epoch in range(opt.nepoch):
        for i, sample in enumerate(train_iter):
            model.train()
            inputs = sample.seq

            model.zero_grad()

            logits_left, logits_right, negLogProb = \
                model(inputs)

            loss_lm_left = criterion_lm(logits_left.view(-1, model.voc_size),
                                        inputs[1:].view(-1))
            loss_lm_right = criterion_lm(logits_right.view(-1, model.voc_size),
                                         inputs[:-1].view(-1))
            loss = (1-opt.lm_coef) * (loss_lm_left + loss_lm_right)/2 + \
                   opt.lm_coef * negLogProb

            loss.backward()
            optim.step()

            loss = {'lm_l': loss_lm_left.item(),
                    'lm_r': loss_lm_right.item(),
                    'negLogProb': negLogProb.item()}

            utils.progress_bar(i/len(train_iter), loss, epoch)

        # valid(model, valid_iter, opt)

        if (epoch + 1) % opt.save_per == 0:
            basename = "{}-epoch-{}".format(opt.name, epoch)
            model_fname = basename + ".model"
            torch.save(model.state_dict(), model_fname)

if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    for i in range(opt.stack_len):
        ACTIONS['POP%d' % (i + 1)] = i + 1
    ACTIONS['PASS'] = opt.stack_len + 1

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)
    train_iter, valid_iter, SEQ = preproc.get_iters(ftrain=opt.ftrain,
                                    fvalid=opt.fvalid,
                                    bsz=opt.bsz,
                                    min_freq=opt.min_freq,
                                    device=opt.gpu)

    model = nets.StackRNN(voc_size=len(SEQ.vocab.itos),
                          edim=opt.edim,
                          hdim=opt.hdim,
                          stack_len=opt.stack_len,
                          padding_idx=SEQ.vocab.stoi[PAD]).to(device)

    utils.init_model(model)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                         model.parameters()),
                           lr=opt.lr)

    criterion_lm = nn.CrossEntropyLoss(ignore_index=SEQ.vocab.stoi[PAD])

    train(model, {'train': train_iter,
                  'valid': valid_iter},
          opt,
          criterion_lm,
          optimizer)