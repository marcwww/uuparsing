import torch
from torch import nn
import torch.nn.functional as F
import utils
from macros import *
import crash_on_ipy

class Avg(nn.Module):

    def __init__(self):
        super(Avg, self).__init__()

    def forward(self, vec):
        # vec: (bsz, 1)
        bsz, _ = vec.shape

        return vec.sum(dim=0)/bsz

class StackRNN(nn.Module):

    def __init__(self, voc_size, edim, hdim, stack_len, padding_idx):
        super(StackRNN, self).__init__()

        self.voc_size = voc_size
        self.hdim = hdim
        self.stack_len = stack_len
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                padding_idx=padding_idx)
        self.buf_rnn = nn.GRU(edim, hdim // 2, bidirectional=True)
        self.stack_rnncell = nn.GRUCell(edim, hdim)
        self.stack = nn.Parameter(torch.zeros(stack_len, hdim),
                                  requires_grad=False)
        self.init_stack_hid = nn.Parameter(torch.zeros(hdim),
                                           requires_grad=False)
        self.config2act = nn.Linear(3 * hdim, len(ACTIONS))
        W_up, W_down = torch.Tensor(utils.shift_matrix(stack_len))
        self.W_up = nn.Parameter(W_up,
                                 requires_grad=False)

        self.W_up_n = []
        W_up_n = self.W_up
        for i in range(stack_len):
            self.W_up_n.append(W_up_n)
            W_up_n = self.W_up.matmul(W_up_n)

        # self.W_up_n: (stack_len, stack_len, stack_len)
        self.W_up_n = torch.cat(self.W_up_n).view(-1, stack_len, stack_len)
        self.W_up_n = nn.Parameter(self.W_up_n,
                                   requires_grad=False)
        self.W_down = nn.Parameter(W_down,
                                   requires_grad=False)

        # V_avg: (stack_len, stack_len)
        V_avg = torch.\
            Tensor([utils.avg_vector(i+1, stack_len) for i in range(stack_len)])
        self.V_avg = nn.Parameter(V_avg,
                              requires_grad=False)

        self.top2logProb = nn.Sequential(nn.Linear(hdim, 1),
                                         nn.LogSigmoid(),
                                         Avg())

    def attention(self, input, mems):
        # input: (bsz, hdim)
        # mems: (seq_len, bsz, hdim)

        # mems: (bsz, seq_len, hdim)
        mems = mems.transpose(0,1)
        # input: (bsz, hdim, 1)
        input = input.unsqueeze(-1)
        # a_raw: (bsz, seq_len, 1)
        a_raw = mems.matmul(input)
        # a: (bsz, seq_len, 1)
        a = F.softmax(a_raw, dim=1)
        # outputs: (bsz, seq_len, hdim)
        outputs = mems * a
        # output: (bsz, hdim)
        output = outputs.sum(dim=1)

        return output

    def update_stack(self, stack, action, to_push):
        # stack: (bsz, stack_len, hdim)
        # action: (bsz, nact)
        # to_push: (bsz, hdim)
        # p_xxx: (bsz, 1, 1)

        # p_pop_n: (bsz, stack_len)
        # p_push: (bsz, 1, 1)
        # p_pass: (bsz, 1, 1)
        p_pop_n = action[:, 1:-1]
        p_push = action[:, 0].unsqueeze(-1).unsqueeze(-1)
        p_pass = action[:, -1].unsqueeze(-1).unsqueeze(-1)

        # W_up_n: (stack_len, stack_len, stack_len)
        # res_pop_n: (stack_len, bsz, stack_len, hdim)
        res_pop_n = self.W_up_n.unsqueeze(1).matmul(stack.unsqueeze(0))

        # V_avg: (stack_len, stack_len)
        # to_push_back: (stack_len, bsz, 1, hdim)
        to_push_back = self.V_avg.unsqueeze(1).unsqueeze(1).\
            matmul(stack.unsqueeze(0))

        # res_pop_push: (stack_len, bsz, stack_len, hdim)
        res_pop_push = res_pop_n + to_push_back

        # p_pop_n: (stack_len, bsz, 1, 1)
        p_pop_n = p_pop_n.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)

        # res_xxx: (bsz, stack_len, hdim)
        res_pop_wsum = (p_pop_n * res_pop_push).sum(0)

        res_push = self.W_down.matmul(stack)
        res_push[:, -1, :] = to_push

        res_pass = stack

        # res: (bsz, stack_len, hdim)
        res = res_pop_wsum + \
              p_push * res_push + \
              p_pass * res_pass

        return res

    def forward(self, inputs):

        seq_len, bsz = inputs.shape
        # stack: (bsz, stack_len, hdim)
        stack = self.stack.expand(bsz, self.stack_len, self.hdim)
        # stack_hid: (bsz, hdim)
        stack_hid = self.init_stack_hid.expand(bsz, self.hdim)

        # embs: (seq_len, bsz, edim)
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        mask_embs = mask.unsqueeze(-1).expand_as(embs)
        embs.masked_fill_(mask_embs, 0)

        # outputs: (seq_len, bsz, hdim)
        buf_outs, _ = self.buf_rnn(embs)

        for emb in embs:
            # emb: (bsz, edim)
            # to_push: (bsz, hdim)
            to_push = self.stack_rnncell(emb, stack_hid)
            # stack_top: (bsz, hdim)
            stack_top = stack[:, -1, :]
            buf_atten = self.attention(to_push, buf_outs)
            # config: (bsz, hdim * 3)
            config = torch.cat([to_push, stack_top, buf_atten], dim=1)
            # action: (bsz, 1 + stack_len + 1)
            action = self.config2act(config)

            stack = self.update_stack(stack, action, to_push)

        # negLogProb: (1)
        negLogProb = -1 * self.top2logProb(stack[:, -1, :])

        we_T = self.embedding.weight.transpose(0, 1)
        logits_right = torch.matmul(buf_outs[:-1], we_T)
        logits_left = torch.matmul(buf_outs[1:], we_T)

        return logits_left, logits_right, negLogProb





