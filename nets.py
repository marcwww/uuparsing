import torch
from torch import nn
import torch.nn.functional as F
import utils
from macros import *
import crash_on_ipy
from torch.nn.utils.rnn import pad_packed_sequence,\
    pack_padded_sequence
from collections import defaultdict

class Avg(nn.Module):

    def __init__(self):
        super(Avg, self).__init__()

    def forward(self, vec):
        # vec: (bsz, 1)
        bsz, _ = vec.shape

        return vec.sum(dim=0)/bsz

class BaseRNN(nn.Module):

    def __init__(self, voc_size, edim, hdim, padding_idx):
        super(BaseRNN, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                padding_idx=padding_idx)

        assert edim * 2 == hdim, 'hdim must be 2 times of edim'
        self.bi_rnn = nn.GRU(edim, hdim // 2, bidirectional=True)
        self.top2logProb = nn.Sequential(nn.Linear(hdim, 1),
                                         nn.LogSigmoid(),
                                         Avg())

    def forward(self, inputs):
        # inputs: (seq_len, bsz)
        seq_len, bsz = inputs.shape
        embs = self.embedding(inputs)
        # mask: (seq_len, bsz)
        mask = inputs.data.eq(self.padding_idx)
        # input_lens: (bsz)
        input_lens = seq_len - mask.sum(dim=0)
        embs_p = pack_padded_sequence(embs, input_lens)

        # outputs: (seq_len, bsz, hdim)
        outputs_p, hidden = self.bi_rnn(embs_p)
        we_T = self.embedding.weight.transpose(0, 1)

        outputs, output_lens = pad_packed_sequence(outputs_p)
        hid_f, hid_b = outputs[-1,:,:self.hdim//2], \
                       outputs[0,:,self.hdim//2:]
        hid = torch.cat([hid_f, hid_b], dim=1)
        negLogProb = -1 * self.top2logProb(hid)

        # logits: (seq_len, bsz, voc_size)
        outputs_f = outputs[:-1, :, :self.edim]
        outputs_b = outputs[1:, :, self.edim:]
        logits_f = torch.matmul(outputs_f, we_T)
        logits_b = torch.matmul(outputs_b, we_T)

        return logits_f, logits_b, negLogProb

class StackRNN(nn.Module):

    def __init__(self, voc_size, edim, hdim, stack_len, nsteps, padding_idx):
        super(StackRNN, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.stack_len = stack_len
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                padding_idx=padding_idx)
        self.nsteps = nsteps

        assert edim == hdim, 'hdim must be equal to edim'

        self.buf_rnn = nn.GRU(edim, hdim)
        self.stack_rnncell = nn.GRUCell(edim, hdim)
        self.stack = nn.Parameter(torch.zeros(stack_len, hdim),
                                  requires_grad=False)
        self.init_stack_hid = nn.Parameter(torch.zeros(hdim),
                                           requires_grad=False)
        self.config2chid = nn.Linear(3 * hdim, hdim)
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
        self.unify = nn.ReLU()

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
        # leave room to the unified hidden
        res_pop_n = self.W_down.matmul(res_pop_n)

        # V_avg: (stack_len, stack_len)
        # to_push_back: (stack_len, bsz, 1, hdim)
        to_push_back = self.V_avg.unsqueeze(1).unsqueeze(1).\
            matmul(stack.unsqueeze(0))
        res_pop_n[:, :, :1, :] = self.unify(to_push_back)

        # p_pop_n: (stack_len, bsz, 1, 1)
        p_pop_n = p_pop_n.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)

        # res_xxx: (bsz, stack_len, hdim)
        res_pop_wsum = (p_pop_n * res_pop_n).sum(0)

        res_push = self.W_down.matmul(stack)
        res_push[:, 0, :] = to_push

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

        # embs: (seq_len, bsz, edim)
        embs = self.embedding(inputs)
        # mask = inputs.data.eq(self.padding_idx)
        # mask_embs = mask.unsqueeze(-1).expand_as(embs)
        # embs.masked_fill_(mask_embs, 0)

        # outputs: (seq_len, bsz, hdim)
        buf_outs, _ = self.buf_rnn(embs)

        to_push = embs[0]
        buf_atten = self.attention(to_push, buf_outs)
        for _ in range(self.nsteps):
            stack_top = stack[:, 0, :]
            # config: (bsz, hdim * 3)
            config = torch.cat([to_push, stack_top, buf_atten], dim=1)
            chid = self.config2chid(config)
            action = self.config2act(config)
            to_push = self.attention(chid, embs)

            stack = self.update_stack(stack, action, to_push)

        # negLogProb: (1)
        stack_top = stack[:, 0, :]
        negLogProb = -1 * self.top2logProb(stack_top)

        we_T = self.embedding.weight.transpose(0, 1)

        # logits: (seq_len - 1, bsz, voc_size)
        logits = torch.matmul(buf_outs, we_T)

        return logits, negLogProb

class PCFGRNN(nn.Module):

    def __init__(self, voc_size, edim, hdim, padding_idx):
        super(PCFGRNN, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                padding_idx=padding_idx)

        self.lex_rnn = nn.GRU(edim, hdim)
        self.syn_rnn = nn.GRU(hdim, hdim)
        self.lex2POS = nn.Linear(hdim, hdim)
        self.top2logProb = nn.Sequential(nn.Linear(hdim, 1),
                                         nn.LogSigmoid(),
                                         Avg())
        self.compos_w = nn.Parameter(torch.Tensor(hdim, hdim))

    def _compos_prob(self, shid1, shid2):
        shid2 = shid2.transpose(0, 1)

        out = shid1.matmul(self.compos_w).matmul(shid2)
        return F.logsigmoid(out)


    def forward(self, inputs):
        inputs = inputs
        embs = self.embedding(inputs)

        table_logProbs = defaultdict(float)
        table_hids = defaultdict(torch.Tensor)

        lex_hids, _ = self.lex_rnn(embs)
        syn_hids = self.lex2POS(lex_hids)

        n = len(syn_hids)

        for i in range(n):
            # log(1) = 0
            table_logProbs[i, i] = 0
            table_hids[i, i] = syn_hids[i]

        for l in range(1, n):
            for i in range(n-l):
                j = i + l
                max_score = -10000
                shid1_opt = None
                shid2_opt = None
                for k in range(i, j):
                    shid1 = table_hids[i, k]
                    shid2 = table_hids[k+1, j]
                    candi_prob = table_logProbs[i, k] + table_logProbs[k+1, j] + self._compos_prob(shid1, shid2)
                    if candi_prob.item() > max_score:
                        max_score = candi_prob
                        shid1_opt = shid1
                        shid2_opt = shid2

                table_logProbs[i, j] = max_score
                syn_bigram = torch.cat([shid1_opt, shid2_opt], dim=0).unsqueeze(1)
                syn_hid = self.syn_rnn(syn_bigram)[-1].squeeze(1)
                table_hids[i, j] = syn_hid


        logProb_whole_syn = table_logProbs[n-1, n-1]
        we_T = self.embedding.weight.transpose(0, 1)
        logProbs = F.log_softmax(torch.matmul(lex_hids, we_T))
        logProbs_trans = torch.max(logProbs, dim=-1)[0]
        logProb_whole_lex = logProbs_trans.sum()

        return logProbs, logProb_whole_syn, logProb_whole_lex
















