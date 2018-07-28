STACK_LEN = 20
ACTIONS={'PUSH':0, 'PASS':STACK_LEN+1}
for i in range(STACK_LEN):
    ACTIONS['POP%d' % (i+1)] = i+1

DATA = './data'

PAD = '<pad>'
UNK = '<unk>'
EOS = '<eos>'
