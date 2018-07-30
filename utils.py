import numpy as np
from torch.nn.init import xavier_uniform_

def shift_matrix(n):
    W_up = np.eye(n)
    for i in range(n-1):
        W_up[i,:] = W_up[i+1,:]
    W_up[n-1,:] *= 0
    W_down = np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:] = W_down[i-1,:]
    W_down[0,:] *= 0
    return W_up,W_down

def avg_vector(i, n):
    V = np.zeros(n)
    V[:i+1] = 1/(i+1)
    return V

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)

def progress_bar(percent, last_loss, epoch):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f}/epoch {:d} (Loss: {:.4f} {:.4f})".format(
        "=" * fill,
        " " * (40 - fill),
        percent,
        epoch,
        last_loss['lm'],
        last_loss['negLogProb']), end='')

if __name__ == '__main__':
    up, down = shift_matrix(3)
    x = np.array([[0,1,2]]).transpose()
    print(x)
    print(up.dot(x))
    print(down)