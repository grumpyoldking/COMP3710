import torch
import matplotlib.pyplot as plt


def grid(nrows):

    n = torch.arange(nrows, dtype=torch.int32)
    k = torch.arange(nrows, dtype=torch.int32)

    N = n.view(-1,1)
    K = k.view(1,-1)

    #check if grid is in triangle 
    valid = K <= N 

    N_minus_K = torch.where(valid, N-K, torch.zeros_like(K))
    odd = ((k & N_minus_K) == 0) & valid 

    return odd.to(torch.float32)

def plot_grid(img):
    plt.imshow(img, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    nrows = 2048 
    img = grid(nrows)
    plot_grid(img)
    print("Grid generated and displayed.")






