import torch
import matplotlib.pyplot as plt


def grid(nrows):
    # Generate a grid of points in a triangular lattice
    n = torch.arange(nrows, dtype=torch.int32)
    k = torch.arange(nrows, dtype=torch.int32)
    # Create a meshgrid of n and k 
    # resahpe n into column vector and k into row vector
    N = n.view(-1,1)
    K = k.view(1,-1)

    #check if grid is in pascal triangle 
    triangle = K <= N 

    #check if k is odd using lucas's theorem
    N_minus_K = torch.where(triangle, N-K, torch.zeros_like(K))
    odd = ((k & N_minus_K) == 0) & triangle

    return odd

# Plot the grid using matplotlib
def plot_grid(img):
    plt.imshow(img, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()

nrows = 1024
img = grid(nrows)
plot_grid(img)







