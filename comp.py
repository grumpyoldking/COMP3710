import numpy as np
import matplotlib.pyplot as plt

def sin_2d(x,y, mu = 0.0, sigma = 1.0):
  return np.sin(x+y)

def main():
    # Grid settings
    grid_size = 2000       # number of points along each axis
    grid_limit = 3.0      # ± limit on both axes

    x = np.linspace(-grid_limit, grid_limit, grid_size)
    y = np.linspace(-grid_limit, grid_limit, grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute Gaussian values on the grid
    Z = sin_2d(X, Y, mu = 0.0, sigma = 1.0)

    # Plot using a contour map
    plt.figure()
    plt.contour(X, Y, Z)
    plt.title("2D Gaussian Distribution (μ=0, σ=1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")  # Ensure aspect ratio is equal
    plt.show()

if __name__ == "__main__":
    main()
