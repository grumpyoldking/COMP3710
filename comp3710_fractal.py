# koch_snowflake_torch.py
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def koch_subdivide(p0, p1):
    """
    Subdivide segment p0->p1 into 4 segments (Koch rule) using PyTorch tensors.
    Returns shape (5, 2) tensor of points.
    """
    v = p1 - p0
    a = p0 + v / 3.0
    b = p0 + 2.0 * v / 3.0

    # Rotation matrix for -60 degrees
    angle = torch.tensor(-torch.pi / 3)
    rot = torch.tensor([
        [torch.cos(angle), -torch.sin(angle)],
        [torch.sin(angle),  torch.cos(angle)]
    ])
    tip = a + torch.matmul(rot, v / 3.0)

    return torch.stack([p0, a, tip, b, p1])

def koch_iterate(polyline):
    """
    Apply Koch subdivision to each edge of a closed polyline.
    polyline: (N, 2) tensor, last point == first point.
    """
    new_pts = []
    for i in range(polyline.shape[0] - 1):
        seg_pts = koch_subdivide(polyline[i], polyline[i + 1])
        new_pts.append(seg_pts[:-1])  # drop last point to avoid duplicates
    new_pts.append(polyline[-1].unsqueeze(0))
    return torch.cat(new_pts, dim=0)

def initial_triangle(side=1.0):
    """
    Return a closed equilateral triangle as a (4, 2) tensor.
    Centered for nice framing.
    """
    h = (torch.sqrt(torch.tensor(3.0)) / 2.0) * side
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([side, 0.0])
    p2 = torch.tensor([side / 2.0, h])
    tri = torch.stack([p0, p1, p2, p0])
    center = tri.mean(dim=0)
    return tri - center

def koch_snowflake(order=4, side=1.0):
    pts = initial_triangle(side)
    for _ in range(order):
        pts = koch_iterate(pts)
    return pts

def plot_snowflake(pts, linewidth=1.0, fill=False):
    """
    Plot using LineCollection for speed.
    Converts from torch tensor to numpy for matplotlib.
    """
    pts_np = pts.numpy()
    segments = torch.stack([pts[:-1], pts[1:]], dim=1).numpy()
    lc = LineCollection(segments, linewidths=linewidth)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(lc)

    if fill:
        ax.fill(pts_np[:, 0], pts_np[:, 1], alpha=0.15)

    ax.autoscale()
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.set_title(f'Koch Snowflake (order={order})')
    plt.show()

if __name__ == "__main__":
    order = 6  # try 0..6
    pts = koch_snowflake(order=order, side=1.0)
    plot_snowflake(pts, linewidth=0.8, fill=False)
