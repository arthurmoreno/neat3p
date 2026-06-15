"""
Substrate coordinate layouts for HyperNEAT-family nets on benchmark envs.

A HyperNEAT/Adaptive net needs every input/hidden/output neuron placed at a 2-D
coordinate so the CPPN can paint weights as a function of geometry. ``voxel_forage_substrate``
lays out the VoxelForage observation (a flattened (C, D, H, W) egocentric patch plus a few
scalar attributes) onto a 2-D plane that preserves spatial structure: voxels keep their
(x, y) position, with channel and depth (z) applied as small offsets so they don't collide.
"""

import numpy as np


def voxel_forage_substrate(patch_shape=(3, 3, 3, 3), n_scalars=2, n_hidden=9, n_actions=7):
    """Return (input_coords, hidden_coords, output_coords) as lists of [x, y].

    Input order matches the env's flattened observation: patch.reshape(-1) in (C, D, H, W)
    row-major order, followed by the scalar attributes.
    """
    C, D, H, W = patch_shape
    input_coords = []
    for c in range(C):
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    px = (2 * x / (W - 1) - 1) if W > 1 else 0.0
                    py = (2 * y / (H - 1) - 1) if H > 1 else 0.0
                    cx = px * 0.7 + (c - (C - 1) / 2) * 0.08
                    cy = 0.45 + py * 0.35 + (z - (D - 1) / 2) * 0.07
                    input_coords.append([float(cx), float(cy)])
    # scalar attributes sit along the very top edge
    for s in range(n_scalars):
        sx = ((2 * s / (n_scalars - 1) - 1) * 0.3) if n_scalars > 1 else 0.0
        input_coords.append([float(sx), 0.98])

    # hidden nodes on a square grid in the middle band
    side = int(np.ceil(np.sqrt(n_hidden)))
    hidden_coords = []
    for i in range(n_hidden):
        r, cc = divmod(i, side)
        hx = ((2 * cc / (side - 1) - 1) * 0.6) if side > 1 else 0.0
        hy = ((r / (side - 1)) * 0.3 - 0.15) if side > 1 else 0.0
        hidden_coords.append([float(hx), float(hy)])

    # action outputs along the bottom edge
    output_coords = []
    for a in range(n_actions):
        ax = (2 * a / (n_actions - 1) - 1) if n_actions > 1 else 0.0
        output_coords.append([float(ax), -0.9])

    return input_coords, hidden_coords, output_coords
