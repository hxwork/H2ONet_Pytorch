import torch
import numpy as np


def row_col2theta_phi(row, col, width, height):
    theta = ((row + 0.5) / height) * np.pi
    phi = (0.5 - (col + 0.5) / width) * 2.0 * np.pi
    phi = (col - (width - 1) / 2) / width * 2 * np.pi
    return theta, phi


def theta_phi2row_col_array(theta, phi, width, height):
    row = (theta / np.pi) * height - 0.5
    col = (0.5 - phi / (2.0 * np.pi)) * width - 0.5
    row = row.astype(int)
    col = col.astype(int)
    row = np.clip(row, 0, height - 1)  # make sure the lights do not sink to bottom
    col = col % width  # handle the negative cols
    return row, col


def np_theta_phi2xyz(theta, phi):
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.cos(phi)
    return np.array((x, y, z))


def np_xyz2theta_phi(x, y, z):
    theta = np.arccos(y)
    phi = np.arctan2(x, z)  # quadrant awareness
    return theta, phi


def torch_theta_phi2xyz(theta, phi):
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.cos(phi)
    return torch.tensor((x, y, z)).cuda()


def torch_xyz2theta_phi(x, y, z):
    theta = torch.arccos(y)
    phi = torch.arctan2(x, z)  # quadrant awareness
    return theta, phi
