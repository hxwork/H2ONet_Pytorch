import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    return np.stack((x, y, z), 1)


def torch_cam2pixel(cam_coord, f, c):
    x = cam_coord[:, :, 0] / cam_coord[:, :, 2] * f[:, 0, None] + c[:, 0, None]
    y = cam_coord[:, :, 1] / cam_coord[:, :, 2] * f[:, 1, None] + c[:, 1, None]
    z = cam_coord[:, :, 2]
    return torch.stack((x, y, z), dim=2)


def torch_pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, :, 0] - c[:, 0][:, None]) / f[:, 0][:, None] * pixel_coord[:, :, 2]
    y = (pixel_coord[:, :, 1] - c[:, 1][:, None]) / f[:, 1][:, None] * pixel_coord[:, :, 2]
    z = pixel_coord[:, :, 2]
    return torch.stack((x, y, z), dim=2)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)).transpose(1, 0)
    return world_coord


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num, ) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1

    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def torch_align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = torch.mean(mtx1, dim=1, keepdim=True)
    t2 = torch.mean(mtx2, dim=1, keepdim=True)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = torch.linalg.matrix_norm(mtx1_t, dim=(-2, -1), keepdim=True) + 1e-8
    mtx1_t /= s1
    s2 = torch.linalg.matrix_norm(mtx2_t, dim=(-2, -1), keepdim=True) + 1e-8
    mtx2_t /= s2

    # orth alignment
    u, w, v = torch.svd(torch.matmul(mtx2_t.transpose(1, 2), mtx1_t).transpose(1, 2))
    R = torch.matmul(u, v.transpose(1, 2))
    s = torch.sum(w, dim=1, keepdim=True).unsqueeze(-1)

    # apply trafos to the second matrix
    mtx2_t = torch.matmul(mtx2_t, R.transpose(1, 2)) * s
    mtx2_t = mtx2_t * s1 + t1

    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t
