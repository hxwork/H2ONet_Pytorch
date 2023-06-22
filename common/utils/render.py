import torch
import os
import numpy as np
import json
import cv2

from common.utils.coor_converter import *


def tonemapping(x):
    if isinstance(x, np.ndarray):
        x = (np.log(1 + 5000 * x)) / np.log(1 + 5000)
        x = np.clip(x * 255, 0, 255).astype("uint8")
        return x
    elif isinstance(x, torch.Tensor):
        x = (torch.log(1 + 5000 * x)) / torch.log(torch.tensor(1 + 5000.).to(x.device))
        x = torch.clip(x * 255., 0., 255.)
        return x
    else:
        raise TypeError("Not support type {} for tonemapping".format(type(x)))


class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, img, clip=True, alpha=None, gamma=True):
        if isinstance(img, np.ndarray):
            if gamma:
                power_img = np.power(img, 1 / self.gamma)
            else:
                power_img = img
            non_zero = power_img > 0
            if non_zero.any():
                r_percentile = np.percentile(power_img[non_zero], self.percentile)
            else:
                r_percentile = np.percentile(power_img, self.percentile)
            if alpha is None:
                alpha = self.max_mapping / (r_percentile + 1e-10)
            tonemapped_img = np.multiply(alpha, power_img)

            if clip:
                tonemapped_img = np.clip(tonemapped_img, 0, 1)

            return tonemapped_img.astype('float32'), alpha

        elif isinstance(img, torch.Tensor):
            if gamma:
                img = torch.clamp(img, min=1e-8)  # ensure there is no zero in img
                power_img = torch.pow(img, 1 / self.gamma)
                # img.register_hook(lambda grad: print(grad.min(), grad.max()))
            else:
                power_img = img

            if alpha is None:
                if power_img.dim() == 3:
                    r_percentile = torch.quantile(power_img.contiguous().view(1, -1),
                                                  self.percentile / 100.,
                                                  dim=1,
                                                  interpolation="midpoint")
                    alpha = self.max_mapping / (r_percentile + 1e-10)
                    alpha = alpha[:, None, None]
                elif power_img.dim() == 4:
                    r_percentile = torch.quantile(power_img.contiguous().view(img.size()[0], -1),
                                                  self.percentile / 100.,
                                                  dim=1,
                                                  interpolation="midpoint")
                    alpha = self.max_mapping / (r_percentile + 1e-10)
                    alpha = alpha[:, None, None, None]
                else:
                    raise RuntimeError("Wrong input image dim, should be 3 or 4")
            else:
                if power_img.dim() == 3:
                    alpha = alpha[:, None, None]
                elif power_img.dim() == 4:
                    if alpha.dim() != 4:
                        alpha = alpha[:, None, None, None]
                else:
                    raise RuntimeError("Wrong input image dim, should be 3 or 4")
            tonemapped_img = torch.multiply(alpha, power_img)

            if clip:
                tonemapped_img = torch.clip(tonemapped_img, 0, 1)

            return tonemapped_img.float(), alpha

        else:
            raise TypeError("Not support type {} for tonemapping".format(type(img)))


def inv_tonemapping(x):
    if isinstance(x, np.ndarray):
        low = (np.exp((x - 0.5) / 255. * np.log(1 + 5000)) - 1) / 5000
        up = (np.exp((x + 0.5) / 255. * np.log(1 + 5000)) - 1) / 5000
        low = np.clip(low, 0., 1.)
        up = np.clip(up, 0., 1.)
        return low, up
    elif isinstance(x, torch.Tensor):
        low = (torch.exp((x - 0.5) / 255. * torch.log(torch.tensor(1 + 5000.).to(x.device))) - 1) / 5000
        up = (torch.exp((x + 0.5) / 255. * torch.log(torch.tensor(1 + 5000.).to(x.device))) - 1) / 5000
        low = torch.clip(low, 0., 1.)
        up = torch.clip(up, 0., 1.)
        return low, up
    else:
        raise TypeError("Not support type {} for tonemapping".format(type(x)))


def np_render_sg(param_l, param_s, param_c):
    H = 256
    W = 512
    pano = np.zeros((H, W, 3))
    if (os.path.isfile("common/utils/map_of_u/map_of_u_h{}_w{}.npy".format(H, W))):
        map_of_u = np.load("common/utils/map_of_u/map_of_u_h{}_w{}.npy".format(H, W))
    else:
        print("executed map_of_u calculation")
        map_of_u = np.zeros((H, W, 3))
        for row in range(H):
            for col in range(W):
                theta, phi = row_col2theta_phi(row, col, W, H)
                u = np_theta_phi2xyz(theta, phi)
                map_of_u[row, col, :] = u
        np.save("common/utils/map_of_u/map_of_u_h{}_w{}.npy".format(H, W), map_of_u)
        map_of_u = np.load("common/utils/map_of_u/map_of_u_h{}_w{}.npy".format(H, W))

    l_dot_u = np.dot(map_of_u, param_l)
    expo = (l_dot_u - 1.0) / (param_s / (4 * np.pi))
    single_channel_weight = np.exp(expo)
    repeated_weight = np.repeat(single_channel_weight[:, :, np.newaxis], 3, axis=2)  # (H, W, 3, N)
    single_light_pano = np.multiply(param_c, repeated_weight)
    pano = np.sum(single_light_pano, axis=-1)

    return pano


def torch_single_render_sg(param_l, param_s, param_c):
    H = 256
    W = 512
    pano = torch.zeros((H, W, 3)).cuda()
    map_of_u = torch.zeros((H, W, 3)).cuda()
    if (os.path.isfile("common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))):
        # print("loaded map_of_u")
        map_of_u = torch.load("common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))
    else:
        print("executed map_of_u calculation")
        for row in range(H):
            for col in range(W):
                theta, phi = row_col2theta_phi(row, col, W, H)
                u = torch_theta_phi2xyz(theta, phi)
                map_of_u[row, col, :] = u
        torch.save(map_of_u, "common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))
        # map_of_u = torch.load("common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))

    l_dot_u = torch.matmul(map_of_u.cuda(), param_l)
    expo = (l_dot_u - 1.0) / (param_s / (4 * np.pi))
    single_channel_weight = torch.exp(expo)
    repeated_weight = single_channel_weight[:, :, None].repeat(1, 1, 3, 1)  # (H, W, 3, N)
    single_light_pano = param_c * repeated_weight
    pano = torch.sum(single_light_pano, dim=-1)

    return pano


def torch_batch_render_sg(sg_coeffs, H, W):

    # param_l: (B, 3, N)
    # param_s:  # (B, N)
    # param_c:  # (B, 3, N)
    param_l = sg_coeffs[:, :3, :]
    param_s = sg_coeffs[:, 3, :]
    param_c = sg_coeffs[:, 4:, :]

    B = param_l.size()[0]
    pano = torch.zeros((B, H, W, 3))
    if (os.path.isfile("common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))):
        # print("loaded map_of_u")
        map_of_u = torch.load("common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))
    else:
        print("executed map_of_u calculation")
        map_of_u = torch.zeros((H, W, 3))
        for row in range(H):
            for col in range(W):
                theta, phi = row_col2theta_phi(row, col, W, H)
                u = torch_theta_phi2xyz(theta, phi)
                map_of_u[row, col, :] = u
        torch.save(map_of_u, "common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))
        map_of_u = torch.load("common/utils/map_of_u/map_of_u_h{}_w{}.pth".format(H, W))
    map_of_u = map_of_u.cuda()
    map_of_u = map_of_u[None, ...].repeat(B, 1, 1, 1)  # (B, H, W, 3)

    # # For debug
    # all_l, all_s, all_c = [], [], []
    # for p in param:
    #     all_l.append(torch.tensor(p["l"]))
    #     all_s.append(p["s"])
    #     all_c.append(torch.tensor(p["c"]))
    # param_l = torch.stack(all_l, dim=1)[None, ...].repeat(B, 1, 1)  # (B, 3, N)
    # param_s = torch.tensor(all_s)[None, ...].repeat(B, 1)  # (B, N)
    # param_c = torch.stack(all_c, dim=1)[None, ...].repeat(B, 1, 1)  # (B, 3, N)

    # (B, H, W, 3) -> (B, H*W, 3) * (B, 3, N) = (B, H*W, N) -> (B, H, W, N)
    l_dot_u = torch.bmm(map_of_u.view(B, -1, 3), param_l).view(B, H, W, -1)  # refer to the comment above
    expo = (l_dot_u - 1.0) / (param_s[:, None, None, :] / (4 * np.pi))  # (B, H, W, N)
    single_channel_weight = torch.exp(expo)  # (B, H, W, N)
    repeated_weight = single_channel_weight[:, :, :, None].repeat(1, 1, 1, 3, 1)  # (B, H, W, 3, N)
    single_light_pano = param_c[:, None, None] * repeated_weight  # (B, H, W, 3, N)
    pano = torch.sum(single_light_pano, dim=-1)  # (B, H, W, 3)

    return pano


def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi / 2., H), torch.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)], dim=-1)  # [H, W, 3]
    # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    # lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([
        1,
    ] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + 1e-8)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    # rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = lgtSGMus * torch.exp(4 * np.pi / lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap


# def EMLight_SG2Envmap(dirs, sizes, colors):
#     grid_latitude, grid_longitude = torch.meshgrid([torch.arange(128, dtype=torch.float), torch.arange(2 * 128, dtype=torch.float)])
#     grid_latitude = grid_latitude.add(0.5)
#     grid_longitude = grid_longitude.add(0.5)
#     grid_latitude = grid_latitude.mul(np.pi / 128)
#     grid_longitude = grid_longitude.mul(np.pi / 128)

#     x = torch.sin(grid_latitude) * torch.cos(grid_longitude)
#     y = torch.sin(grid_latitude) * torch.sin(grid_longitude)
#     z = torch.cos(grid_latitude)
#     xyz = torch.stack((x, y, z)).to(dirs.device)
#     print(xyz.size())

#     nbatch = colors.shape[0]
#     lights = torch.zeros((nbatch, 3, 128, 256), dtype=dirs.dtype, device=dirs.device, requires_grad=True)
#     _, tmp = colors.shape
#     nlights = int(tmp / 3)
#     for i in range(nlights):
#         lights = lights + (colors[:, 3 * i + 0:3 * i + 3][:, :, None, None]) * (torch.exp(
#             (torch.matmul(dirs[:, 3 * i + 0:3 * i + 3], xyz.view(3, -1)).view(-1, xyz.shape[1], xyz.shape[2]) - 1) /
#             (sizes[:, i]).view(-1, 1, 1))[:, None, :, :])
#     return lights


def EMLight_SG2Envmap(dirs, sizes, colors):

    B, N = colors.shape[0], colors.shape[1]

    grid_latitude, grid_longitude = torch.meshgrid([torch.arange(128, dtype=torch.float), torch.arange(2 * 128, dtype=torch.float)])
    grid_latitude = grid_latitude.add(0.5)
    grid_longitude = grid_longitude.add(0.5)
    grid_latitude = grid_latitude.mul(np.pi / 128)
    grid_longitude = grid_longitude.mul(np.pi / 128)

    x = torch.sin(grid_latitude) * torch.cos(grid_longitude)
    y = torch.sin(grid_latitude) * torch.sin(grid_longitude)
    z = torch.cos(grid_latitude)
    xyz = torch.stack((x, y, z)).to(dirs.device)  # (3, H, W)
    xyz = xyz.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    xyz = xyz.unsqueeze(0).repeat(B, 1, 1, 1).unsqueeze(-2)  # (B, H, W, 1, 3)

    dirs = dirs.view(B, 1, 1, N, 3).expand(B, 128, 256, N, 3)  # (B, H, W, N, 3)
    sizes = sizes.view(B, 1, 1, N, 1).expand(B, 128, 256, N, 1)  # (B, H, W, N, 1)
    colors = colors.view(B, 1, 1, N, 3).expand(B, 128, 256, N, 3)  # (B, H, W, N, 3)

    rgb = colors * (torch.exp((torch.sum(xyz * dirs, dim=-1, keepdim=True) - 1.) / sizes))  # (B, H, W, N, 3)
    rgb = torch.sum(rgb, dim=-2)  # (B, H, W, 3)
    envmap = rgb.reshape((B, 128, 256, 3)).permute(0, 3, 1, 2)
    return envmap


def batch_SG2Envmap(lgtSGs, H, W, upper_hemi=False):
    B = lgtSGs.size()[0]
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi / 2., H), torch.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)], dim=-1)  # [H, W, 3]
    viewdirs = viewdirs.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 3]

    # lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [B, H, W, 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])  # [B, H, W]
    B, M = lgtSGs.shape[0], lgtSGs.shape[1]  # [B, num_lights]
    lgtSGs = lgtSGs.view(B, 1, 1, M, 7).expand(B, H, W, M, 7)
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + 1e-8)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    lgtSGMus = torch.clip(lgtSGMus, 0., 1.)
    # [..., M, 3]
    # rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = lgtSGMus * torch.exp(4 * np.pi / lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((B, H, W, 3))
    return envmap




if __name__ == "__main__":
    param_lights_info_path = "/research/d4/rshr/xuhao/code/obman_render2.8/datageneration/lighting/00001/param_lights.json"
    with open(param_lights_info_path, "r") as f:
        param_lights_info = json.load(f)

    np_pano = np_render_sg(param_lights_info["param_lights"])
    np_pano = tonemapping(np_pano)
    np_pano = np.clip(np_pano * 255, 0, 255).astype("uint8")
    cv2.imwrite("/research/d4/rshr/xuhao/code/ARHandLighting/common/utils/np_pano.png", np_pano)

    # torch_single_pano = torch_single_render_sg(param_lights_info["param_lights"])
    # torch_single_pano = tonemapping(torch_single_pano)
    # torch_single_pano = torch_single_pano.numpy()
    # torch_single_pano = np.clip(torch_single_pano * 255, 0, 255).astype("uint8")
    # cv2.imwrite("/research/d4/rshr/xuhao/code/ARHandLighting/common/utils/torch_single_pano.png", torch_single_pano)

    torch_batch_pano = torch_batch_render_sg(param_lights_info["param_lights"], 3)
    torch_batch_pano = tonemapping(torch_batch_pano)
    torch_batch_pano = torch_batch_pano.numpy()
    torch_batch_pano = np.clip(torch_batch_pano * 255, 0, 255).astype("uint8")
    cv2.imwrite("/research/d4/rshr/xuhao/code/ARHandLighting/common/utils/torch_batch_pano_0.png", torch_batch_pano[0])
    cv2.imwrite("/research/d4/rshr/xuhao/code/ARHandLighting/common/utils/torch_batch_pano_1.png", torch_batch_pano[1])
