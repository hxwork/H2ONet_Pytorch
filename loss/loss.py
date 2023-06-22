import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.nn import functional as F
from scipy.linalg import orthogonal_procrustes
from common.tool import tensor_gpu
from common.utils.transforms import torch_pixel2cam
from common.utils.mano import MANO

mano = MANO()


class CoordLoss(nn.Module):

    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:, :, 2:] * is_3D[:, None, None].float()
            loss = torch.cat((loss[:, :, :2], loss_z), 2)

        return loss


class ParamLoss(nn.Module):

    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss


class NormalVectorLoss(nn.Module):

    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, is_valid=None):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        # valid_mask = valid[:, face[:, 0], :] * valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))  # * valid_mask
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))  # * valid_mask
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))  # * valid_mask
        loss = torch.cat((cos1, cos2, cos3), 1)
        if is_valid is not None:
            loss *= is_valid
        return torch.mean(loss)


class EdgeLengthLoss(nn.Module):

    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, is_valid=None):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :])**2, 2, keepdim=True))
        d2_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :])**2, 2, keepdim=True))
        d3_out = torch.sqrt(torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :])**2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :])**2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :])**2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :])**2, 2, keepdim=True))

        # valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
        # valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
        # valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

        diff1 = torch.abs(d1_out - d1_gt)  # * valid_mask_1
        diff2 = torch.abs(d2_out - d2_gt)  # * valid_mask_2
        diff3 = torch.abs(d3_out - d3_gt)  # * valid_mask_3
        loss = torch.cat((diff1, diff2, diff3), 1)
        if is_valid is not None:
            loss *= is_valid
        return torch.mean(loss)


def compute_loss(cfg, input, output):

    loss = {}
    if cfg.model.name == "i2l":

        coord_loss = CoordLoss()
        param_loss = ParamLoss()
        normal_loss = NormalVectorLoss(output["mesh_face"])
        edge_loss = EdgeLengthLoss(output["mesh_face"])

        if cfg.model.stage == "lixel":
            joint_coord_img = output["joint_coord_img"]
            mesh_coord_img = output["mesh_coord_img"]
            joint_img_from_mesh = output["joint_img_from_mesh"]
            loss["joint_fit"] = coord_loss(joint_coord_img, input["fit_joint_img"], input["fit_joint_trunc"] * input["is_valid_fit"][:, None, None])
            loss["joint_orig"] = coord_loss(joint_coord_img, input["orig_joint_img"], input["orig_joint_trunc"], input["is_3D"])
            loss["mesh_fit"] = coord_loss(mesh_coord_img, input["fit_mesh_img"], input["fit_mesh_trunc"] * input["is_valid_fit"][:, None, None])
            loss["mesh_joint_orig"] = coord_loss(joint_img_from_mesh, input["orig_joint_img"], input["orig_joint_trunc"], input["is_3D"])
            loss["mesh_joint_fit"] = coord_loss(joint_img_from_mesh, input["fit_joint_img"], input["fit_joint_trunc"] * input["is_valid_fit"][:, None, None])
            loss["mesh_normal"] = normal_loss(mesh_coord_img, input["fit_mesh_img"],
                                              input["fit_mesh_trunc"] * input["is_valid_fit"][:, None, None]) * cfg.loss.normal_loss_weight
            loss["mesh_edge"] = edge_loss(mesh_coord_img, input["fit_mesh_img"], input["fit_mesh_trunc"] * input["is_valid_fit"][:, None, None])
        else:
            pose_param = output["pose_param"]
            shape_param = output["shape_param"]
            joint_coord_cam = output["joint_coord_cam"]
            loss["pose_param"] = param_loss(pose_param, input["pose_param"], input["is_valid_fit"][:, None])
            loss["shape_param"] = param_loss(shape_param, input["shape_param"], input["is_valid_fit"][:, None])
            loss["joint_orig_cam"] = coord_loss(joint_coord_cam, input["orig_joint_cam"], input["orig_joint_valid"] * input["is_3D"][:, None, None])
            loss["joint_fit_cam"] = coord_loss(joint_coord_cam, input["fit_joint_cam"], input["is_valid_fit"][:, None, None])

    elif cfg.model.name == "hand_occ_net":

        loss["mano_verts"] = 1e4 * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["mano_joints"] = 1e4 * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        loss["mano_pose"] = 10 * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        loss["mano_shape"] = 0.1 * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        loss["joints_img"] = 100 * F.mse_loss(output["pred_joints_img"], input["joints_img"])

    elif cfg.model.name == "semi_hand":

        loss["mano_verts"] = 1e4 * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["mano_joints"] = 1e4 * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        loss["mano_pose"] = 10 * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        loss["mano_shape"] = 0.1 * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        loss["joints_img"] = 1e2 * F.mse_loss(output["pred_joints_img"], input["joints_img"])
        loss["shape_regul"] = 1e2 * F.mse_loss(output["pred_mano_shape"], torch.zeros_like(output["pred_mano_shape"]))
        loss["pose_regul"] = 1 * F.mse_loss(output["pred_mano_pose"][:, 3:], torch.zeros_like(output["pred_mano_pose"][:, 3:]))

    elif cfg.model.name == "mobrecon":
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        loss["verts_loss"] = F.l1_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])

    elif cfg.model.name in ["mobrecon_gr_v1", "mobrecon_gr_v2", "mobrecon_gr_v3", "mobrecon_r50_v1"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        # loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        # loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        # loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        # loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        # loss["joints2d_w_gr_loss"] = 100 * F.mse_loss(output["pred_joints2d_w_gr"], input["joints_img"])
        # loss["glob_rot_loss"] = 10 * F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
        #                                         torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        # loss["rot_mat_loss"] = 10 * F.mse_loss(output["pred_rot_mat"], output["gt_rot_mat"])
        # loss["joint_img_loss"] = 100 * F.mse_loss(output["pred_joints_img"], input["joints_img"])
        # loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        # loss["edge_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        # loss["normal_w_gr_loss"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        # loss["edge_w_gr_loss"] = edge_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])

        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        loss["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
                                           torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

    elif cfg.model.name in ["mobrecon_gr_v4"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        loss["glob_rot_loss"] = F.mse_loss(output["pred_glob_rot_mat"], output["gt_glob_rot_mat"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

    elif cfg.model.name in ["mobrecon_gr_v5"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        loss["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
                                           torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])

    elif cfg.model.name in ["mobrecon_wo_gr", "mobrecon_r50_wogr"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        # loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        # loss["glob_rot_loss"] = F.l1_loss(output["pred_glob_rot"], output["gt_glob_rot"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

    elif cfg.model.name in ["mobrecon_wogr_occ_v1"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long())

    elif cfg.model.name in ["mobrecon_mf_wogr_occ_v1", "mobrecon_mf_wogr_occ_v2", "mobrecon_mf_wogr_occ_v3"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["gt_joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long())

    elif cfg.model.name in ["mobrecon_mf_wogr_occ_v4"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["gt_joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long())

    elif cfg.model.name in ["mobrecon_mf_gr_occ_v1"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        loss["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
                                           torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["gt_joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long())

    elif cfg.model.name in ["mobrecon_mf_gr_occ_v2"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        loss["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
                                           torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["gt_joints_img"])

        loss["normal_wo_gr_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_wo_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long())

    elif cfg.model.name in ["mobrecon_mf_gr_occ_v3", "mobrecon_mf_gr_occ_v5", "mobrecon_mf_gr_occ_v5_abl_go"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        # loss["verts_wo_gr_loss"] = F.mse_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"]) * 1e4
        # loss["joints_wo_gr_loss"] = F.mse_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"]) * 1e4
        # loss["verts_w_gr_loss"] = F.mse_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"]) * 1e4
        # loss["joints_w_gr_loss"] = F.mse_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"]) * 1e4
        # loss["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
        #                                    torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device)) * 10
        # loss["joint_img_loss"] = F.mse_loss(output["pred_joints_img"], output["gt_joints_img"]) * 100
        # loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long()) * 100

        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        loss["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
                                           torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["gt_joints_img"])

        loss["normal_wo_gr_loss"] = 10 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_wo_loss"] = 10 * edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

        # loss["normal_loss"] = 10 * normal_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        # loss["edge_loss"] = 10 * edge_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long())

    elif cfg.model.name in ["mobrecon_mf_gr_occ_v4"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        # loss["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
        #                                    torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["gt_joints_img"])

        loss["normal_wo_gr_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_wo_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        loss["occ_loss"] = cls_loss(output["pred_occ"], output["gt_occ"].long())

    elif cfg.model.name in ["mobrecon_multi_v0", "mobrecon_multi_v1", "mobrecon_multi_v2"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        loss["verts_loss"] = F.l1_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["gt_joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])

    else:
        raise NotImplementedError

    # Common operation
    loss = {k: loss[k].mean() for k in loss}
    loss["total"] = sum(loss[k] for k in loss)
    return loss


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
    # t1 = mtx1.mean(0)
    # t2 = mtx2.mean(0)
    # mtx1_t = mtx1 - t1
    # mtx2_t = mtx2 - t2

    t1 = torch.mean(mtx1, dim=1, keepdim=True)
    t2 = torch.mean(mtx2, dim=1, keepdim=True)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    # s1 = np.linalg.norm(mtx1_t) + 1e-8
    # mtx1_t /= s1
    # s2 = np.linalg.norm(mtx2_t) + 1e-8
    # mtx2_t /= s2
    s1 = torch.linalg.matrix_norm(mtx1_t, dim=(-2, -1), keepdim=True) + 1e-8
    mtx1_t /= s1
    s2 = torch.linalg.matrix_norm(mtx2_t, dim=(-2, -1), keepdim=True) + 1e-8
    mtx2_t /= s2

    # orth alignment
    # u, w, vt = np.linalg.svd(mtx2_t.T.dot(mtx1_t).T)
    # R = u.dot(vt)
    # s = w.sum()
    u, w, v = torch.svd(torch.matmul(mtx2_t.transpose(1, 2), mtx1_t).transpose(1, 2))
    R = torch.matmul(u, v.transpose(1, 2))
    s = torch.sum(w, dim=1, keepdim=True).unsqueeze(-1)
    # apply trafos to the second matrix
    # mtx2_t = np.dot(mtx2_t, R.T) * s
    # mtx2_t = mtx2_t * s1 + t1
    mtx2_t = torch.matmul(mtx2_t, R.transpose(1, 2)) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def compute_occ_acc(pred_occ, gt_occ):
    pred_occ = torch.argmax(pred_occ, dim=1)  # (6B, 1)
    pred_occ = torch.chunk(pred_occ, 6, dim=0)  # 6 * (B, 1)
    gt_occ = torch.chunk(gt_occ, 6, dim=0)  # 6 * (B, 1)
    occ_acc = {}
    finger_names = ["thumb", "index", "middle", "ring", "little", "palm"]
    for pred, gt, name in zip(pred_occ, gt_occ, finger_names):
        occ_acc[name] = torch.mean((pred == gt).float())
    return occ_acc


def compute_metric(cfg, input, output):
    metric = {}
    if cfg.model.name == "i2l":
        # x,y: resize to input image space and perform bbox to image affine transform
        mesh_out_img = output["mesh_coord_img"]
        mesh_out_img[:, :, 0] = mesh_out_img[:, :, 0] / cfg.data.output_hm_shape[2] * cfg.data.input_img_shape[1]
        mesh_out_img[:, :, 1] = mesh_out_img[:, :, 1] / cfg.data.output_hm_shape[1] * cfg.data.input_img_shape[0]
        mesh_out_img_xy1 = torch.cat((mesh_out_img[:, :, :2], torch.ones_like(mesh_out_img[:, :, :1])), dim=2)
        mesh_out_img[:, :, :2] = torch.matmul(input["bb2img_trans"], mesh_out_img_xy1.transpose(2, 1)).transpose(2, 1)[:, :, :2]

        # z: devoxelize and translate to absolute depth
        root_joint_depth = input["root_joint_depth"]
        mesh_out_img[:, :, 2] = (mesh_out_img[:, :, 2] / cfg.data.output_hm_shape[0] * 2. - 1) * (cfg.data.bbox_3d_size / 2)
        mesh_out_img[:, :, 2] = mesh_out_img[:, :, 2] + root_joint_depth[:, None]

        # camera back-projection
        focal = input["focal"]
        princpt = input["princpt"]
        joint_regressor = input["joint_regressor"]
        mesh_out_cam = torch_pixel2cam(mesh_out_img, focal, princpt)

        if cfg.model.stage == "param":
            mesh_out_cam = output["mesh_coord_cam"]
        joint_out_cam = torch.matmul(joint_regressor, mesh_out_cam)

        vertex_3d = input["vertex_3d"]
        joint_3d = input["joint_3d"]

        vertex_3d_aligned = torch_align_w_scale(vertex_3d, mesh_out_cam)
        joint_3d_aligned = torch_align_w_scale(joint_3d, joint_out_cam)

        mesh_err = torch.mean(torch.norm(vertex_3d_aligned - vertex_3d, p=2, dim=2)) * 1000.
        joint_err = torch.mean(torch.norm(joint_3d_aligned - joint_3d, p=2, dim=2)) * 1000.
        metric["mesh_err"] = mesh_err
        metric["joint_err"] = joint_err
        metric["score"] = mesh_err + joint_err

    elif cfg.model.name == "hand_occ_net":
        metric["mano_verts"] = cfg.loss.lambda_mano_verts * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["mano_joints"] = cfg.loss.lambda_mano_joints * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        metric["mano_pose"] = cfg.loss.lambda_mano_pose * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        metric["mano_shape"] = cfg.loss.lambda_mano_shape * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        metric["joints_img"] = cfg.loss.lambda_joints_img * F.mse_loss(output["pred_joints_img"], input["joints_img"])
        metric["score"] = metric["mano_verts"] + metric["mano_joints"] + metric["mano_pose"] + metric["mano_shape"] + metric["joints_img"]

    elif cfg.model.name == "semi_hand":
        metric["mano_verts"] = 1e4 * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["mano_joints"] = 1e4 * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        metric["mano_pose"] = 10 * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        metric["mano_shape"] = 0.1 * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        metric["joints_img"] = 1e2 * F.mse_loss(output["pred_joints_img"], input["joints_img"])
        metric["shape_regul"] = 1e2 * F.mse_loss(output["pred_mano_shape"], torch.zeros_like(output["pred_mano_shape"]))
        metric["pose_regul"] = 1 * F.mse_loss(output["pred_mano_pose"][:, 3:], torch.zeros_like(output["pred_mano_pose"][:, 3:]))
        metric["score"] = metric["mano_verts"] + metric["mano_joints"] + metric["mano_pose"] + metric["mano_shape"] + metric["joints_img"]

    elif cfg.model.name == "mobrecon":
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        metric["verts_loss"] = F.l1_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        metric["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["edge_loss"] = edge_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["score"] = metric["verts_loss"] + metric["joint_img_loss"] + metric["normal_loss"] + metric["edge_loss"]

    elif cfg.model.name in ["mobrecon_gr_v1", "mobrecon_gr_v2", "mobrecon_gr_v3"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        metric["joints_w_gr"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        # metric["joints2d_w_gr"] = F.l1_loss(output["pred_joints2d_w_gr"], input["joints_img"])
        # metric["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_val_glob_rot_mat"]),
        #                                      torch.eye(3).view(1, 3, 3).repeat(output["gt_val_glob_rot_mat"].size()[0], 1, 1).to(output["gt_val_glob_rot_mat"].device))
        metric["joint_img"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["edge"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["score"] = metric["verts_w_gr"] + metric["verts_wo_gr"]

        # # metrics
        # pred_verts_w_gr = output["pred_verts3d_w_gr"]
        # pred_joints_w_gr = output["pred_joints3d_w_gr"]
        # # root centered
        # pred_verts_w_gr -= pred_joints_w_gr[:, 0, None]
        # pred_joints_w_gr -= pred_joints_w_gr[:, 0, None]
        # # flip back to left hand
        # do_flip = input["do_flip"].to(torch.int32)
        # do_flip = do_flip * (-1) + (1 - do_flip)
        # do_flip = torch.stack([do_flip, torch.ones_like(do_flip), torch.ones_like(do_flip)], dim=1)[:, None, :]
        # pred_verts_w_gr *= do_flip
        # pred_joints_w_gr *= do_flip

        # gt_joints3d_w_gr = input["joints_coord_cam"]
        # gt_verts3d_w_gr = output["gt_verts3d_w_gr"]
        # gt_verts3d_w_gr = gt_verts3d_w_gr - output["gt_joints3d_w_gr"][:, 0, None]
        # gt_joints3d_w_gr = gt_joints3d_w_gr - gt_joints3d_w_gr[:, 0, None]

        # pred_verts_w_gr_aligned = torch_align_w_scale(gt_verts3d_w_gr, pred_verts_w_gr)
        # pred_joints_w_gr_aligned = torch_align_w_scale(gt_joints3d_w_gr, pred_joints_w_gr)

        # # m to mm
        # gt_verts3d_w_gr *= 1000
        # gt_joints3d_w_gr *= 1000

        # pred_verts_w_gr *= 1000
        # pred_joints_w_gr *= 1000
        # pred_verts_w_gr_aligned *= 1000
        # pred_joints_w_gr_aligned *= 1000

        # metric["MPJPE"] = torch.sqrt(torch.sum((pred_joints_w_gr - gt_joints3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)
        # metric["PA-MPJPE"] = torch.sqrt(torch.sum((pred_joints_w_gr_aligned - gt_joints3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)
        # metric["MPVPE"] = torch.sqrt(torch.sum((pred_verts_w_gr - gt_verts3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)
        # metric["PA-MPVPE"] = torch.sqrt(torch.sum((pred_verts_w_gr_aligned - gt_verts3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)

    elif cfg.model.name in ["mobrecon_gr_v4"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        metric["joints_w_gr"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        metric["glob_rot_loss"] = F.mse_loss(output["pred_glob_rot_mat"], output["gt_glob_rot_mat"])
        metric["joint_img"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["edge"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["score"] = metric["verts_w_gr"] + metric["verts_wo_gr"]

    elif cfg.model.name in ["mobrecon_gr_v5"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        metric["joints_w_gr"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        metric["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_glob_rot_mat"]),
                                             torch.eye(3).view(1, 3, 3).repeat(output["gt_glob_rot"].size()[0], 1, 1).to(output["gt_glob_rot_mat"].device))
        metric["joint_img"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        metric["edge"] = edge_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        metric["score"] = metric["verts_w_gr"] + metric["joints_w_gr"]

    elif cfg.model.name in ["mobrecon_wo_gr", "mobrecon_r50_wogr"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        # metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        # metric["glob_rot"] = F.l1_loss(output["pred_glob_rot"], output["gt_glob_rot"])
        metric["joint_img"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["edge"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["score"] = metric["verts_wo_gr"]

    elif cfg.model.name in ["mobrecon_wogr_occ_v1"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["joint_img"] = F.l1_loss(output["pred_joints_img"], input["joints_img"])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["edge"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["occ"] = cls_loss(output["pred_occ"], output["gt_occ"].long())
        occ_acc = compute_occ_acc(output["pred_occ"], output["gt_occ"])
        metric.update(occ_acc)
        metric["score"] = metric["verts_wo_gr"] + metric["occ"]

    elif cfg.model.name in ["mobrecon_mf_wogr_occ_v1", "mobrecon_mf_wogr_occ_v2", "mobrecon_mf_wogr_occ_v3", "mobrecon_mf_wogr_occ_v4"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()

        B = output["pred_verts3d_wo_gr"].size()[0] // 3
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"][:B, ...], output["gt_verts3d_wo_gr"][:B, ...])
        metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"][:B, ...], output["gt_joints3d_wo_gr"][:B, ...])
        metric["joint_img"] = F.l1_loss(output["pred_joints_img"][:B, ...], output["gt_joints_img"][:B, ...])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"][:B, ...], output["gt_verts3d_wo_gr"][:B, ...])
        metric["edge"] = edge_loss(output["pred_verts3d_wo_gr"][:B, ...], output["gt_verts3d_wo_gr"][:B, ...])
        metric["occ"] = cls_loss(output["pred_occ"][:B, ...], output["gt_occ"][:B, ...].long())
        occ_acc = compute_occ_acc(output["pred_occ"][:B, ...], output["gt_occ"][:B, ...])
        metric.update(occ_acc)
        metric["score"] = metric["verts_wo_gr"]

    # elif cfg.model.name in ["mobrecon_mf_gr_occ_v1", "mobrecon_mf_gr_occ_v2", "mobrecon_mf_gr_occ_v3", "mobrecon_mf_gr_occ_v4"]:
    #     normal_loss = NormalVectorLoss(mano.face)
    #     edge_loss = EdgeLengthLoss(mano.face)
    #     cls_loss = torch.nn.CrossEntropyLoss()

    #     B = output["pred_verts3d_wo_gr"].size()[0] // 3
    #     metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
    #     metric["joints_w_gr"] = F.l1_loss(output["pred_joints3d_w_gr"][:B, ...], output["gt_joints3d_w_gr"][:B, ...])
    #     metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"][:B, ...], output["gt_verts3d_wo_gr"][:B, ...])
    #     metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"][:B, ...], output["gt_joints3d_wo_gr"][:B, ...])
    #     metric["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"][:B, ...].permute(0, 2, 1), output["gt_glob_rot_mat"][:B, ...]),
    #                                          torch.eye(3).view(1, 3, 3).repeat(B, 1, 1).to(output["gt_glob_rot_mat"].device))
    #     metric["joint_img"] = F.l1_loss(output["pred_joints_img"][:B, ...], output["gt_joints_img"][:B, ...])
    #     metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
    #     metric["edge"] = edge_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
    #     metric["occ"] = cls_loss(output["pred_occ"][:B, ...], output["gt_occ"][:B, ...].long())
    #     occ_acc = compute_occ_acc(output["pred_occ"][:B, ...], output["gt_occ"][:B, ...])
    #     metric.update(occ_acc)
    #     metric["score"] = metric["verts_w_gr"] + metric["joints_w_gr"]

    elif cfg.model.name in [
            "mobrecon_mf_gr_occ_v1", "mobrecon_mf_gr_occ_v2", "mobrecon_mf_gr_occ_v3", "mobrecon_mf_gr_occ_v4", "mobrecon_mf_gr_occ_v5", "mobrecon_mf_gr_occ_v5_abl_go"
    ]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()

        B = output["pred_verts3d_wo_gr"].size()[0] // 3
        metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
        metric["joints_w_gr"] = F.l1_loss(output["pred_joints3d_w_gr"][:B, ...], output["gt_joints3d_w_gr"][:B, ...])
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"][:B, ...], output["gt_verts3d_wo_gr"][:B, ...])
        metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"][:B, ...], output["gt_joints3d_wo_gr"][:B, ...])
        if "gt_val_glob_rot_mat" not in output:
            metric["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"][:B, ...].permute(0, 2, 1), output["gt_glob_rot_mat"][:B, ...]),
                                                 torch.eye(3).view(1, 3, 3).repeat(B, 1, 1).to(output["gt_glob_rot_mat"].device))
        else:
            metric["glob_rot_loss"] = F.mse_loss(
                torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_val_glob_rot_mat"]),
                torch.eye(3).view(1, 3, 3).repeat(output["gt_val_glob_rot_mat"].size()[0], 1, 1).to(output["gt_val_glob_rot_mat"].device))

        metric["joint_img"] = F.l1_loss(output["pred_joints_img"][:B, ...], output["gt_joints_img"][:B, ...])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
        metric["edge"] = edge_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
        metric["occ"] = cls_loss(output["pred_occ"][:B, ...], output["gt_occ"][:B, ...].long())
        occ_acc = compute_occ_acc(output["pred_occ"][:B, ...], output["gt_occ"][:B, ...])
        metric.update(occ_acc)
        # metric["score"] = metric["verts_w_gr"] + metric["joints_w_gr"]

        # # metrics
        # pred_verts_w_gr = output["pred_verts3d_w_gr"][:B, ...]
        # pred_joints_w_gr = output["pred_joints3d_w_gr"][:B, ...]
        # # root centered
        # pred_verts_w_gr -= pred_joints_w_gr[:, 0, None]
        # pred_joints_w_gr -= pred_joints_w_gr[:, 0, None]
        # # flip back to left hand
        # do_flip = input["do_flip"].to(torch.int32)
        # do_flip = do_flip * (-1) + (1 - do_flip)
        # do_flip = torch.stack([do_flip, torch.ones_like(do_flip), torch.ones_like(do_flip)], dim=1)[:, None, :]
        # pred_verts_w_gr *= do_flip
        # pred_joints_w_gr *= do_flip

        # gt_joints3d_w_gr = input["joints_coord_cam_0"]
        # gt_verts3d_w_gr = output["gt_verts3d_w_gr"][:B, ...]
        # gt_verts3d_w_gr -= output["gt_joints3d_w_gr"][:B, 0, None]
        # gt_joints3d_w_gr -= gt_joints3d_w_gr[:, 0, None]

        # pred_verts_w_gr_aligned = torch_align_w_scale(gt_verts3d_w_gr, pred_verts_w_gr)
        # pred_joints_w_gr_aligned = torch_align_w_scale(gt_joints3d_w_gr, pred_joints_w_gr)

        # # m to mm
        # gt_verts3d_w_gr *= 1000
        # gt_joints3d_w_gr *= 1000

        # pred_verts_w_gr *= 1000
        # pred_joints_w_gr *= 1000
        # pred_verts_w_gr_aligned *= 1000
        # pred_joints_w_gr_aligned *= 1000

        # metric["MPJPE"] = torch.sqrt(torch.sum((pred_joints_w_gr - gt_joints3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)
        # metric["PA-MPJPE"] = torch.sqrt(torch.sum((pred_joints_w_gr_aligned - gt_joints3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)
        # metric["MPVPE"] = torch.sqrt(torch.sum((pred_verts_w_gr - gt_verts3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)
        # metric["PA-MPVPE"] = torch.sqrt(torch.sum((pred_verts_w_gr_aligned - gt_verts3d_w_gr)**2, dim=2)).mean(dim=1).mean(dim=0)

    elif cfg.model.name in ["mobrecon_mf_gr_occ_v5_abl_go"]:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        cls_loss = torch.nn.CrossEntropyLoss()

        B = output["pred_verts3d_wo_gr"].size()[0] // 3
        metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
        metric["joints_w_gr"] = F.l1_loss(output["pred_joints3d_w_gr"][:B, ...], output["gt_joints3d_w_gr"][:B, ...])
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"][:B, ...], output["gt_verts3d_wo_gr"][:B, ...])
        metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"][:B, ...], output["gt_joints3d_wo_gr"][:B, ...])
        if "gt_val_glob_rot_mat" not in output:
            metric["glob_rot_loss"] = F.mse_loss(torch.matmul(output["pred_glob_rot_mat"][:B, ...].permute(0, 2, 1), output["gt_glob_rot_mat"][:B, ...]),
                                                 torch.eye(3).view(1, 3, 3).repeat(B, 1, 1).to(output["gt_glob_rot_mat"].device))
        else:
            metric["glob_rot_loss"] = F.mse_loss(
                torch.matmul(output["pred_glob_rot_mat"].permute(0, 2, 1), output["gt_val_glob_rot_mat"]),
                torch.eye(3).view(1, 3, 3).repeat(output["gt_val_glob_rot_mat"].size()[0], 1, 1).to(output["gt_val_glob_rot_mat"].device))

        metric["joint_img"] = F.l1_loss(output["pred_joints_img"][:B, ...], output["gt_joints_img"][:B, ...])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])
        metric["edge"] = edge_loss(output["pred_verts3d_w_gr"][:B, ...], output["gt_verts3d_w_gr"][:B, ...])

    elif cfg.model.name in ["mobrecon_mf_gr_occ_v3_fps", "mobrecon_gr_v2_fps", "handoccnet_fps", "semihand_fps", "mobrecon_fps"]:
        metric["inference_time"] = output["inference_time"]
        metric["score"] = metric["inference_time"]

    else:
        raise NotImplementedError

    metric = tensor_gpu(metric, check_on=False)

    return metric
