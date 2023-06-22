import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from pytorch3d import transforms as p3dt

from common.utils.mano import MANO

from model.modules.utils.read import spiral_tramsform
from model.modules.conv.spiralconv import SpiralConv
from model.modules.models.densestack import H2ONet_Backnone
from model.modules.models.modules import H2ONet_Decoder

logger = logging.getLogger(__name__)
mano = MANO()


# Init model weights
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


class H2ONet(nn.Module):

    def __init__(self, cfg):
        super(H2ONet, self).__init__()
        self.cfg = cfg
        self.backbone = H2ONet_Backnone(cfg=self.cfg, latent_size=256, kpts_num=21)
        template_fp = "model/h2onet/template/template.ply"
        transform_fp = "model/h2onet/template/transform.pkl"
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors=[2, 2, 2, 2], seq_length=[9, 9, 9, 9], dilation=[1, 1, 1, 1])
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())
        self.decoder3d = H2ONet_Decoder(cfg=self.cfg,
                                        latent_size=256,
                                        out_channels=[32, 64, 128, 256],
                                        spiral_indices=spiral_indices,
                                        up_transform=up_transform,
                                        uv_channel=21,
                                        meshconv=SpiralConv)
        self.mano_pose_size = 16 * 3
        self.mano_layer = mano.layer
        self.mano_joint_reg = torch.from_numpy(mano.joint_regressor)
        self.mano_joint_reg = torch.nn.Parameter(self.mano_joint_reg)

    def forward(self, input):
        x = input["img"]
        B = x.size(0)
        latent, j_latent, rot_latent, pred2d_pt = self.backbone(x)
        pred_verts_wo_gr, pred_glob_rot = self.decoder3d(pred2d_pt, latent, j_latent, rot_latent)
        pred_joints_wo_gr = torch.matmul(self.mano_joint_reg.to(pred_verts_wo_gr.device), pred_verts_wo_gr)

        pred_glob_rot_mat = p3dt.rotation_6d_to_matrix(pred_glob_rot)

        pred_root_joint_wo_gr = pred_joints_wo_gr[:, 0, None, ...]
        pred_verts_w_gr = torch.matmul(pred_verts_wo_gr.detach() - pred_root_joint_wo_gr.detach(), pred_glob_rot_mat.permute(0, 2, 1))
        pred_verts_w_gr = pred_verts_w_gr + pred_root_joint_wo_gr
        pred_joints_w_gr = torch.matmul(pred_joints_wo_gr.detach() - pred_root_joint_wo_gr.detach(), pred_glob_rot_mat.permute(0, 2, 1)) + pred_root_joint_wo_gr

        if "mano_pose" in input:
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_verts_w_gr, gt_joints_w_gr = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)
            gt_glob_rot = gt_mano_pose[:, :3].clone()
            gt_glob_rot_mat = p3dt.axis_angle_to_matrix(gt_glob_rot)
            gt_mano_pose[:, :3] = 0
            gt_verts_wo_gr, gt_joints_wo_gr = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts_w_gr /= 1000
            gt_joints_w_gr /= 1000
            gt_verts_wo_gr /= 1000
            gt_joints_wo_gr /= 1000

        else:
            gt_mano_params = None

        output = {}
        output["pred_verts3d_wo_gr"] = pred_verts_wo_gr
        output["pred_joints3d_wo_gr"] = pred_joints_wo_gr
        output["pred_verts3d_w_gr"] = pred_verts_w_gr
        output["pred_joints3d_w_gr"] = pred_joints_w_gr
        output["pred_joints_img"] = pred2d_pt
        output["pred_glob_rot"] = pred_glob_rot
        output["pred_glob_rot_mat"] = pred_glob_rot_mat
        if gt_mano_params is not None:
            output["gt_glob_rot"] = gt_glob_rot
            output["gt_glob_rot_mat"] = gt_glob_rot_mat
            output["gt_verts3d_w_gr"] = gt_verts_w_gr
            output["gt_joints3d_w_gr"] = gt_joints_w_gr
            output["gt_verts3d_wo_gr"] = gt_verts_wo_gr
            output["gt_joints3d_wo_gr"] = gt_joints_wo_gr

            if "val_mano_pose" in input:
                # for eval
                val_gt_verts, val_gt_joints = self.mano_layer(th_pose_coeffs=input["val_mano_pose"], th_betas=input["mano_shape"])
                output["gt_verts3d_w_gr"], output["gt_joints3d_w_gr"] = val_gt_verts / 1000, val_gt_joints / 1000

        return output


def fetch_model(cfg):
    if cfg.model.name == "h2onet":
        model = H2ONet(cfg)

    else:
        raise NotImplementedError

    if cfg.base.cuda:
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            torch.cuda.set_device(0)
        model = model.cuda()
        model = DP(model)

    return model
