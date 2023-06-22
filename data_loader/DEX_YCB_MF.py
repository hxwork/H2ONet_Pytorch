import logging
import os
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from common.utils.preprocessing import load_img, process_bbox, get_bbox, augmentation_multi_frames
from common.utils.transforms import align_w_scale
from common.utils.mano import MANO

logger = logging.getLogger(__name__)


class DEX_YCB_MF(torch.utils.data.Dataset):

    def __init__(self, cfg, transform, data_split):
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split if data_split == "train" else "test"
        self.root_dir = "data/DEX_YCB"
        self.annot_path = os.path.join(self.root_dir, "annotations")
        self.root_joint_idx = 0
        self.mano = MANO()

        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(os.path.join(self.annot_path, "DEX_YCB_s0_{}_data.json".format(self.data_split)))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]
            img_path = os.path.join(self.root_dir, img["file_name"])
            img_shape = (img["height"], img["width"])
            if self.data_split == "train":
                joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
                joints_coord_img = np.array(ann["joints_img"], dtype=np.float32)
                hand_type = ann["hand_type"]

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)

                if bbox is None:
                    continue

                mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
                mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

                data = {
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "joints_coord_cam": joints_coord_cam,
                    "joints_coord_img": joints_coord_img,
                    "bbox": bbox,
                    "cam_param": cam_param,
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                    "hand_type": hand_type
                }
            else:
                joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)
                root_joint_cam = copy.deepcopy(joints_coord_cam[0])
                joints_coord_img = np.array(ann["joints_img"], dtype=np.float32)
                hand_type = ann["hand_type"]

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)

                if bbox is None:
                    bbox = np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}

                mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
                mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

                data = {
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "joints_coord_cam": joints_coord_cam,
                    "joints_coord_img": joints_coord_img,
                    "root_joint_cam": root_joint_cam,
                    "bbox": bbox,
                    "cam_param": cam_param,
                    "image_id": image_id,
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                    "hand_type": hand_type
                }

            datalist.append(data)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data_0 = copy.deepcopy(self.datalist[idx])
        img_path_0, bbox_0, cam_param_0 = data_0["img_path"], data_0["bbox"], data_0["cam_param"]
        hand_type = data_0["hand_type"]
        do_flip = (hand_type == "left")
        filename = img_path_0.split("/")[-1]

        # multi-frame index selection
        frame_idx_0 = int(filename.split(".")[0].split("_")[-1])
        frame_range_1, frame_range_2 = self.cfg.data.frame_range[0], self.cfg.data.frame_range[1]
        if frame_idx_0 < frame_range_2:
            frame_idx_1 = frame_idx_0 // 2
            frame_idx_2 = frame_idx_0
        else:
            if self.data_split == "train":
                frame_idx_1 = frame_range_1 + int((np.random.random() - 0.5) * 2 * 2)  # [range_1-2, range_1+2]
                frame_idx_2 = frame_range_2 - int(np.random.random() * 2)  # [range_2-2]
            else:
                frame_idx_1 = frame_range_1
                frame_idx_2 = frame_range_2
        data_1 = self.datalist[idx - frame_idx_1]
        data_2 = self.datalist[idx - frame_idx_2]
        img_path_1, bbox_1 = data_1['img_path'], data_1['bbox']
        img_path_2, bbox_2 = data_2['img_path'], data_2['bbox']

        # img
        img_0 = load_img(img_path_0)
        img_1 = load_img(img_path_1)  # previous short term
        img_2 = load_img(img_path_2)  # previous long term

        img_list = [img_0, img_1, img_2]
        bbox_list = [bbox_0, bbox_1, bbox_2]
        data_list = [data_0, data_1, data_2]
        img_shape_list = [data_0["img_shape"], data_1["img_shape"], data_2["img_shape"]]
        img_list, img2bb_trans_list, bb2img_trans_list, rot_list, scale = augmentation_multi_frames(img_list=img_list,
                                                                                                    bbox_list=bbox_list,
                                                                                                    data_split=self.data_split,
                                                                                                    input_img_shape=self.cfg.data.input_img_shape,
                                                                                                    scale_factor=0.25,
                                                                                                    rot_factor=30,
                                                                                                    rot_prob=self.cfg.data.rot_prob,
                                                                                                    same_rot=True,
                                                                                                    color_factor=0.2,
                                                                                                    do_flip=do_flip)
        img_list = [self.transform(img.astype(np.float32)) / 255. for img in img_list]
        input = {}
        for img_idx, (img, data, img2bb_trans, rot, img_shape) in enumerate(zip(img_list, data_list, img2bb_trans_list, rot_list, img_shape_list)):
            if self.data_split == "train":
                # 2D joint coordinate
                joints_img = data["joints_coord_img"]
                if do_flip:
                    joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
                joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
                joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

                # normalize to [0,1]
                joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
                joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

                # 3D joint camera coordinate
                joints_coord_cam = data["joints_coord_cam"]
                root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
                joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative
                if do_flip:
                    joints_coord_cam[:, 0] *= -1

                # 3D data rotation augmentation
                rot_aug_mat = np.array(
                    [[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32)
                joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

                # mano parameter
                mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]

                # 3D data rotation augmentation
                mano_pose = mano_pose.reshape(-1, 3)
                if do_flip:
                    mano_pose[:, 1:] *= -1
                root_pose = mano_pose[self.root_joint_idx, :]
                root_pose, _ = cv2.Rodrigues(root_pose)
                root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
                mano_pose[self.root_joint_idx] = root_pose.reshape(3)
                mano_pose = mano_pose.reshape(-1)

                # occ
                occ_info_path = data["img_path"].replace("color", "occ_v2").replace("jpg", "json")
                with open(occ_info_path, "r") as f:
                    occ_info = json.load(f)
                occ_gt = np.array([occ_info["thumb"], occ_info["index"], occ_info["middle"], occ_info["ring"], occ_info["little"], occ_info["global"]])

                input["img_{}".format(img_idx)] = img
                input["joints_img_{}".format(img_idx)] = joints_img
                input["img2bb_trans_{}".format(img_idx)] = img2bb_trans,
                input["joints_coord_cam_{}".format(img_idx)] = joints_coord_cam
                input["mano_pose_{}".format(img_idx)] = mano_pose
                input["mano_shape_{}".format(img_idx)] = mano_shape
                input["root_joint_cam_{}".format(img_idx)] = root_joint_cam
                input["gt_occ_{}".format(img_idx)] = occ_gt

            else:
                root_joint_cam = data["root_joint_cam"]
                joints_coord_cam = data["joints_coord_cam"]

                # mano parameter
                mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]

                # 2D joint coordinate
                joints_img = data["joints_coord_img"]

                # Only for joint img metric
                if do_flip:
                    joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
                joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
                joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

                # normalize to [0,1]
                joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
                joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

                # occ
                occ_info_path = data["img_path"].replace("color", "occ_v2").replace("jpg", "json")
                with open(occ_info_path, "r") as f:
                    occ_info = json.load(f)
                occ_gt = np.array([occ_info["thumb"], occ_info["index"], occ_info["middle"], occ_info["ring"], occ_info["little"], occ_info["global"]])

                # Only for rotation metric
                val_mano_pose = copy.deepcopy(mano_pose).reshape(-1, 3)
                if do_flip:
                    val_mano_pose[:, 1:] *= -1
                val_mano_pose = val_mano_pose.reshape(-1)

                input["img_{}".format(img_idx)] = img
                input["joints_img_{}".format(img_idx)] = joints_img
                input["img2bb_trans_{}".format(img_idx)] = img2bb_trans
                input["joints_coord_cam_{}".format(img_idx)] = joints_coord_cam
                input["mano_pose_{}".format(img_idx)] = mano_pose
                input["val_mano_pose_{}".format(img_idx)] = val_mano_pose
                input["mano_shape_{}".format(img_idx)] = mano_shape
                input["root_joint_cam_{}".format(img_idx)] = root_joint_cam
                input["gt_occ_{}".format(img_idx)] = occ_gt
                input["do_flip"] = np.array(do_flip).astype(np.int)

        return input

    def evaluate(self, batch_output, cur_sample_idx):
        batch_size = len(batch_output)
        eval_result = [[], [], [], []]  # [mpjpe_list, pa-mpjpe_list]
        for n in range(batch_size):
            data = copy.deepcopy(self.datalist[cur_sample_idx + n])
            output = batch_output[n]
            pred_verts_w_gr = output["pred_verts3d_w_gr"]
            pred_joints_w_gr = output["pred_joints3d_w_gr"]

            # root centered
            pred_verts_w_gr -= pred_joints_w_gr[0, None]
            pred_joints_w_gr -= pred_joints_w_gr[0, None]

            joints_coord_cam = data["joints_coord_cam"]
            if data["hand_type"] == "left":
                pred_joints_w_gr[:, 0] *= -1

            # GT and rigid align
            gt_verts_w_gr = output["gt_verts3d_w_gr"] - output["gt_joints3d_w_gr"][0, None]
            gt_joints_w_gr = joints_coord_cam - joints_coord_cam[0, None]

            # align predictions
            pred_verts_w_gr_aligned = align_w_scale(gt_verts_w_gr, pred_verts_w_gr)
            pred_joints_w_gr_aligned = align_w_scale(gt_joints_w_gr, pred_joints_w_gr)

            # m to mm
            gt_verts_w_gr *= 1000
            gt_joints_w_gr *= 1000

            pred_verts_w_gr *= 1000
            pred_joints_w_gr *= 1000
            pred_verts_w_gr_aligned *= 1000
            pred_joints_w_gr_aligned *= 1000

            eval_result[0].append(np.sqrt(np.sum((pred_joints_w_gr - gt_joints_w_gr)**2, 1)).mean())
            eval_result[1].append(np.sqrt(np.sum((pred_joints_w_gr_aligned - gt_joints_w_gr)**2, 1)).mean())
            eval_result[2].append(np.sqrt(np.sum((pred_verts_w_gr - gt_verts_w_gr)**2, 1)).mean())
            eval_result[3].append(np.sqrt(np.sum((pred_verts_w_gr_aligned - gt_verts_w_gr)**2, 1)).mean())

        MPJPE = np.mean(eval_result[0])
        PAMPJPE = np.mean(eval_result[1])
        MPVPE = np.mean(eval_result[2])
        PAMPVPE = np.mean(eval_result[3])
        score = PAMPJPE + PAMPVPE
        metric = {
            "MPJPE": MPJPE,
            "PA-MPJPE": PAMPJPE,
            "MPVPE": MPVPE,
            "PA-MPVPE": PAMPVPE,
            "score": score,
        }
        return metric

    def print_eval_result(self, test_epoch):
        print("MPJPE : %.2f mm" % np.mean(self.eval_result[0]))
        print("PA MPJPE : %.2f mm" % np.mean(self.eval_result[1]))
        print("MPVPE : %.2f mm" % np.mean(self.eval_result[2]))
        print("PA MPVPE : %.2f mm" % np.mean(self.eval_result[3]))
