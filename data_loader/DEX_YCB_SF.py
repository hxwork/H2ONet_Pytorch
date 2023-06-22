import os
import numpy as np
import torch
import cv2
import copy
from pycocotools.coco import COCO
from common.utils.preprocessing import load_img, process_bbox, augmentation, get_bbox
from common.utils.transforms import align_w_scale
from common.utils.mano import MANO

mano = MANO()


class DEX_YCB_SF(torch.utils.data.Dataset):

    def __init__(self, cfg, transform, data_split):
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split if data_split == "train" else "test"
        self.root_dir = "data/DEX_YCB"
        self.annot_path = os.path.join(self.root_dir, "annotations")
        self.root_joint_idx = 0
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

                bbox_center_x = (bbox[0] + bbox[2]) / 2.0 - (img["width"] / 2.0)
                bbox_center_y = (bbox[1] + bbox[3]) / 2.0 - (img["height"] / 2.0)
                bbox_size_x = bbox[2] / img["width"]
                bbox_size_y = bbox[3] / img["height"]
                bbox_pos = np.array([bbox_center_x, bbox_center_y, bbox_size_x, bbox_size_y])

                mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
                mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

                data = {
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "joints_coord_cam": joints_coord_cam,
                    "joints_coord_img": joints_coord_img,
                    "bbox": bbox,
                    "bbox_pos": bbox_pos,
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
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, cam_param = data["img_path"], data["img_shape"], data["bbox"], data["cam_param"]
        hand_type = data["hand_type"]
        do_flip = (hand_type == "left")

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, self.cfg.data.input_img_shape, do_flip=do_flip)
        img = self.transform(img.astype(np.float32)) / 255.

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
            bbox_pos = data["bbox_pos"]

            input = {
                "img": img,
                "bbox_pos": bbox_pos,
                "joints_img": joints_img,
                "img2bb_trans": img2bb_trans,
                "rot_aug_mat": rot_aug_mat,
                "joints_coord_cam": joints_coord_cam,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
                "root_joint_cam": root_joint_cam,
                "cam_focal": cam_param["focal"],
                "cam_princpt": cam_param["princpt"],
            }

        else:
            root_joint_cam = data["root_joint_cam"]
            joints_coord_cam = data["joints_coord_cam"]

            # mano parameter
            mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]

            # Only for rotation metric
            val_mano_pose = copy.deepcopy(mano_pose).reshape(-1, 3)
            if do_flip:
                val_mano_pose[:, 1:] *= -1
            val_mano_pose = val_mano_pose.reshape(-1)

            # 2D joint coordinate
            joints_img = data["joints_coord_img"]
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize to [0,1]
            joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
            joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

            input = {
                "img": img,
                "joints_img": joints_img,
                "img2bb_trans": img2bb_trans,
                "joints_coord_cam": joints_coord_cam,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
                "val_mano_pose": val_mano_pose,
                "root_joint_cam": root_joint_cam,
                "cam_focal": cam_param["focal"],
                "cam_princpt": cam_param["princpt"],
                "do_flip": np.array(do_flip).astype(np.int),
            }

        return input

    def evaluate(self, batch_output, cur_sample_idx):
        batch_size = len(batch_output)
        eval_result = [[], [], [], []]  #[mpjpe_list, pa-mpjpe_list]
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
            "PAMPJPE": PAMPJPE,
            "MPVPE": MPVPE,
            "PAMPVPE": PAMPVPE,
            "score": score,
        }
        return metric

    def print_eval_result(self, test_epoch):
        print("MPJPE : %.2f mm" % np.mean(self.eval_result[0]))
        print("PA MPJPE : %.2f mm" % np.mean(self.eval_result[1]))
        print("MPVPE : %.2f mm" % np.mean(self.eval_result[2]))
        print("PA MPVPE : %.2f mm" % np.mean(self.eval_result[3]))
