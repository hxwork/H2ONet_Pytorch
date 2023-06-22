import logging
import os
import os.path as osp
from re import L
import numpy as np
import torch
import cv2
import json
import copy
import random
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from common.utils.preprocessing import load_img, process_bbox, augmentation_multi_frames, get_bbox
from common.utils.transforms import cam2pixel, transform_joint_to_other_db
from common.utils.mano import MANO

logger = logging.getLogger(__name__)


class HO3DM(Dataset):

    def __init__(self, cfg, transform, data_split):
        super(HO3DM, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split
        self.data_split_for_load = "train" if data_split == "train" or data_split == "val" else "evaluation"
        self.root_dir = "/research/d4/rshr/xuhao/data/HO3D_v2"
        self.annot_path = osp.join(self.root_dir, "annotations")
        self.root_joint_idx = 0
        self.mano = MANO()

        self.datalist, self.sample_idices = self.load_data()
        if self.data_split == "test":
            self.eval_result = [[], []]  #[pred_joints_list, pred_verts_list]
        self.joints_name = ("Wrist", "Index_1", "Index_2", "Index_3", "Middle_1", "Middle_2", "Middle_3", "Pinky_1", "Pinky_2", "Pinky_3",
                            "Ring_1", "Ring_2", "Ring_3", "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4", "Index_4", "Middle_4", "Ring_4",
                            "Pinly_4")

    def load_data(self):
        db = COCO(osp.join(self.annot_path, "HO3D_{}_data.json".format(self.data_split_for_load)))

        datalist = []
        cnt = 0
        for aid in db.anns.keys():
            cnt += 1
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, self.data_split_for_load, img["file_name"])
            img_shape = (img["height"], img["width"])
            if self.data_split == "train" or self.data_split == "val":
                joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
                joints_coord_img = cam2pixel(joints_coord_cam, cam_param["focal"], cam_param["princpt"])
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
                    "mano_shape": mano_shape
                }
            else:
                root_joint_cam = np.array(ann["root_joint_cam"], dtype=np.float32)
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
                bbox = np.array(ann["bbox"], dtype=np.float32)
                bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.5)

                data = {
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "root_joint_cam": root_joint_cam,
                    "bbox": bbox,
                    "cam_param": cam_param
                }

            datalist.append(data)

        if self.data_split == "train" or self.data_split == "val":
            sample_idices = []
            with open(osp.join(self.root_dir, "{}.txt".format(self.data_split_for_load)), "r") as f:
                lines = f.readlines()
            sort_idx = np.argsort(lines)
            sort_datlist = []
            for cnt, i in enumerate(sort_idx):
                if self.data_split == "train" and cnt % 20 != 0:
                    sample_idices.append(cnt)
                if self.data_split == "val" and cnt % 20 == 0:
                    sample_idices.append(cnt)
                sort_datlist.append(datalist[i])

            return sort_datlist, sample_idices
        else:
            sample_idices = [i for i in range(len(datalist))]
            return datalist, sample_idices

    def __len__(self):
        return len(self.sample_idices)

    def __getitem__(self, idx):
        real_idx = self.sample_idices[idx]
        data0 = copy.deepcopy(self.datalist[real_idx])
        img_path0, img_shape, bbox0 = data0["img_path"], data0["img_shape"], data0["bbox"]
        file_name = img_path0.split("/")[-1]
        # multi-frame index selection
        frame_idx0 = int(file_name.split(".")[0])
        frame_idx0 = int(file_name.split(".")[0])
        frame1_range, frame2_range = 25, 50
        if frame_idx0 < frame2_range:
            frame_idx1 = frame_idx0 // 2
            frame_idx2 = frame_idx0
        else:
            frame_idx1 = frame1_range
            frame_idx2 = frame2_range
        data1 = self.datalist[real_idx - frame_idx1]
        data2 = self.datalist[real_idx - frame_idx2]
        img_path1, bbox1 = data1['img_path'], data1['bbox']
        img_path2, bbox2 = data2['img_path'], data2['bbox']
        # img
        img0 = load_img(img_path0)  # current
        img1 = load_img(img_path1)  # previous short term
        img2 = load_img(img_path2)  # previous long term
        img_list = [img0, img1, img2]
        bbox_list = [bbox0, bbox1, bbox2]
        data_list = [data0, data1, data2]
        img_list, img2bb_trans_list, bb2img_trans_list, rot, scale = augmentation_multi_frames(
            img_list, bbox_list, self.data_split, input_img_shape=self.cfg.data.input_img_shape, do_flip=False)
        img_list = [self.transform(img.astype(np.float32)) / 255. for img in img_list]

        input = {}
        for img_idx, (img, data, img2bb_trans) in enumerate(zip(img_list, data_list, img2bb_trans_list)):
            if self.data_split == "train" or self.data_split == "val":
                # 2D joint coordinate
                joints_img = data['joints_coord_img']
                joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
                joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                # normalize to [0,1]
                joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
                joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

                # 3D joint camera coordinate
                joints_coord_cam = data['joints_coord_cam']
                root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
                joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative

                # 3D data rotation augmentation
                rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                        [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]],
                                       dtype=np.float32)
                joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

                # mano parameter
                mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]
                # 3D data rotation augmentation
                mano_pose = mano_pose.reshape(-1, 3)
                root_pose = mano_pose[self.root_joint_idx, :]
                root_pose, _ = cv2.Rodrigues(root_pose)
                root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
                mano_pose[self.root_joint_idx] = root_pose.reshape(3)
                mano_pose = mano_pose.reshape(-1)

                input["img{}".format(img_idx)] = img
                input["joints_img{}".format(img_idx)] = joints_img
                input["joints_coord_cam{}".format(img_idx)] = joints_coord_cam
                input["mano_pose{}".format(img_idx)] = mano_pose
                input["mano_shape{}".format(img_idx)] = mano_shape
                input["root_joint_cam{}".format(img_idx)] = root_joint_cam

            else:
                input["img{}".format(img_idx)] = img
                input["root_joint_cam{}".format(img_idx)] = data["root_joint_cam"]

        return input

    def evaluate(self, batch_output, cur_sample_idx):
        annots = self.datalist
        batch_size = len(batch_output)
        for n in range(batch_size):
            data = annots[cur_sample_idx + n]
            output = batch_output[n]
            verts_out = output["pred_verts3d_cam"]
            joints_out = output["pred_joints3d_cam"]

            # root align
            gt_root_joint_cam = data["root_joint_cam"]
            verts_out = verts_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
            joints_out = joints_out - joints_out[self.root_joint_idx] + gt_root_joint_cam

            # convert to openGL coordinate system.
            verts_out *= np.array([1, -1, -1])
            joints_out *= np.array([1, -1, -1])

            # convert joint ordering from MANO to HO3D.
            joints_out = transform_joint_to_other_db(joints_out, self.mano.joints_name, self.joints_name)

            self.eval_result[0].append(joints_out.tolist())
            self.eval_result[1].append(verts_out.tolist())
        return None

    def print_eval_result(self, epoch):
        output_json_file = osp.join(self.cfg.base.model_dir, "pred{}.json".format(epoch))
        output_zip_file = osp.join(self.cfg.base.model_dir, "pred{}.zip".format(epoch))

        # with open(osp.join(self.root_dir, "{}.txt".format(self.data_split_for_load)), "r") as f:
        #     lines = f.readlines()
        # sort_idx = np.argsort(lines)
        # undo_sort_idx = np.argsort(sort_idx)
        # self.undo_sort_eval_result = [[], []]
        # for cnt, i in enumerate(undo_sort_idx):
        #     self.undo_sort_eval_result[0].append(self.eval_result[0][i])
        #     self.undo_sort_eval_result[1].append(self.eval_result[1][i])

        with open(output_json_file, "w") as f:
            json.dump(self.eval_result, f)
        print("Dumped %d joints and %d verts predictions to %s" % (len(self.eval_result[0]), len(self.eval_result[1]), output_json_file))

        cmd = "zip -j {} {}".format(output_zip_file, output_json_file)
        print(cmd)
        os.system(cmd)
