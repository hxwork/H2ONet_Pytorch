import logging
import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import random
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from pytorch3d import transforms as p3dt
from collections import OrderedDict

from common.utils.preprocessing import load_img, process_bbox, augmentation, augmentation_multi_frames, get_bbox
from common.utils.transforms import cam2pixel, transform_joint_to_other_db
from common.utils.mano import MANO

logger = logging.getLogger(__name__)


class HO3DGR_MF_RAM(Dataset):

    def __init__(self, cfg, transform, data_split):
        super(HO3DGR_MF_RAM, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split
        self.data_split_for_load = "train" if data_split == "train" or data_split == "val" else "evaluation"
        self.server_root_dir = "/research/d4/rshr/xuhao/data/HO3D_v2"
        self.local_root_dir = "/data1/dataset/HO3D_v2"
        if os.path.exists(self.server_root_dir):
            self.root_dir = self.server_root_dir
        else:
            self.root_dir = self.local_root_dir
        self.annot_path = osp.join(self.root_dir, "annotations")
        self.root_joint_idx = 0
        self.mano = MANO()

        self.datalist = self.load_data()
        if self.data_split == "test":
            self.eval_result = [[], []]  #[pred_joints_list, pred_verts_list]
        self.joints_name = ("Wrist", "Index_1", "Index_2", "Index_3", "Middle_1", "Middle_2", "Middle_3", "Pinky_1", "Pinky_2", "Pinky_3", "Ring_1", "Ring_2", "Ring_3",
                            "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4", "Index_4", "Middle_4", "Ring_4", "Pinly_4")

    def split_train_val_data(self, sort_datlist, split_manner="uniform"):
        if split_manner == "consequtive":
            scene_img_num = OrderedDict()
            for idx, data in enumerate(sort_datlist):
                img_path = data["img_path"]
                img_idx = int(img_path.split("/")[-1].split(".")[0])
                scene_name = img_path.split("/")[-3]
                if scene_name not in scene_img_num:
                    scene_img_num[scene_name] = {"start": idx, "len": 1}
                else:
                    scene_img_num[scene_name]["len"] += 1
            val_idx = []
            for name, info in scene_img_num.items():
                start, length = info["start"], info["len"]
                cur_val_length = int(0.1 * length)
                cur_val_start = np.random.randint(start, start + length - cur_val_length)
                cur_val_idx = np.arange(cur_val_start, cur_val_start + cur_val_length).tolist()
                val_idx.extend(cur_val_idx)
        elif split_manner == "uniform":
            val_idx = np.arange(0, len(sort_datlist), 20).tolist()
        else:
            raise NotImplementedError
        return val_idx

    def load_data(self):
        db = COCO(osp.join(self.annot_path, "HO3D_{}_data.json".format(self.data_split_for_load)))

        datalist = []
        cnt = 0
        for aid in tqdm(db.anns.keys()):
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

                bbox_center_x = (bbox[0] + bbox[2]) / 2.0 - (img["width"] / 2.0)
                bbox_center_y = (bbox[1] + bbox[3]) / 2.0 - (img["height"] / 2.0)
                bbox_size = np.max(np.array([bbox[2] / img["width"], bbox[3] / img["height"]]))
                bbox_pos = np.array([bbox_center_x, bbox_center_y, bbox_size])

                mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
                mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

                img = load_img(img_path)

                data = {
                    "img": img,
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "joints_coord_cam": joints_coord_cam,
                    "joints_coord_img": joints_coord_img,
                    "bbox": bbox,
                    "bbox_pos": bbox_pos,
                    "cam_param": cam_param,
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape
                }
            else:
                root_joint_cam = np.array(ann["root_joint_cam"], dtype=np.float32)
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
                bbox = np.array(ann["bbox"], dtype=np.float32)
                bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.5)

                bbox_center_x = (bbox[0] + bbox[2]) / 2.0 - (img["width"] / 2.0)
                bbox_center_y = (bbox[1] + bbox[3]) / 2.0 - (img["height"] / 2.0)
                bbox_size = np.max(np.array([bbox[2] / img["width"], bbox[3] / img["height"]]))
                bbox_pos = np.array([bbox_center_x, bbox_center_y, bbox_size])

                img = load_img(img_path)

                data = {
                    "img": img,
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "root_joint_cam": root_joint_cam,
                    "bbox": bbox,
                    "bbox_pos": bbox_pos,
                    "cam_param": cam_param
                }

            datalist.append(data)

        if self.data_split == "train" or self.data_split == "val":
            with open(osp.join(self.root_dir, "{}.txt".format(self.data_split_for_load)), "r") as f:
                lines = f.readlines()
            sort_idx = np.argsort(lines)
            sort_datlist = []
            for cnt, i in enumerate(sort_idx):
                sort_datlist.append(datalist[i])

            val_idx = self.split_train_val_data(sort_datlist, split_manner=self.cfg.data.split_manner)
            if self.data_split == "train":
                sort_datlist = [sort_datlist[i] for i in range(len(sort_datlist)) if i not in val_idx]
            else:
                sort_datlist = [sort_datlist[i] for i in val_idx]

            return sort_datlist
        else:
            return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data_0 = copy.deepcopy(self.datalist[idx])
        img_path_0, bbox_0, cam_param = data_0["img_path"], data_0["bbox"], data_0["cam_param"]
        filename = img_path_0.split("/")[-1]

        # multi-frame index selection
        frame_idx_0 = int(filename.split(".")[0])
        frame_range_1, frame_range_2 = self.cfg.data.frame_range[0], self.cfg.data.frame_range[1]
        if frame_idx_0 < frame_range_2:
            frame_idx_1 = frame_idx_0 // 2
            frame_idx_2 = frame_idx_0
        else:
            if self.data_split == "train":
                frame_idx_1 = frame_range_1 + int((np.random.random() - 0.5) * 10)  # [range_1-5, range_1+5]
                frame_idx_2 = frame_range_2 - int(np.random.random() * 5)  # [range_2-5]
            else:
                frame_idx_1 = frame_range_1
                frame_idx_2 = frame_range_2
        data_1 = self.datalist[idx - frame_idx_1]
        data_2 = self.datalist[idx - frame_idx_2]
        img_path_1, bbox_1 = data_1['img_path'], data_1['bbox']
        img_path_2, bbox_2 = data_2['img_path'], data_2['bbox']

        # img
        img_0 = data_0["img"]
        img_1 = data_1["img"]  # previous short term
        img_2 = data_2["img"]  # previous long term
        img_list = [img_0, img_1, img_2]
        bbox_list = [bbox_0, bbox_1, bbox_2]
        data_list = [data_0, data_1, data_2]
        img_list, img2bb_trans_list, bb2img_trans_list, rot_list, scale = augmentation_multi_frames(img_list=img_list,
                                                                                                    bbox_list=bbox_list,
                                                                                                    data_split=self.data_split,
                                                                                                    input_img_shape=self.cfg.data.input_img_shape,
                                                                                                    scale_factor=0.25,
                                                                                                    rot_factor=30,
                                                                                                    rot_prob=self.cfg.data.rot_prob,
                                                                                                    same_rot=True,
                                                                                                    color_factor=0.2,
                                                                                                    do_flip=False)
        img_list = [self.transform(img.astype(np.float32)) / 255. for img in img_list]

        input = {}
        for img_idx, (img, data, img2bb_trans, rot) in enumerate(zip(img_list, data_list, img2bb_trans_list, rot_list)):
            if self.data_split == "train" or self.data_split == "val":
                # 2D joint coordinate
                joints_img = data["joints_coord_img"]
                joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
                joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                # normalize to [0,1]
                joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
                joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

                # 3D joint camera coordinate
                joints_coord_cam = data["joints_coord_cam"]
                # root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
                # joints_coord_cam -= root_joint_cam[None, :]  # root-relative
                # 3D data rotation augmentation
                rot_aug_mat = np.array(
                    [[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32)
                joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)
                root_joint_cam = joints_coord_cam[self.root_joint_idx, :]
                joints_coord_cam = joints_coord_cam - root_joint_cam[None, :]  # root-relative

                # mano parameter
                mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]
                # 3D data rotation augmentation
                mano_pose = mano_pose.reshape(-1, 3)
                root_pose = mano_pose[self.root_joint_idx, :]
                root_pose, _ = cv2.Rodrigues(root_pose)
                root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
                mano_pose[self.root_joint_idx] = root_pose.reshape(3)
                mano_pose = mano_pose.reshape(-1)

                # occ
                occ_info_path = data["img_path"].replace("rgb", "occ_v2").replace("png", "json")
                with open(occ_info_path, "r") as f:
                    occ_info = json.load(f)
                occ_gt = np.array([occ_info["thumb"], occ_info["index"], occ_info["middle"], occ_info["ring"], occ_info["little"], occ_info["global"]])

                input["img_{}".format(img_idx)] = img
                input["joints_img_{}".format(img_idx)] = joints_img
                input["joints_coord_cam_{}".format(img_idx)] = joints_coord_cam
                input["mano_pose_{}".format(img_idx)] = mano_pose
                input["mano_shape_{}".format(img_idx)] = mano_shape
                input["root_joint_cam_{}".format(img_idx)] = root_joint_cam
                input["gt_occ_{}".format(img_idx)] = occ_gt

            else:
                input["img_{}".format(img_idx)] = img
                input["root_joint_cam_{}".format(img_idx)] = data["root_joint_cam"]

        return input

    def evaluate(self, batch_output, cur_sample_idx):
        annots = self.datalist
        batch_size = len(batch_output)
        for n in range(batch_size):
            data = annots[cur_sample_idx + n]
            output = batch_output[n]
            # pred_verts = output["pred_verts3d_wo_gr"]
            # pred_joints = output["pred_joints3d_wo_gr"]
            # pred_glob_rot = output["pred_glob_rot"]

            # # root align
            # gt_root_joint_cam = data["root_joint_cam"]
            # pred_verts = pred_verts - pred_joints[self.root_joint_idx]
            # pred_joints = pred_joints - pred_joints[self.root_joint_idx]
            # pred_glob_rot_mat = p3dt.rotation_6d_to_matrix(torch.from_numpy(pred_glob_rot)).numpy()
            # pred_verts = np.matmul(pred_verts, pred_glob_rot_mat.T)
            # pred_joints = np.matmul(pred_joints, pred_glob_rot_mat.T)
            # pred_verts = pred_verts + gt_root_joint_cam
            # pred_joints = pred_joints + gt_root_joint_cam

            pred_verts = output["pred_verts3d_w_gr"]
            pred_joints = output["pred_joints3d_w_gr"]
            gt_root_joint_cam = data["root_joint_cam"]
            pred_verts = pred_verts - pred_joints[self.root_joint_idx] + gt_root_joint_cam
            pred_joints = pred_joints - pred_joints[self.root_joint_idx] + gt_root_joint_cam

            # convert to openGL coordinate system.
            pred_verts *= np.array([1, -1, -1])
            pred_joints *= np.array([1, -1, -1])

            # convert joint ordering from MANO to HO3D.
            pred_joints = transform_joint_to_other_db(pred_joints, self.mano.joints_name, self.joints_name)

            self.eval_result[0].append(pred_joints.tolist())
            self.eval_result[1].append(pred_verts.tolist())

    def print_eval_result(self, epoch):
        output_json_file = osp.join(self.cfg.base.model_dir, "pred.{}.e{}.json".format(self.cfg.base.exp_name, epoch))
        output_zip_file = osp.join(self.cfg.base.model_dir, "pred.{}.e{}.zip".format(self.cfg.base.exp_name, epoch))

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
