import os
import numpy as np
import cv2
import torch
import copy
from pycocotools.coco import COCO
import json
from common.utils.preprocessing import load_img, process_bbox, get_bbox
from common.utils.vis import vis_mesh, render_mesh_seg
from common.utils.mano import MANO
from common.utils.transforms import cam2pixel
from occ_label_preparation.seg import *

mano = MANO()


def load_dexycb_data(data_split):
    root_dir = "data/DEX_YCB"
    annot_path = os.path.join(root_dir, "annotations")
    db = COCO(os.path.join(annot_path, "DEX_YCB_s0_{}_data.json".format(data_split)))

    datalist = []
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann["image_id"]
        img = db.loadImgs(image_id)[0]
        img_path = os.path.join(root_dir, img["file_name"])
        img_shape = (img["height"], img["width"])
        if data_split == "train":
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


def get_fingers_occ_label(gt_verts_pixel, render_seg, img_shape):
    # occ
    w_out_of_range_mask = gt_verts_pixel[:, 0] >= img_shape[1]
    h_out_of_range_mask = gt_verts_pixel[:, 1] >= img_shape[0]
    out_of_range_mask = np.logical_or(w_out_of_range_mask, h_out_of_range_mask)
    gt_verts_pixel[out_of_range_mask] = 0
    gt_verts_pixel = gt_verts_pixel.astype(np.int32)

    thumb_vertex_pixel = gt_verts_pixel[thumb_verts_idx]
    thumb_occ_mask = np.all(render_seg[thumb_vertex_pixel[:, 1], thumb_vertex_pixel[:, 0]] == thumb_color, axis=1)

    index_vertex_pixel = gt_verts_pixel[index_verts_idx]
    index_occ_mask = np.all(render_seg[index_vertex_pixel[:, 1], index_vertex_pixel[:, 0]] == index_color, axis=1)

    middle_vertex_pixel = gt_verts_pixel[middle_verts_idx]
    middle_occ_mask = np.all(render_seg[middle_vertex_pixel[:, 1], middle_vertex_pixel[:, 0]] == middle_color, axis=1)

    ring_vertex_pixel = gt_verts_pixel[ring_verts_idx]
    ring_occ_mask = np.all(render_seg[ring_vertex_pixel[:, 1], ring_vertex_pixel[:, 0]] == ring_color, axis=1)

    little_vertex_pixel = gt_verts_pixel[little_verts_idx]
    little_occ_mask = np.all(render_seg[little_vertex_pixel[:, 1], little_vertex_pixel[:, 0]] == little_color, axis=1)

    palm_vertex_pixel = gt_verts_pixel[palm_verts_idx]
    palm_occ_mask = np.all(render_seg[palm_vertex_pixel[:, 1], palm_vertex_pixel[:, 0]] == palm_color, axis=1)

    finger_names = ["thumb", "index", "middle", "ring", "little", "palm"]
    occ_mask_list = [thumb_occ_mask, index_occ_mask, middle_occ_mask, ring_occ_mask, little_occ_mask, palm_occ_mask]
    occ_label_list = []
    occ_ratio_list = []
    occ_count_list = []
    for finger_name, occ_mask in zip(finger_names, occ_mask_list):
        occ_ratio_list.append(round(occ_mask.sum() / occ_mask.shape[0], 2))
        occ_label_list.append(int(occ_mask.sum() < 30))
        occ_count_list.append(occ_mask.sum())
    return occ_label_list, occ_ratio_list, occ_count_list


def generate_dexycb_occ_gt():
    data_split = "test"
    datalist = load_dexycb_data(data_split)

    for idx, data in enumerate(datalist):
        img_path, img_shape, bbox = data["img_path"], data["img_shape"], data["bbox"]
        scene_name = ".".join(i for i in [img_path.split("/")[-4], img_path.split("/")[-3], img_path.split("/")[-2]])
        file_name = img_path.split("/")[-1]
        frame_idx = int(file_name.split(".")[0].split("_")[-1])

        # flip hand
        hand_type = data["hand_type"]
        do_flip = (hand_type == "left")

        # occ label output filepath
        occ_info_ouput_path = img_path.replace("color", "occ").replace("jpg", "json")
        if os.path.exists(occ_info_ouput_path):
            print("exist: {}, skip it!".format(occ_info_ouput_path))
            continue

        # seg
        label_path = img_path.replace("color", "labels").replace("jpg", "npz")
        label = np.load(label_path)
        seg = label["seg"] > 200
        seg = np.tile(seg[:, :, None], (1, 1, 3))

        # gt joints and verts
        gt_mano_pose = torch.from_numpy(data["mano_pose"][None, ...])
        if do_flip:
            gt_mano_pose = gt_mano_pose.reshape(-1, 3)
            gt_mano_pose[:, 1:] *= -1
            gt_mano_pose = gt_mano_pose.reshape(1, -1)
        gt_mano_shape = torch.from_numpy(data["mano_shape"][None, ...])

        gt_verts, gt_joints = mano.layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)
        gt_verts = gt_verts.squeeze().cpu().numpy()
        gt_joints = gt_joints.squeeze().cpu().numpy()
        gt_verts = gt_verts / 1000.
        gt_joints = gt_joints / 1000.

        # occlusion mask
        root_joint_idx = 0
        root_joint_cam = copy.deepcopy(data["joints_coord_cam"][root_joint_idx])
        gt_verts = gt_verts - gt_joints[0]
        if do_flip:
            gt_verts[:, 0] *= -1
        gt_verts = gt_verts + root_joint_cam
        gt_verts_pixel = cam2pixel(gt_verts, data["cam_param"]["focal"], data["cam_param"]["princpt"])

        # img
        img = load_img(img_path)  # current
        ori_img = copy.deepcopy(img)

        # render predicted mesh for input hand image
        render_seg_ori, render_mask_ori = render_mesh_seg(ori_img, gt_verts, mano.face, data["cam_param"])
        render_seg = seg * render_seg_ori + (1 - seg) * ori_img
        occ_label, occ_ratio, occ_count_list = get_fingers_occ_label(gt_verts_pixel, render_seg, img.shape)

        # global occ
        global_occ_ratio = np.sum(seg[:, :, 0]) / np.sum(render_mask_ori[:, :, 0])
        occ_label.append(int(global_occ_ratio < 0.40))
        occ_ratio.append(round(global_occ_ratio, 2))
        occ_count_list.append(0)

        # print info
        occ_label_str = "Th: {}, Ind: {}, Mid: {}, Ring: {}, Lit: {}, Palm: {}, Glob: {}".format(*occ_label)
        occ_ratio_str = "Th: {}, Ind: {}, Mid: {}, Ring: {}, Lit: {}, Palm: {}, Glob: {}".format(*occ_ratio)
        occ_count_str = "Th: {}, Ind: {}, Mid: {}, Ring: {}, Lit: {}, Palm: {}, Glob: {}".format(*occ_count_list)

        # dump json
        occ_info_dict = {
            "thumb": occ_label[0],
            "index": occ_label[1],
            "middle": occ_label[2],
            "ring": occ_label[3],
            "little": occ_label[4],
            "palm": occ_label[5],
            "global": occ_label[6],
        }
        with open(occ_info_ouput_path, "w") as f:
            json.dump(occ_info_dict, f)

        # save intermediate results
        if idx % 50 == 0:
            mesh_ori = vis_mesh(ori_img, gt_verts_pixel)
            render_seg = cv2.putText(render_seg, "{}".format(frame_idx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            render_seg = cv2.putText(render_seg, "{}".format(occ_label_str), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            render_seg = cv2.putText(render_seg, "{}".format(occ_count_str), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            render_seg = cv2.putText(render_seg, "{}".format(occ_ratio_str), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            img = cv2.putText(img, "{}".format(frame_idx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cat_img = np.concatenate([img, render_seg_ori, seg * 255, render_seg, mesh_ori], axis=1)
            print("writing {}/{}".format(scene_name, frame_idx))
            os.makedirs("demo/dexycb.occ_gt_images/{}".format(scene_name), exist_ok=True)
            cv2.imwrite("demo/dexycb.occ_gt_images/{}/{}".format(scene_name, file_name), cat_img[:, :, ::-1])


if __name__ == "__main__":
    generate_dexycb_occ_gt()
