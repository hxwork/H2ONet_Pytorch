import logging
import torch
import torchvision
import numpy as np

logger = logging.getLogger(__name__)


class Template:

    def __init__(self, a=0.01, b=0.05):
        self.a = a
        self.b = a

    def __call__(self, sample):

        return sample


class ToTensor():

    def __init__(self):
        pass

    def __call__(self, input):
        # input = {
        #     "hand_rgb": hand_rgb.transpose(2, 0, 1),
        #     # "hand_seg": hand_seg.transpose(1, 2, 0),
        #     # "gt_param_l": param_l,
        #     # "gt_param_s": param_s,
        #     # "gt_param_c": param_c,
        #     "gt_param_a": param_a,
        #     "gt_bg_sg": bg_sg.transpose(2, 0, 1),
        #     "pano_lights": pano_lights.transpose(2, 0, 1),
        # }
        # input["img"] = torch.from_numpy(input["img"]) / 255.
        # input["hand_seg"] = torch.from_numpy(input["hand_seg"])
        return input


def fetch_transforms(cfg):

    if cfg.data.transforms_type == "i2lmeshnet":
        train_transforms = [torchvision.transforms.ToTensor()]
        test_transforms = [torchvision.transforms.ToTensor()]

    elif cfg.data.transforms_type == "hand_occ_net":
        train_transforms = [torchvision.transforms.ToTensor()]
        test_transforms = [torchvision.transforms.ToTensor()]

    else:
        raise NotImplementedError

    logger.info("Train transforms: {}".format(", ".join([type(t).__name__ for t in train_transforms])))
    logger.info("Val/Test transforms: {}".format(", ".join([type(t).__name__ for t in test_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    test_transforms = torchvision.transforms.Compose(test_transforms)
    return train_transforms, test_transforms
