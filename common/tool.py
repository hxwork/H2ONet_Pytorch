import os
import sys
import json
import logging
import torch
import coloredlogs
import numpy as np


class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_previous = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def set(self, val):
        self.val = val
        self.avg = val

    def update(self, val, num):
        self.val_previous = self.val
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


class NpzMaker():

    @classmethod
    def save_npz(cls, files, npz_save_path):
        np.savez(npz_save_path, files=[files, 0])

    @classmethod
    def load_npz(cls, npz_save_path):
        with np.load(npz_save_path, allow_pickle=True) as fin:
            files = fin["files"]
            files = list(files)
            return files[0]


def loss_meter_manager_intial(loss_meter_names):
    loss_meters = []
    for name in loss_meter_names:
        exec("%s = %s" % (name, "AverageMeter()"))
        exec("loss_meters.append(%s)" % name)

    return loss_meters


def tensor_gpu(batch, check_on=True):

    def check_on_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            tensor_g = tensor_
        else:
            tensor_g = tensor_.cuda(non_blocking=True).float()
        return tensor_g

    def check_off_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            return tensor_

        if tensor_.is_cuda:
            tensor_c = tensor_.cpu()
        else:
            tensor_c = tensor_
        tensor_c = tensor_c.detach().numpy()
        return tensor_c

    if torch.cuda.is_available():
        if check_on:
            for k, v in batch.items():
                batch[k] = check_on_gpu(v)
        else:
            for k, v in batch.items():
                batch[k] = check_off_gpu(v)
    else:
        if check_on:
            batch = batch
        else:
            for k, v in batch.items():
                batch[k] = v.detach().numpy()

    return batch


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    coloredlogs.install(level="INFO", logger=logger, fmt="%(asctime)s %(name)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info("Output and logs will be saved to {}".format(log_path))
    return logger


def save_dict_to_json(d, json_path):
    save_dict = {}
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn"t accept np.array, np.float, )
        for k, v in d.items():
            if isinstance(v, AverageMeter):
                save_dict[k] = float(v.avg)
            else:
                save_dict[k] = float(v)
        json.dump(save_dict, f, indent=4)


def make_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
