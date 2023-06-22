import random
import logging
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_loader.transforms import fetch_transforms
from data_loader.DEX_YCB_SF import DEX_YCB_SF
from data_loader.DEX_YCB_MF import DEX_YCB_MF

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2**32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def fetch_dataloader(cfg):
    logger.info("Dataset: {}".format(cfg.data.name))
    # Train and test transforms
    train_transforms, test_transforms = fetch_transforms(cfg)
    # Train dataset
    train_ds = eval(cfg.data.name)(cfg, train_transforms, "train")
    # Val dataset
    if "val" in cfg.data.eval_type:
        val_ds = eval(cfg.data.name)(cfg, test_transforms, "val")
    # Test dataset
    if "test" in cfg.data.eval_type:
        test_ds = eval(cfg.data.name)(cfg, test_transforms, "test")

    # Determine if my pc
    if os.path.exists("/data1/code"):
        is_my_pc = True
    else:
        is_my_pc = False

    # Data loader
    if is_my_pc:
        cfg.data.prefetch_factor = 4
        cfg.train.num_workers = 20
        cfg.test.num_workers = 8
    else:
        cfg.data.prefetch_factor = 1

    if cfg.train.num_workers > 1:
        train_dl = DataLoader(train_ds,
                              batch_size=cfg.train.batch_size,
                              num_workers=cfg.train.num_workers,
                              pin_memory=cfg.base.cuda,
                              shuffle=True,
                              prefetch_factor=cfg.data.prefetch_factor,
                              drop_last=True,
                              worker_init_fn=worker_init_fn)
    else:
        train_dl = DataLoader(train_ds,
                              batch_size=cfg.train.batch_size,
                              num_workers=cfg.train.num_workers,
                              pin_memory=cfg.base.cuda,
                              shuffle=True,
                              drop_last=True,
                              worker_init_fn=worker_init_fn)

    if cfg.test.num_workers > 1:
        if "val" in cfg.data.eval_type:
            val_dl = DataLoader(val_ds,
                                batch_size=cfg.test.batch_size,
                                num_workers=cfg.test.num_workers,
                                pin_memory=cfg.base.cuda,
                                shuffle=False,
                                prefetch_factor=cfg.data.prefetch_factor,
                                drop_last=False)
        else:
            val_dl = None
    else:
        if "val" in cfg.data.eval_type:
            val_dl = DataLoader(val_ds, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=cfg.base.cuda, shuffle=False, drop_last=False)
        else:
            val_dl = None

    if cfg.test.num_workers > 1:
        if "test" in cfg.data.eval_type:
            test_dl = DataLoader(test_ds,
                                 batch_size=cfg.test.batch_size,
                                 num_workers=cfg.test.num_workers,
                                 pin_memory=cfg.base.cuda,
                                 shuffle=False,
                                 prefetch_factor=cfg.data.prefetch_factor,
                                 drop_last=False)
        else:
            test_dl = None
    else:
        if "test" in cfg.data.eval_type:
            test_dl = DataLoader(test_ds, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=cfg.base.cuda, shuffle=False, drop_last=False)
        else:
            test_dl = None

    dl, ds = {}, {}
    dl["train"] = train_dl
    ds["train"] = train_ds
    if val_dl is not None:
        dl["val"] = val_dl
        ds["val"] = val_ds
    if test_dl is not None:
        dl["test"] = test_dl
        ds["test"] = test_ds

    return dl, ds
