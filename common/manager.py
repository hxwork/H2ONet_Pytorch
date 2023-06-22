import os
import numpy as np
import torch
from termcolor import colored
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from common import tool


class Manager():

    def __init__(self, model, optimizer, scheduler, cfg, dataloader, dataset, logger):
        # Config status
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.dataset = dataset
        self.logger = logger

        # Init some recorders
        self.init_status()
        self.init_tb()

    def init_status(self):
        self.epoch = 0
        self.step = 0
        # Train status: model, optimizer, scheduler, epoch, step
        self.train_status = {}
        # Loss status
        self.loss_status = defaultdict(tool.AverageMeter)
        # Metric status: val, test
        self.metric_status = defaultdict(lambda: defaultdict(tool.AverageMeter))
        # Score status: val, test
        self.score_status = {}
        for split in ["val", "test"]:
            self.score_status[split] = {"cur": np.inf, "best": np.inf}

    def init_tb(self):
        # Tensorboard
        loss_tb_dir = os.path.join(self.cfg.base.model_dir, "summary/loss")
        os.makedirs(loss_tb_dir, exist_ok=True)
        self.loss_writter = SummaryWriter(log_dir=loss_tb_dir)
        metric_tb_dir = os.path.join(self.cfg.base.model_dir, "summary/metric")
        os.makedirs(metric_tb_dir, exist_ok=True)
        self.metric_writter = SummaryWriter(log_dir=metric_tb_dir)

    def update_step(self):
        self.step += 1

    def update_epoch(self):
        self.epoch += 1

    def update_loss_status(self, loss, batch_size):
        for k, v in loss.items():
            self.loss_status[k].update(val=v.item(), num=batch_size)

    def update_metric_status(self, metric, split, batch_size):
        for k, v in metric.items():
            self.metric_status[split][k].update(val=v.item(), num=batch_size)
            self.score_status[split]["cur"] = self.metric_status[split][self.cfg.metric.major_metric].avg

    def reset_loss_status(self):
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_status[split][k].reset()

    def tqdm_info(self, split):
        if split == "train":
            exp_name = self.cfg.base.model_dir.split("/")[-1]
            print_str = "{} E:{:2d}, lr:{:.2E} ".format(exp_name, self.epoch, self.scheduler.get_last_lr()[0])
            print_str += "loss: {:.4f}/{:.4f}".format(self.loss_status["total"].val, self.loss_status["total"].avg)
        else:
            print_str = ""
            for k, v in self.metric_status[split].items():
                print_str += "{}: {:.4f}/{:.4f}".format(k, v.val, v.avg)
        return print_str

    def print_metric(self, split, only_best=False):
        is_best = self.score_status[split]["cur"] < self.score_status[split]["best"]
        color = "white" if split == "val" else "red"
        print_str = " | ".join("{}: {:4g}".format(k, v.avg) for k, v in self.metric_status[split].items())
        if only_best:
            if is_best:
                self.logger.info(colored("Best Epoch: {}, {} Results: {}".format(self.epoch, split, print_str), color, attrs=["bold"]))
        else:
            self.logger.info(colored("Epoch: {}, {} Results: {}".format(self.epoch, split, print_str), color, attrs=["bold"]))

    def write_loss_to_tb(self, split):
        if self.step % self.cfg.summary.save_summary_steps == 0:
            for k, v in self.loss_status.items():
                self.loss_writter.add_scalar("{}_loss/{}".format(split, k), v.val, self.step)

    def write_metric_to_tb(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_writter.add_scalar("{}_metric/{}".format(split, k), v.avg, self.epoch)

    def write_custom_info_to_tb(self, input, output, split):

        # if self.cfg.model.name == "hand_occ_net":
        #     pass

        # else:
        #     raise NotImplementedError
        pass

    def save_ckpt(self):
        # Save latest and best metrics
        for split in ["val", "test"]:
            if split not in self.dataloader:
                continue
            latest_metric_path = os.path.join(self.cfg.base.model_dir, "{}_metric_latest.json".format(split))
            tool.save_dict_to_json(self.metric_status[split], latest_metric_path)
            is_best = self.score_status[split]["cur"] < self.score_status[split]["best"]
            if is_best:
                self.score_status[split]["best"] = self.score_status[split]["cur"]
                best_metric_path = os.path.join(self.cfg.base.model_dir, "{}_metric_best.json".format(split))
                tool.save_dict_to_json(self.metric_status[split], best_metric_path)

        # Model states
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "score_status": self.score_status
        }
        # Save middle checkpoint
        if self.epoch % 10 == 0:
            middle_ckpt_path = os.path.join(self.cfg.base.model_dir, "model_{}.pth".format(self.epoch))
            torch.save(state, middle_ckpt_path)
            self.logger.info("Saved middle checkpoint to: {}".format(middle_ckpt_path))

        # Save latest checkpoint
        if self.epoch % self.cfg.summary.save_latest_freq == 0:
            latest_ckpt_path = os.path.join(self.cfg.base.model_dir, "model_latest.pth")
            torch.save(state, latest_ckpt_path)
            self.logger.info("Saved latest checkpoint to: {}".format(latest_ckpt_path))

        # Save latest and best checkpoints
        for split in ["val", "test"]:
            if split not in self.dataloader:
                continue
            # Above code has updated the best score to cur
            is_best = self.score_status[split]["cur"] == self.score_status[split]["best"]
            if is_best:
                self.logger.info("Current is {} best, score={:.7f}".format(split, self.score_status[split]["best"]))
                # Save best checkpoint
                if self.epoch > self.cfg.summary.save_best_after:
                    best_ckpt_path = os.path.join(self.cfg.base.model_dir, "{}_model_best.pth".format(split))
                    torch.save(state, best_ckpt_path)
                    self.logger.info("Saved {} best checkpoint to: {}".format(split, best_ckpt_path))

    def load_ckpt(self):
        state = torch.load(self.cfg.base.resume)

        ckpt_component = []

        if "state_dict" in state and self.model is not None:
            try:
                self.model.load_state_dict(state["state_dict"])
            except RuntimeError:
                self.logger.info("Using custom loading net")
                net_dict = self.model.state_dict()
                if "module" not in list(state["state_dict"].keys())[0]:
                    state_dict = {"module." + k: v for k, v in state["state_dict"].items() if "module." + k in net_dict.keys()}
                else:
                    state_dict = {k: v for k, v in state["state_dict"].items() if k in net_dict.keys()}
                net_dict.update(state_dict)
                self.model.load_state_dict(net_dict, strict=False)
            ckpt_component.append("net")

        if not self.cfg.base.only_weights:
            if "optimizer" in state and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                except RuntimeError:
                    self.logger.info("Using custom loading optimizer")
                    optimizer_dict = self.optimizer.state_dict()
                    state_dict = {k: v for k, v in state["optimizer"].items() if k in optimizer_dict.keys()}
                    optimizer_dict.update(state_dict)
                    self.optimizer.load_state_dict(optimizer_dict)
                ckpt_component.append("opt")

            if "scheduler" in state and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(state["scheduler"])
                except RuntimeError:
                    self.logger.info("Using custom loading scheduler")
                    scheduler_dict = self.scheduler.state_dict()
                    state_dict = {k: v for k, v in state["scheduler"].items() if k in scheduler_dict.keys()}
                    scheduler_dict.update(state_dict)
                    self.scheduler.load_state_dict(scheduler_dict)
                ckpt_component.append("sch")

            if "step" in state:
                self.step = state["step"] + 1
                ckpt_component.append("step")

            if "epoch" in state:
                self.epoch = state["epoch"] + 1
                ckpt_component.append("epoch")

            if "score_status" in state:
                self.score_status = state["score_status"]
                ckpt_component.append("score status: {}".format(self.score_status))

        ckpt_component = ", ".join(i for i in ckpt_component)
        self.logger.info("Loaded models from: {}".format(self.cfg.base.resume))
        self.logger.info("Ckpt load: {}".format(ckpt_component))
