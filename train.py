import argparse
import os
import torch
from functools import partial
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)
from data_loader.data_loader import fetch_dataloader
from model.model import fetch_model
from optimizer.optimizer import fetch_optimizer
from loss.loss import compute_loss, compute_metric
from common import tool
from common.manager import Manager
from common.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="", type=str, help="Directory containing params.json")
parser.add_argument("--resume", default=None, type=str, help="Path of model weights")
parser.add_argument("-ow", "--only_weights", action="store_true", help="Only load model weights or load all train status")


def train(model, mng: Manager):
    # Reset loss status
    mng.reset_loss_status()
    # Set model to training mode
    torch.cuda.empty_cache()
    model.train()
    # Use tqdm for progress bar
    t = tqdm(total=len(mng.dataloader["train"]))
    # Train loop
    for batch_idx, batch_input in enumerate(mng.dataloader["train"]):
        # Move input to GPU if available
        batch_input = tool.tensor_gpu(batch_input)
        # Compute model output and loss
        batch_output = model(batch_input)
        loss = compute_loss(mng.cfg, batch_input, batch_output)
        # Update loss status and print current loss and average loss
        mng.update_loss_status(loss=loss, batch_size=mng.cfg.train.batch_size)
        # Clean previous gradients, compute gradients of all variables wrt loss
        mng.optimizer.zero_grad()
        # import pdb
        # pdb.set_trace()
        loss["total"].backward()
        # add gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # Perform updates using calculated gradients
        mng.optimizer.step()
        # Update step: step += 1
        mng.update_step()
        # Write loss to tensorboard
        mng.write_loss_to_tb(split="train")
        # Write custom info to tensorboard
        mng.write_custom_info_to_tb(batch_input, batch_output, split="train")
        # Training info print
        print_str = mng.tqdm_info(split="train")
        # Tqdm settings
        t.set_description(desc=print_str)
        t.update()

    # Close tqdm
    t.close()


def evaluate(model, mng: Manager):
    # Set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        # Compute metrics over the dataset
        for split in ["val", "test"]:
            if split not in mng.dataloader:
                continue
            # Initialize loss and metric statuses
            mng.reset_loss_status()
            mng.reset_metric_status(split)
            cur_sample_idx = 0
            for batch_idx, batch_input in enumerate(mng.dataloader[split]):
                # Move data to GPU if available
                batch_input = tool.tensor_gpu(batch_input)
                # Compute model output
                batch_output = model(batch_input)
                # Get real batch size
                if "img" in batch_input:
                    batch_size = batch_input["img"].size()[0]
                elif "img_0" in batch_input:
                    batch_size = batch_input["img_0"].size()[0]
                else:
                    batch_size = mng.cfg.test.batch_size
                # # Compute all loss on this batch
                # loss = compute_loss(mng.cfg, batch_input, batch_output)
                # mng.update_loss_status(loss, batch_size)
                # Compute all metrics on this batch
                if "DEX_YCB" in mng.cfg.data.name:
                    metric = compute_metric(mng.cfg, batch_input, batch_output)
                    batch_output = tool.tensor_gpu(batch_output, check_on=False)
                    batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
                    # evaluate
                    custom_metric = mng.dataset[split].evaluate(batch_output, cur_sample_idx)
                    cur_sample_idx += len(batch_output)

                    metric.update(custom_metric)
                else:
                    metric = compute_metric(mng.cfg, batch_input, batch_output)
                mng.update_metric_status(metric, split, batch_size)

            # Update data to tensorboard
            mng.write_metric_to_tb(split)
            # # Write custom info to tensorboard
            # mng.write_custom_info_to_tb(batch_input, batch_output, split)
            # For each epoch, update and print the metric
            mng.print_metric(split, only_best=False)


def train_and_evaluate(model, mng: Manager):
    mng.logger.info("Starting training for {} epoch(s)".format(mng.cfg.train.num_epochs))
    # Load weights from restore_file if specified
    if mng.cfg.base.resume is not None:
        mng.load_ckpt()

    for epoch in range(mng.epoch, mng.cfg.train.num_epochs):
        # Train one epoch
        train(model, mng)
        # Evaluate one epoch
        evaluate(model, mng)
        # Check if current is best, save best and latest checkpoints
        mng.save_ckpt()
        # Update scheduler
        mng.scheduler.step()
        # Update epoch: epoch += 1
        mng.update_epoch()


def main(cfg):
    # Set the logger
    logger = tool.set_logger(os.path.join(cfg.base.model_dir, "train.log"))
    # Print GPU ids

    gpu_ids = ", ".join(str(i) for i in [j for j in range(cfg.base.num_gpu)])
    logger.info("Using GPU ids: [{}]".format(gpu_ids))
    # Fetch dataloader
    dl, ds = fetch_dataloader(cfg)
    # Fetch model
    model = fetch_model(cfg)
    # Define optimizer and scheduler
    optimizer, scheduler = fetch_optimizer(cfg, model)
    # Initialize manager
    mng = Manager(model=model, optimizer=optimizer, scheduler=scheduler, cfg=cfg, dataloader=dl, dataset=ds, logger=logger)
    # Train the model
    train_and_evaluate(model, mng)


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "cfg.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path).cfg
    # Update args into cfg.base
    cfg.base.update(vars(args))
    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    # Main function
    main(cfg)
