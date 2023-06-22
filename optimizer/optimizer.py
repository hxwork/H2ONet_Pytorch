import torch.optim as optim


def fetch_optimizer(cfg, model):
    total_params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.name == "adam":
        # optimizer = optim.Adam(total_params, lr=cfg.optimizer.lr, weight_decay=1e-4)
        optimizer = optim.Adam(total_params, lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "sgd":
        optimizer = optim.SGD(total_params, lr=cfg.optimizer.lr)
    else:
        raise NotImplementedError("Unknown optimizer type: {}.".format(cfg.optimizer.name))

    if cfg.scheduler.name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    elif cfg.scheduler.name == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma)
    else:
        raise NotImplementedError("Unknown scheduler type: {}.".format(cfg.scheduler.name))
    return optimizer, scheduler
