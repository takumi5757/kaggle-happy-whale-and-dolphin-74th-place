from torch import optim
from timm.scheduler import CosineLRScheduler


def get_optim_and_scheduler(network, optimizer_config):
    params = network.parameters()

    if optimizer_config["optim_type"] == "sgd":
        optimizer = optim.SGD(
            params,
            weight_decay=optimizer_config["weight_decay"],
            momentum=optimizer_config["momentum"],
            nesterov=optimizer_config["nesterov"],
            lr=optimizer_config["lr"],
        )
    elif optimizer_config["optim_type"] == "adam":
        optimizer = optim.Adam(
            params,
            weight_decay=optimizer_config["weight_decay"],
            lr=optimizer_config["lr"],
        )
    else:
        raise ValueError("Optimizer not implemented")

    if optimizer_config["sched_type"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=optimizer_config["lr_decay_step"],
            gamma=optimizer_config["lr_decay_rate"],
        )
    elif optimizer_config["sched_type"] == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optimizer_config["lr_decay_step"],
            gamma=optimizer_config["lr_decay_rate"],
        )
    elif optimizer_config["sched_type"] == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=optimizer_config["lr_decay_rate"]
        )
    elif optimizer_config["sched_type"] == "cosineannealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimizer_config["t_max"], eta_min=0.0
        )
    elif optimizer_config["sched_type"] == "cosineannealingwarmup":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=optimizer_config["t_max"],
            lr_min=1e-6,
            warmup_t=5,
            warmup_lr_init=5e-6,
            warmup_prefix=False,
        )
    else:
        raise ValueError(f'Scheduler {optimizer_config["sched_type"]} not implemented')

    return optimizer, scheduler
