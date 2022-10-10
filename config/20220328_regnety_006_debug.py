config = {}

batch_size = 512
epoch = 2
lr = 0.01
lr_decay_rate = 0.1
image_size = 64
num_classes = 15587

config["batch_size"] = batch_size
config["epoch"] = epoch
config["num_classes"] = num_classes
config["image_size"] = image_size
# network configs
networks = {}

encoder = {"name": "regnety_006", "pretrained": True}
networks["encoder"] = encoder

classifier = {
    "name": "poolarcface",
    "in_dim": 608,
    "num_classes": num_classes,
    "cls_type": "linear",
    "s": 30,
    "m": 0.30,
    "ls_eps": 0.0,
}
networks["classifier"] = classifier

config["networks"] = networks

config["loss"] = {"name": "crossentropy", "label_smoothing": 0.0, "gamma": 2}
# optimizer configs
optimizer = {}

encoder_optimizer = {
    "optim_type": "sgd",
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "cosineannealingwarmup",
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate,
    "t_max": int(epoch),
}
optimizer["encoder_optimizer"] = encoder_optimizer

classifier_optimizer = {
    "optim_type": "sgd",
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "cosineannealingwarmup",
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate,
    "t_max": int(epoch),
}
optimizer["classifier_optimizer"] = classifier_optimizer
config["optimizer"] = optimizer
