import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import argparse
from clearml import Task
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
from utils.tools import calculate_correct
from datetime import datetime as dt

from data.data import get_data_transforms, get_loader
from model.model import get_classifier_from_config, get_encoder_from_config
from data.data import get_data_transforms, get_loader, HappyWhaleDataset
from utils.Logger import Logger
from utils.tools import calculate_correct
from tqdm.auto import tqdm
from collections import OrderedDict
from evaluate import Evaluator
from evaluate import PredictGrid, PredictGridV2
from sklearn.neighbors import NearestNeighbors

ROOT_DIR = "."
TRAIN_DIR = "./input/happy-whale-and-dolphin/20220414/train_backfin_box"
TEST_DIR = "./input/happy-whale-and-dolphin/20220414/test_backfin_box"


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    # reproducibile mode
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # fastest mode
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_train_file_path(id):
    return f"{TRAIN_DIR}/{id}"


# temporary func for load multi-GPU trained model
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def GetSubmission(
    train_data,
    valid_data,
    train_labels,
    label_encoder,
    neighbors=100,
    metric="cosine",
    new_individual_thres=0.6,
    csv_name="submission.csv",
):

    neigh = NearestNeighbors(n_neighbors=neighbors, metric=metric)
    neigh.fit(train_data)
    distances, idxs = neigh.kneighbors(valid_data, return_distance=True)
    conf = 1 - distances
    preds = []
    df = pd.read_csv("sample_submission.csv")
    predictTopDecoded = {}
    for i in range(len(idxs)):
        preds.append(train_labels[idxs[i]])

    for i in range(len(distances)):

        predictTop = list(preds[i][:5])
        predictTop = label_encoder.inverse_transform(predictTop)
        topValues = conf[i][:5]

        if topValues[0] < new_individual_thres:

            tempList = [
                "new_individual",
                predictTop[0],
                predictTop[1],
                predictTop[2],
                predictTop[3],
            ]
            predictTopDecoded[df.iloc[i]["image"]] = tempList

        elif topValues[1] < new_individual_thres:

            tempList = [
                predictTop[0],
                "new_individual",
                predictTop[1],
                predictTop[2],
                predictTop[3],
            ]
            predictTopDecoded[df.iloc[i]["image"]] = tempList

        elif topValues[2] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                "new_individual",
                predictTop[2],
                predictTop[3],
            ]
            predictTopDecoded[df.iloc[i]["image"]] = tempList

        elif topValues[3] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                predictTop[2],
                "new_individual",
                predictTop[3],
            ]
            predictTopDecoded[df.iloc[i]["image"]] = tempList

        elif topValues[4] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                predictTop[2],
                predictTop[3],
                "new_individual",
            ]
            predictTopDecoded[df.iloc[i]["image"]] = tempList

        else:
            predictTopDecoded[df.iloc[i]["image"]] = predictTop

    for x in tqdm(predictTopDecoded):
        predictTopDecoded[x] = " ".join(predictTopDecoded[x])

    predictions = pd.Series(predictTopDecoded).reset_index()
    predictions.columns = ["image", "predictions"]
    predictions.to_csv(f"{csv_name}", index=False)


class Trainer:
    def __init__(
        self,
        args,
        config,
        device,
        train_loader,
        val_loader,
        test_loader,
        clearml_logger,
    ):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        # TODO
        encoder = get_encoder_from_config(self.config["networks"]["encoder"])
        if self.config["networks"]["classifier"]["name"] == "poolarcface":
            if "regnet" in self.config["networks"]["encoder"]["name"]:
                encoder.head.global_pool = nn.Identity()
            elif "convnext" in self.config["networks"]["encoder"]["name"]:
                encoder.head = nn.Identity()
            else:
                encoder.global_pool = nn.Identity()
        self.encoder = encoder.to(device)
        self.classifier = get_classifier_from_config(
            self.config["networks"]["classifier"]
        ).to(device)

        if self.args.multi_gpu:
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.classifier = torch.nn.DataParallel(self.classifier)

        # optimizers
        self.encoder_optim, self.encoder_sched = get_optim_and_scheduler(
            self.encoder, self.config["optimizer"]["encoder_optimizer"]
        )
        self.classifier_optim, self.classifier_sched = get_optim_and_scheduler(
            self.classifier, self.config["optimizer"]["classifier_optimizer"]
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.eval_loader = {
            "val": self.val_loader,
            #             "test": self.test_loader,
        }

        self.clearml_logger = clearml_logger

    def _do_epoch(self):
        # default loss setting
        if "loss" not in self.config.keys():
            self.config["loss"] = {"name": "crossentropy"}

        if self.config["loss"]["name"] == "focalloss":
            # criterion = FocalLoss(gamma=self.config["loss"]["gamma"])
            criterion = None
        elif self.config["loss"]["name"] == "crossentropy":
            if "label_smoothing" in self.config["loss"].keys():
                label_smoothing = self.config["loss"]["label_smoothing"]
            else:
                label_smoothing = 0.0
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # turn on train mode
        self.encoder.train()
        self.classifier.train()

        for it, batch in enumerate(self.train_loader):
            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                images = batch["image"]
                images = images.to(self.device)
                label = batch["label"]
                label = label.to(self.device)

                # forward
                loss_dict = {}
                correct_dict = {}
                num_samples_dict = {}
                total_loss = 0.0

                features = self.encoder(images)
                if self.config["networks"]["classifier"]["name"] == "poolarcface":
                    scores = self.classifier(features, label)
                else:
                    scores = self.classifier(features)

                loss_cls = criterion(scores, label)
                loss_dict["main"] = loss_cls.item()
                correct_dict["main"] = calculate_correct(scores, label)
                num_samples_dict["main"] = int(scores.size(0))
                loss_dict["total"] = loss_cls.item()

            # backward
            self.scalar.scale(loss_cls).backward()

            # update
            self.scalar.step(self.encoder_optim)
            self.scalar.step(self.classifier_optim)
            self.scalar.update()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict,
            )
            self.clearml_logger.report_scalar(
                "train", "main", iteration=it, value=loss_dict["main"]
            )
            self.clearml_logger.report_scalar(
                "train", "total", iteration=it, value=loss_dict["total"]
            )

        # step schedulers
        if (
            self.config["optimizer"]["encoder_optimizer"]["sched_type"]
            == "cosineannealingwarmup"
        ):
            self.encoder_sched.step(self.current_epoch + 1)
        else:
            self.encoder_sched.step()
        if (
            self.config["optimizer"]["classifier_optimizer"]["sched_type"]
            == "cosineannealingwarmup"
        ):
            self.classifier_sched.step(self.current_epoch + 1)
        else:
            self.classifier_sched.step()

        if (self.current_epoch % 3 == 0) or (self.current_epoch == self.epochs - 1):
            # turn on eval mode
            self.encoder.eval()
            self.classifier.eval()

            # evaluation
            with torch.no_grad():
                for phase, loader in self.eval_loader.items():
                    total = len(loader.dataset)
                    class_correct, y_true, y_pred = self.do_eval(loader)
                    # TODO move to Logger.py
                    #                 report_dict = classification_report(y_true, y_pred, output_dict=True)
                    #                 for k, v in report_dict.items():
                    #                     # report_dict['accuracy'] = float(acc)
                    #                     if k == "accuracy":
                    #                         continue
                    #                     for kk, vv in v.items():
                    #                         if kk == "support":
                    #                             continue
                    #                         if k == "macro avg":
                    #                             self.clearml_logger.report_scalar(
                    #                                 title=phase,
                    #                                 series=f"{k}_{kk}",
                    #                                 value=vv,
                    #                                 iteration=self.current_epoch + 1,
                    #                             )

                    class_acc = float(class_correct) / total
                    self.logger.log_test(phase, {"class": class_acc}, y_true, y_pred)
                    self.results[phase][self.current_epoch] = class_acc
                    self.clearml_logger.report_scalar(
                        phase, "acc", iteration=self.current_epoch + 1, value=class_acc
                    )

                # save from best val
                if self.results["val"][self.current_epoch] >= self.best_val_acc:
                    self.best_val_acc = self.results["val"][self.current_epoch]
                    self.best_val_epoch = self.current_epoch + 1
                    _ = self.logger.save_best_model(
                        self.encoder, self.classifier, self.best_val_acc
                    )
                if self.current_epoch == self.epochs - 1:
                    self.last_model_path = self.logger.save_best_model(
                        self.encoder,
                        self.classifier,
                        self.results["val"][self.current_epoch],
                        name="last",
                    )

    def do_eval(self, loader):
        correct = 0
        y_pred: list = []
        y_true: list = []
        for it, batch in enumerate(loader):
            images = batch["image"]
            images = images.to(self.device)
            label = batch["label"]
            label = label.to(self.device)
            features = self.encoder(images)
            if self.config["networks"]["classifier"]["name"] == "poolarcface":
                scores = self.classifier(features, label)
            else:
                scores = self.classifier(features)
            correct += calculate_correct(scores, label)
            y_true += list(label.cpu().detach().numpy())
            y_pred += list(scores.max(dim=1)[1].cpu().detach().numpy())
        return correct, y_true, y_pred

    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=10)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {
            "val": torch.zeros(self.epochs),
            "test": torch.zeros(self.epochs),
            # TODO
            "test_llc": torch.zeros(self.epochs),
        }

        self.best_val_acc = 0
        self.best_val_epoch = 0
        if self.args.mixed_precision:
            self.use_amp = True
        else:
            self.use_amp = False
        print(f"use automatic mixed precision {self.use_amp}")
        self.scalar = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        for self.current_epoch in range(self.epochs):
            self.logger.new_epoch(
                [group["lr"] for group in self.encoder_optim.param_groups]
            )

            self._do_epoch()
            # TODO
            # if self.current_epoch % 5 == 0 and self.current_epoch != 0:
            #     print(f"current_epoch: {self.current_epoch}")
            #     self.classifier.step_margin(0.1)
            self.logger.finish_epoch()

        # save from best val
        val_res = self.results["val"]
        test_res = self.results["test"]
        self.logger.save_best_acc(
            val_res, test_res, self.best_val_acc, self.best_val_epoch - 1
        )

        return self.logger


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Experiment configs")
    parser.add_argument(
        "--output_dir", default=None, help="The directory to save logs and models"
    )
    parser.add_argument(
        "--tf_logger",
        action="store_true",
        help="If true will save tensorboard compatible logs",
    )

    parser.add_argument(
        "--multi_gpu", action="store_true", help="If true will use multi-gpu"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="If true will use mixed presicion",
    )

    #     args = parser.parse_args()
    args = parser.parse_args()
    config_file = "config." + os.path.basename(args.config).split(".")[0]
    # temporary fix
    # config_file = "config.20220204_random_search." + os.path.basename(args.config).split(".")[0]
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


if __name__ == "__main__":
    dt_now = dt.now()
    task = Task.init(
        project_name="Kaggle Happy whale",
        task_name=f"saito task_{dt_now.strftime('%Y_%m%d_%H%M%S')}",
    )
    # set_seed(42)
    set_seed(5757)
    args, config = get_args()
    clearml_logger = task.get_logger()
    config["task_id"] = task.id
    config = task.connect(config)
    df = pd.read_csv(f"{ROOT_DIR}/train.csv")
    df["file_path"] = df["image"].apply(get_train_file_path)

    label_encoder = LabelEncoder()
    df["individual_id"] = label_encoder.fit_transform(df["individual_id"])

    n_fold = 10
    skf = StratifiedKFold(n_splits=n_fold)

    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.individual_id)):
        df.loc[val_, "kfold"] = fold

    img_size = config["image_size"]
    trainFold = 1  # this model was trained on this fold
    data_transforms = get_data_transforms(img_size)

    df_train = df[df.kfold != trainFold].reset_index(drop=True)
    df_valid = df[df.kfold == trainFold].reset_index(drop=True)
    train_loader = get_loader(df_train, config, data_transforms["train"])
    val_loader = get_loader(df_valid, config, data_transforms["valid"])
    test_loader = val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        args, config, device, train_loader, val_loader, test_loader, clearml_logger
    )

    trainer.do_training()

    print("Start k-nn evaluate")
    model_path = trainer.last_model_path
    evaluator = Evaluator(args, config, device, model_path)
    df_test = pd.read_csv("sample_submission.csv")

    df_test["file_path"] = df_test["image"].apply(lambda x: f"{TEST_DIR}/{x}")

    # predict first on train dataset to extract embeddings
    train_dataset = HappyWhaleDataset(
        df_train, transforms=data_transforms["valid"], dummy_label=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )

    valid_dataset = HappyWhaleDataset(
        df_valid, transforms=data_transforms["valid"], dummy_label=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )

    test_dataset = HappyWhaleDataset(
        df_test, transforms=data_transforms["valid"], dummy_label=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )

    evaluator.encoder.eval()
    evaluator.classifier.eval()
    outputList = []
    with torch.no_grad():
        for it, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = batch["image"]
            images = images.to(device)
            features = evaluator.encoder(images)
            embeddings = evaluator.classifier.extract(features)
            outputList.extend(embeddings.cpu().detach().numpy())

    df_train_cnn_predictions = np.array(outputList)
    train_cnn_labels = np.array(df_train["individual_id"].values)

    evaluator.encoder.eval()
    evaluator.classifier.eval()
    outputList = []
    with torch.no_grad():
        for it, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            images = batch["image"]
            images = images.to(device)
            features = evaluator.encoder(images)
            embeddings = evaluator.classifier.extract(features)
            outputList.extend(embeddings.cpu().detach().numpy())
    df_valid_cnn_predictions = np.array(outputList)
    valid_cnn_labels = np.array(df_valid["individual_id"].values)

    iteration = 0
    best_score = 0
    best_thres = 0
    for thres in np.arange(0.1, 0.9, 0.1):
        print("iteration ", iteration, " of ", len(np.arange(0.3, 0.9, 0.1)))
        iteration += 1
        # score = PredictGrid(
        #     df_train_cnn_predictions,
        #     df_valid_cnn_predictions,
        #     train_cnn_labels,
        #     valid_cnn_labels,
        #     new_individual_thres=thres,
        # )
        score, df_valid = PredictGridV2(
            df_train_cnn_predictions,
            df_valid_cnn_predictions,
            train_cnn_labels,
            valid_cnn_labels,
            new_individual_thres=thres,
            df_valid=df_valid,
        )
        if score > best_score:
            best_score = score
            best_thres = thres
        print("thres: ", thres, ",score: ", score)
    print("Best score is: ", best_score)
    print("Best thres is: ", best_thres)

    # Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    allowed_targets = set([x for x in np.unique(train_cnn_labels)])

    df_valid.loc[
        ~df_valid.individual_id.isin(allowed_targets), "individual_id"
    ] = "new_individual"
    df_valid["is_new_individual"] = df_valid.individual_id == "new_individual"

    val_scores = df_valid.groupby("is_new_individual").mean().T
    val_scores = val_scores.drop("kfold")
    val_scores["adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_thres = val_scores["adjusted_cv"].idxmax()
    print(f"best_th_adjusted={best_thres}")

    trainer.clearml_logger.report_scalar(
        "knn_val", "score", iteration=1, value=best_score
    )

    evaluator.encoder.eval()
    evaluator.classifier.eval()

    outputList = []
    with torch.no_grad():
        for it, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = batch["image"]
            images = images.to(device)
            features = evaluator.encoder(images)
            embeddings = evaluator.classifier.extract(features)
            outputList.extend(embeddings.cpu().detach().numpy())
    test_cnn_predictions = np.array(outputList)

    allTrainData = np.concatenate((df_train_cnn_predictions, df_valid_cnn_predictions))
    allTrainingLabels = np.concatenate((train_cnn_labels, valid_cnn_labels))

    GetSubmission(
        allTrainData,
        test_cnn_predictions,
        allTrainingLabels,
        label_encoder=label_encoder,
        neighbors=100,
        metric="cosine",
        new_individual_thres=best_thres,
        csv_name=f"./csv/{task.id}.csv",
    )

    task.close()
