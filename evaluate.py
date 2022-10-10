import argparse
import os
from datetime import datetime as dt

import numpy as np
import torch

from torch import nn

from model.model import get_classifier_from_config, get_encoder_from_config

from utils.Logger import Logger
from utils.tools import calculate_correct
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors


def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l, p in zip(labels, predictions)])


def PredictGrid(
    train_cnn_predictions,
    valid_cnn_predictions,
    train_labels,
    valid_labels,
    new_individual_thres,
):
    neigh = NearestNeighbors(n_neighbors=100, metric="cosine")
    neigh.fit(train_cnn_predictions)

    distances, idxs = neigh.kneighbors(valid_cnn_predictions, return_distance=True)
    conf = 1 - distances
    preds = []

    for j in range(len(idxs)):
        preds.append(list(train_labels[idxs[j]]))

    allTop5Preds = []
    valid_labels_list = []
    for i in range(len(preds)):
        valid_labels_list.append((valid_labels[i]))

        predictTop = preds[i][:5]
        Top5Conf = conf[i][:5]

        if Top5Conf[0] < new_individual_thres:

            tempList = [
                "new_individual",
                predictTop[0],
                predictTop[1],
                predictTop[2],
                predictTop[3],
            ]
            allTop5Preds.append(tempList)

        elif Top5Conf[1] < new_individual_thres:

            tempList = [
                predictTop[0],
                "new_individual",
                predictTop[1],
                predictTop[2],
                predictTop[3],
            ]
            allTop5Preds.append(tempList)

        elif Top5Conf[2] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                "new_individual",
                predictTop[2],
                predictTop[3],
            ]
            allTop5Preds.append(tempList)

        elif Top5Conf[3] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                predictTop[2],
                "new_individual",
                predictTop[3],
            ]
            allTop5Preds.append(tempList)

        elif Top5Conf[4] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                predictTop[2],
                predictTop[3],
                "new_individual",
            ]
            allTop5Preds.append(tempList)

        else:
            allTop5Preds.append(predictTop)

        if ("new_individual" in allTop5Preds[-1]) and (
            valid_labels_list[i] not in train_labels
        ):
            allTop5Preds[-1] = [
                valid_labels_list[i] if x == "new_individual" else x
                for x in allTop5Preds[-1]
            ]

    score = map_per_set(valid_labels_list, allTop5Preds)

    return score


def PredictGridV2(
    train_cnn_predictions,
    valid_cnn_predictions,
    train_labels,
    valid_labels,
    new_individual_thres,
    df_valid,
):
    neigh = NearestNeighbors(n_neighbors=100, metric="cosine")
    neigh.fit(train_cnn_predictions)

    distances, idxs = neigh.kneighbors(valid_cnn_predictions, return_distance=True)
    conf = 1 - distances
    preds = []

    for j in range(len(idxs)):
        preds.append(list(train_labels[idxs[j]]))

    allTop5Preds = []
    valid_labels_list = []
    for i in range(len(preds)):
        valid_labels_list.append((valid_labels[i]))

        predictTop = preds[i][:5]
        Top5Conf = conf[i][:5]

        if Top5Conf[0] < new_individual_thres:

            tempList = [
                "new_individual",
                predictTop[0],
                predictTop[1],
                predictTop[2],
                predictTop[3],
            ]

        elif Top5Conf[1] < new_individual_thres:

            tempList = [
                predictTop[0],
                "new_individual",
                predictTop[1],
                predictTop[2],
                predictTop[3],
            ]

        elif Top5Conf[2] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                "new_individual",
                predictTop[2],
                predictTop[3],
            ]

        elif Top5Conf[3] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                predictTop[2],
                "new_individual",
                predictTop[3],
            ]

        elif Top5Conf[4] < new_individual_thres:

            tempList = [
                predictTop[0],
                predictTop[1],
                predictTop[2],
                predictTop[3],
                "new_individual",
            ]

        else:
            tempList = predictTop

        if ("new_individual" in tempList) and (
            valid_labels_list[i] not in train_labels
        ):
            tempList = [
                valid_labels_list[i] if x == "new_individual" else x for x in tempList
            ]

        allTop5Preds.append(tempList)
        df_valid.loc[i, new_individual_thres] = map_per_image(valid_labels[i], tempList)

    cv = df_valid[new_individual_thres].mean()

    return cv, df_valid


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Experiment configs")
    parser.add_argument("ckpt", help="checkpoint file")
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

    args = parser.parse_args()
    config_file = "config." + os.path.basename(args.config).split(".")[0]
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Evaluator:
    def __init__(self, args, config, device, model_path):
        self.args = args
        self.config = config
        self.device = device
        self.model_path = model_path
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

        if self.model_path is not None:
            state_dict = torch.load(
                self.model_path, map_location=lambda storage, loc: storage
            )

            encoder_state = state_dict["encoder_state_dict"]
            # if trained multi_GPU
            encoder_state = fix_key(encoder_state)
            classifier_state = state_dict["classifier_state_dict"]
            # if trained multi_GPU
            classifier_state = fix_key(classifier_state)
            self.encoder.load_state_dict(encoder_state)
            self.classifier.load_state_dict(classifier_state)
            print("model loaded")

    def do_eval(self, loader):
        correct = 0
        y_pred: list = []
        y_true: list = []
        for it, (batch, label) in tqdm(enumerate(loader), total=len(loader)):
            batch = batch["image"]
            batch = batch.to(self.device)
            label = label.to(self.device)
            features = self.encoder(batch)
            if self.config["networks"]["classifier"]["name"] == "arcface":
                scores = self.classifier(features, label)
            else:
                scores = self.classifier(features)
            correct += calculate_correct(scores, label)
            y_true += list(label.cpu().numpy())
            y_pred += list(scores.max(dim=1)[1].cpu().numpy())
        return correct, y_true, y_pred

    def do_testing(self):
        # evaluation
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct, y_true, y_pred = self.do_eval(loader)
                class_acc = float(class_correct) / total
                print(f"{phase} acc: {class_acc}")
                show_confusion_matrix(
                    y_true,
                    y_pred,
                    file_path=os.path.join(
                        self.args.output_dir,
                        f"{os.path.basename(self.args.config).split('.')[0]}_{phase}.png",
                    ),
                )


# temporary func for load multi-GPU trained model
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def show_confusion_matrix(y_true, y_pred, file_path):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(
        cm,
        square=True,
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
        annot=True,
        cmap="Blues",
    )
    plt.xlabel("Predict", fontsize=13)
    plt.ylabel("GT", fontsize=13)
    plt.savefig(file_path)
    plt.close()


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(args, config, device)
    evaluator.do_testing()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    dt_now = dt.now()

    main()
