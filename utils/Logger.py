import json
import os
from time import time
from sklearn.metrics import classification_report
import torch
from sklearn.metrics import confusion_matrix

# avoid
# SystemError: google/protobuf/pyext/descriptor.cc:358: bad argument to internal function
# if using torch 1.90
from .tf_logger import TFLogger

default_output_dir = "/tmp/logs"


class Logger:
    def __init__(self, args, config, update_frequency=10, trainFold=0):
        self.current_iter = 0
        self.current_epoch = 0
        self.max_epochs = config["epoch"]
        self.last_update = time()
        self.start_time = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        self.trainFold = trainFold

        self.args = args
        self.config = config
        # TODO add relation with clearml id
        # TODO refactor
        self.log_path = self.get_name_from_args(args, config)
        self.log_path += f"_kf{self.trainFold}"
        if args.tf_logger:
            self.tf_logger = TFLogger(self.log_path)
        else:
            self.tf_logger = None

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        print()
        print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()
        if self.tf_logger:
            for n, v in enumerate(self.lrs):
                self.tf_logger.scalar_summary("aux/lr%d" % n, v, self.current_epoch)

    def log(self, it, iters, losses, samples_right, total_samples):
        self.current_iter += 1

        for k, v in losses.items():
            running_loss = self.epoch_loss.get(k, 0.0)
            self.epoch_loss[k] = running_loss + v
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses.items()])

        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
            sample_num = self.total.get(k, 0.0)
            self.total[k] = sample_num + total_samples[k]
        acc_string = ", ".join(
            [
                "%s : %.2f" % (k, 100 * (v / total_samples[k]))
                for k, v in samples_right.items()
            ]
        )

        if it % self.update_f == 0:
            print(
                "[%d/%d iteration of epoch %d/%d] \n {loss} %s \n {acc} %s"
                % (
                    it + 1,
                    iters,
                    self.current_epoch,
                    self.max_epochs,
                    loss_string,
                    acc_string,
                )
            )
            # update tf log
            if self.tf_logger:
                for k, v in losses.items():
                    self.tf_logger.scalar_summary(
                        f"train_step/loss_kf{self.trainFold}_{k}", v, self.current_iter
                    )

        if (it + 1) % iters == 0:
            epoch_loss_string = ", ".join(
                ["%s : %.3f" % (k, v / iters) for k, v in self.epoch_loss.items()]
            )
            epoch_acc_string = ", ".join(
                [
                    "%s : %.2f" % (k, 100 * (v / self.total[k]))
                    for k, v in self.epoch_stats.items()
                ]
            )
            print("-" * 30)
            print("<Losses on train> " + epoch_loss_string)
            print("<Accuracies on train> " + epoch_acc_string)
            print("Train epoch time: %.3f" % (time() - self.last_update))
            if self.tf_logger:
                for k, v in self.epoch_loss.items():
                    self.tf_logger.scalar_summary(
                        f"train_epoch/loss_kf{self.trainFold}_{k}",
                        v / iters,
                        self.current_epoch,
                    )
                for k, v in self.epoch_stats.items():
                    self.tf_logger.scalar_summary(
                        f"train_epoch/acc__kf{self.trainFold}_{k}",
                        v / self.total[k],
                        self.current_epoch,
                    )

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.epoch_loss = {}
        self.total = {}

    def log_test(self, phase, accuracies, y_true, y_pred):
        print("-" * 30)
        print(
            "<Accuracies on %s> " % phase
            + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()])
        )
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        # report_dict = classification_report(y_true, y_pred, output_dict=True)
        # # TODO refactor
        # if self.tf_logger:
        #     for k, v in report_dict.items():
        #         # report_dict['accuracy'] = float(acc)
        #         if k =='accuracy':
        #             continue
        #         for kk, vv in v.items():
        #             if kk == 'support':
        #                 continue
        #             self.tf_logger.scalar_summary(
        #                 "%s_%s/%s" % (phase,k, kk), vv, self.current_epoch
        #             )

        if self.tf_logger:
            for k, v in accuracies.items():
                self.tf_logger.scalar_summary(
                    f"{phase}/acc_kf{self.trainFold}_{k}", v, self.current_epoch
                )

    def finish_epoch(self):
        print("-" * 30)
        print("Total epoch time: %.3f" % (time() - self.last_update))

    def save_best_acc(
        self,
        val_res=None,
        test_res=None,
        val_best=None,
        idx_val_best=None,
        name="best_acc.json",
    ):
        # idx_val_best = val_res.argmax()
        idx_test_best = test_res.argmax()
        print()
        print("*" * 30)
        # print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_val_best], test_res.max()))
        print(
            "Best val %g, corresponding test %g - best test: %g"
            % (val_best, test_res[idx_val_best], test_res.max())
        )
        best_acc_dict = {
            # 'epoch_val_best': idx_val_best.item() + 1,
            "epoch_val_best": idx_val_best + 1,
            # 'acc_val_best': val_res.max().item(),
            "acc_val_best": val_best.item(),
            "acc_val_best_test": test_res[idx_val_best].item(),
            "epoch_test_best": idx_test_best.item() + 1,
            "acc_test_best": test_res.max().item(),
        }
        with open(os.path.join(self.log_path, name), "w", encoding="utf-8") as file:
            json.dump(best_acc_dict, file, indent=4)

    def save_last_acc(
        self, val_acc, val_test_acc, test_acc_list, epoch, name="last_acc.json"
    ):
        idx_test_best = test_acc_list.argmax()
        best_acc_dict = {
            "epoch_val_last": epoch + 1,
            "acc_val_best": val_acc.item(),
            "acc_val_best_test": val_test_acc.item(),
            "epoch_test_best": idx_test_best.item() + 1,
            "acc_test_best": test_acc_list.max().item(),
        }
        with open(os.path.join(self.log_path, name), "w", encoding="utf-8") as file:
            json.dump(best_acc_dict, file, indent=4)

    def save_best_model(self, encoder, classifier, val_acc, name="best_model"):
        output_path = os.path.join(self.log_path, f"{name}.tar")
        torch.save(
            {
                "epoch": self.current_epoch,
                "val_acc": val_acc,
                "encoder_state_dict": encoder.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
            },
            output_path,
        )
        return output_path

    def save_args(self):
        args_dict = vars(self.args)
        with open(
            os.path.join(self.log_path, "args.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(args_dict, file, indent=4)

    def save_config(self):
        with open(
            os.path.join(self.log_path, "config.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(self.config, file, indent=4)

    @staticmethod
    def get_name_from_args(args, config):
        meta_name = args.config.replace("/", "_")
        if meta_name.endswith(".py"):
            meta_name.replace(".py", "")
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = default_output_dir
        folder_name = os.path.join(output_dir, meta_name)

        # time_name = strftime("%Y-%m-%d-%H-%M-%S", localtime())
        name = os.path.join(folder_name, config["task_id"])

        if not os.path.exists(name):
            os.makedirs(name)
        return name
