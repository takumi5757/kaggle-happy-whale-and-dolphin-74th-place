import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class HappyWhaleDataset(Dataset):
    def __init__(self, df, transforms=None, dummy_label=False):
        self.df = df
        self.file_names = df["file_path"].values
        if dummy_label:
            # native list cause memory leak??
            self.labels = [0] * len(self.df)
            # self.labels = np.zeros(len(self.df), dtype=np.uint8)
        else:
            self.labels = df["individual_id"].values

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


# inputs should be config
def get_data_transforms(img_size):
    data_transforms = {
        "train": A.Compose(
            [
                A.Resize(img_size, img_size, interpolation=2),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=4, sat_shift_limit=8, val_shift_limit=8, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "valid": A.Compose(
            [
                A.Resize(img_size, img_size, interpolation=2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    }
    return data_transforms


def get_loader(df, config, data_transforms):
    df = df.reset_index(drop=True)
    dataset = HappyWhaleDataset(df, transforms=data_transforms)

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    return loader
