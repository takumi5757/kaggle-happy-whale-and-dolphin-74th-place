from __future__ import annotations

import json
import os

import pandas as pd
from tqdm.auto import tqdm


def coco2classification_df(
    ann_file: str, categories: list[dict] = None, img_dir: str = "/data"
) -> pd.DataFrame:

    list_images: list = []
    list_labels: list = []
    with open(ann_file, encoding="utf-8") as f:
        tmp_dict: dict = json.load(f)

    anns: list = tmp_dict["annotations"]
    imgs: list = tmp_dict["images"]

    # categoryの指定があるか
    if categories:
        categories: list = categories
    else:
        categories: list = tmp_dict["categories"]

    for dict_img in tqdm(imgs):
        file_name: str = dict_img["file_name"]
        key: int = dict_img["id"]
        dict_ann: dict = list(filter(lambda x: x["image_id"] == key, anns))[0]
        categoriy_id: int = dict_ann["category_id"]
        label: str = list(filter(lambda x: x["id"] == categoriy_id, categories))[0][
            "name"
        ]
        img_path = os.path.join(img_dir, file_name)
        assert os.path.isfile(img_path), f"FileNotFoundError: No such file: {img_path}"
        list_images.append(os.path.join(img_dir, file_name))
        list_labels.append(label)

    df: pd.DataFrame = pd.DataFrame({"images": list_images, "labels": list_labels})

    return df


# df.to_csv('/data/train.txt',
#           sep=' ', index=False, header=False, columns=['images', 'labels'])
