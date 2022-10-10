import os
import glob


os.system(
    "python train_cropped_all.py "
    "./config/20220326_efficinetnetb6.py "
    "--output_dir ./output/ "
    "--tf_logger "
    "--mixed_precision"
)

os.system(
    "python train_cropped_all.py "
    "./config/20220325_convnext.py "
    "--output_dir ./output/ "
    "--tf_logger "
    "--mixed_precision"
)
