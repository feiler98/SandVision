# imports
# ----------------------------------------------------------------------------------------------------------------------
from lib.data_transform_utils import video_to_image_seq
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
path_data_origin = Path('/Volumes/FR external SSD/ML_RUAN/rename vids/14.nov')
path_data_out = Path('/Volumes/FR external SSD/ML_RUAN/ruan_ml_curated_data')


paths_video = [p for p in path_data_origin.rglob("*.m*")]
for p in paths_video:
    video_to_image_seq(p, p.stem, path_data_out)


if __name__ == "__main__":
    pass