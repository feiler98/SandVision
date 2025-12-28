# imports
# ----------------------------------------------------------------------------------------------------------------------
from lib.img_seg_ml import pred_by_model
from lib.ml_pred_postprocess_utils import mask_result_eval
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------


def analysis_img_seq(path_img_dir: (str | Path)):
    """
    Main prediction-pipeline using an image sequence (time-stamp transformation is recommended).

    Parameters
    ----------
        path_img_dir: str | Path
    """
    # paths
    path_img_dir = Path(path_img_dir)
    path_out = path_img_dir.parent / f"{path_img_dir.name}__out_result"
    path_models = Path(__file__).parent / "ml_model"
    if not path_models.exists():
        raise ValueError(f"Model directory '{path_models}' does not exist!")

    model_tags_list = ["__mask_circle_chamber", "__mask_circle_dot", "__mask_sand"]
    dict_model_file_path = {tag: list(path_models.glob(f"*{tag}.pth.tar"))[0] for tag in model_tags_list}

    for tag in model_tags_list:
        pred_by_model(path_img_dir, dict_model_file_path[tag], tag)

    mask_result_eval(path_img_dir)


if __name__ == "__main__":
    analysis_img_seq("/home/wernerfeiler/muenster/SandVision/input_data/example_out")

