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

    path_models = Path(__file__).parent / "ml_model"
    if not path_models.exists():
        raise ValueError(f"Model directory '{path_models}' does not exist!")

    model_tags_list = ["__mask_circle_chamber", "__mask_circle_dot", "__mask_sand"]
    dict_model_file_path = {tag: list(path_models.glob(f"*{tag}.pth.tar"))[0] for tag in model_tags_list}
    print("""
###################
# Mask Prediction #
###################
""")
    for tag in model_tags_list:
        print_text = f"Prediction of '{tag}'"
        print(print_text)
        print("-"*len(print_text))
        pred_by_model(path_img_dir, dict_model_file_path[tag], tag)

    print("""
###################
# Data Evaluation #
###################
""")
    mask_result_eval(path_img_dir)



if __name__ == "__main__":
    analysis_img_seq("/home/wernerfeiler/run_ml_sandvision/data_reduced")

