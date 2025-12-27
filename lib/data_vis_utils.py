# imports
# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------


def visualize_pred_img(path_img: (str | Path), dict_results: dict):
    """
    Visual conformation of element prediction by mask on the original image.

    Parameters
    ----------
    path_img: str | Path
    dict_results: dict
        Dictionary from get_line_params_from_mask_pred function.
    """

    path_img = Path(path_img)
    img_arr = np.asarray(Image.open(path_img))
    x_sand = [dict_results["sand_coords1"][0], dict_results["sand_coords2"][0]]
    y_sand = [dict_results["sand_coords1"][1], dict_results["sand_coords2"][1]]
    x_chamber = [dict_results["chamber_center_coords"][0], dict_results["chamber_dot_coords"][0]]
    y_chamber = [dict_results["chamber_center_coords"][1], dict_results["chamber_dot_coords"][1]]
    # plotting
    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    ax.plot(x_sand, y_sand, color="#93ed95", linewidth=2.5, label=f"sand axis | y = {dict_results["sand_line__m"]}x + {dict_results["sand_line__t"]}")
    ax.plot(x_chamber, y_chamber, color="#ff6054", marker="D", linewidth=2.0, label=f"chamber axis | y = {dict_results["circle_line__m"]}x + {dict_results["circle_line__t"]}")
    ax.legend()
    plt.title(f"Prediction | {path_img.stem}", fontsize=12, fontweight="medium")
    plt.savefig(path_img.parent / f"{path_img.stem}__pred_result.png", bbox_inches="tight", dpi=150)



