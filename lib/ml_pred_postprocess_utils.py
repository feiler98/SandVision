# imports
# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
from skimage.draw import ellipse
from numpy import ones,vstack
from numpy.linalg import lstsq
import math
from sklearn.linear_model import LinearRegression
import os
from multiprocessing import Pool

# from local lib
from .data_vis_utils import visualize_pred_img
from .general_utils import list_to_chunks

from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
# ----------------------------------------------------------------------------------------------------------------------


def find_mask_circle_center(path_mask: (str | Path)) -> dict:
    """
    Maximal Value is searched for the x and y axis of the mask --> returns coordinates

    Parameters
    ----------
    path_mask: str | Path

    Returns
    -------
    dict
        Keys: cartesian_shape, center_coordinates, xy_radius.
    """

    path_mask = Path(path_mask)
    if not path_mask.exists() and not path_mask.is_file():
        raise ValueError(f"Given path '{path_mask}' was expected to be a file!")
    array_mask = np.asarray(Image.open(path_mask))  # expects shape (width, height)
    # 0 - 1 normalization
    array_mask = array_mask / array_mask.max()
    # signal list for width (row) and height (col)
    row_signal_list = np.sum(array_mask,axis=0).tolist()
    col_signal_list = np.sum(array_mask,axis=1).tolist()
    # calc coordinates of circle center
    max_row = max(row_signal_list)
    max_col = max(col_signal_list)
    half_count_row_max = int(row_signal_list.count(max_row) / 2) - 1
    half_count_col_max = int(col_signal_list.count(max_col) / 2) - 1
    coordinate_x = row_signal_list.index(max_row) + half_count_row_max
    coordinate_y = col_signal_list.index(max_col) + half_count_col_max
    rad_x = int(len([x for x in row_signal_list if x > 0]) / 2)
    rad_y = int(len([x for x in col_signal_list if x > 0]) / 2)
    return {"cartesian_shape": array_mask.shape,
            "center_coordinates": (coordinate_x, coordinate_y),
            "xy_radius": (rad_x, rad_y)}


def create_circle(cartesian_shape: tuple,
                  center_coordinates: tuple,
                  xy_radius: tuple) -> np.array:
    """
    Creates a mask based on the __mask_circle_chamber prediction, decreases size
    for trimming of the __mask_sand prediction.
    Input find_mask_circle_center output as **kwargs.

    Parameters
    ----------
    cartesian_shape: tuple
    center_coordinates: tuple
    xy_radius: tuple

    Returns
    -------
    np.array
        White circle on black canvas (0-1 normalized) as numpy-array.
    """

    x_radius = int(xy_radius[0]*0.75)  # remove teething for better prediction
    y_radius = int(xy_radius[1]*0.75)
    nx = cartesian_shape[0]  # number of pixels in x-dir
    ny = cartesian_shape[1]  # number of pixels in y-dir

    # Create an array of int32 zeros
    black_canvas = np.zeros((nx, ny), dtype=np.int32)
    rr, cc = ellipse(center_coordinates[1], center_coordinates[0], y_radius, x_radius, cartesian_shape)
    black_canvas[rr, cc] = 1
    return black_canvas


def calc_line_by_two_points(xy_point1: tuple,
                            xy_point2: tuple) -> tuple:
    """
    Transforms two points into a linear-function for angle calculation downstream.

    Parameters
    ----------
    xy_point1: tuple
        (x-coordinate, y-coordinate)
    xy_point2: tuple
        (x-coordinate, y-coordinate)

    Returns
    -------
        Slope m & intersection t for equation y = mx + t.
    """

    if xy_point1[0] == xy_point2[0]:
        xy_point1 = (xy_point1[0]+0.0001, xy_point1[1])
    x_coords, y_coords = zip(*[xy_point1, xy_point2])
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, t = lstsq(A, y_coords)[0]
    print(f"Linear equation | y = {m}x + {t}")
    return round(m, 2), round(t, 2)


def calc_two_line_angle(m_line1: (int | float), m_line2: (int | float)) -> float:
    """
    Angle calculation between two lines.

    Parameters
    ----------
    m_line1: int | float
    m_line2: int | float

    Returns
    -------
    float
        Angle between the lines.
    """

    return round(math.atan(abs((m_line1 - m_line2) / (1 + m_line1 * m_line2))), 2)


def sand_mask_get_outline_mtx(path_mask: (str | Path)) -> np.array:
    """
    Generates outline around shape --> extract coordinates of each pixel post smaller-circle overlay.

    Parameters
    ----------
    path_mask: str | Path

    Returns
    -------
    np.array
    """

    img_sand_mask = Image.open(path_mask)
    img_outline = img_sand_mask.filter(ImageFilter.FIND_EDGES)
    arr_outline = np.asarray(img_outline)
    # normalize output
    return arr_outline / arr_outline.max()


def lin_reg_sand(arr: np.array) -> tuple:
    """
    Scikit-Learn linear-regression model for point extraction.

    Parameters
    ----------
    arr: np.array

    Returns
    -------
    tuple
        Point1 & point2 are returned for linear function calculation.
    """

    y, X = list(np.where(arr == 1))
    # too many points clustered decrease linear-reg performance --> thin out by creating steps
    list_idx = list(range(0, len(y), 30))
    y = y[list_idx]
    X = X[list_idx]
    model_lin_reg = LinearRegression(n_jobs=-1)
    model_lin_reg.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    X_pred = np.array([X.min(), X.max()]).reshape(-1, 1)
    y_pred = model_lin_reg.predict(X_pred)

    # plt.scatter(X, y)  # visualize for testing purposes
    # plt.show()

    point1, point2 = [(float(x), float(y)) for x, y in zip(X_pred, y_pred)]
    return point1, point2


def get_line_params_from_mask_pred(path_mask_chamber: (str | Path),
                                   path_mask_dot: (str | Path),
                                   path_mask_sand: (str | Path)) -> dict:
    """
    Parameters
    ----------
    path_mask_chamber: str | Path
    path_mask_dot: str | Path
    path_mask_sand: str | Path

    Returns
    -------
    dict
        Coordinates & linear-functions extracted from mask-element prediction of the sand chamber.
    """

    # pathing
    path_mask_chamber = Path(path_mask_chamber)
    path_mask_dot = Path(path_mask_dot)
    path_mask_sand = Path(path_mask_sand)

    # center of circle masks
    dict_center_chamber = find_mask_circle_center(path_mask_chamber)
    dict_center_dot = find_mask_circle_center(path_mask_dot)

    # get linear regression points of sand mask
    arr_circle = create_circle(**dict_center_chamber)
    arr_sand_outline = sand_mask_get_outline_mtx(path_mask_sand)
    arr_masked_sand_outline = arr_circle*arr_sand_outline
    point1_sand, point2_sand = lin_reg_sand(arr_masked_sand_outline)
    print(point1_sand, point2_sand)
    # Image.fromarray(arr_masked_sand_outline*255).show()  # validation of masked sand-border estimation

    # calculate y = mx + t linear equation parameters
    x_chamber, y_chamber = dict_center_chamber["center_coordinates"]
    y_chamber = dict_center_chamber["cartesian_shape"][1] - y_chamber
    x_dot, y_dot = dict_center_dot["center_coordinates"]
    y_dot = dict_center_dot["cartesian_shape"][1] - y_dot
    m_circle, t_circle = calc_line_by_two_points((x_chamber, y_chamber), (x_dot, y_dot))
    x_sand1, y_sand1 = point1_sand
    x_sand2, y_sand2 = point2_sand
    m_sand, t_sand = calc_line_by_two_points((x_sand1, y_sand1), (x_sand2, y_sand2))

    return {"chamber_center_coords": dict_center_chamber["center_coordinates"],
            "chamber_dot_coords": dict_center_dot["center_coordinates"],
            "sand_coords1": (int(x_sand1), int(y_sand1)),
            "sand_coords2": (int(x_sand2), int(y_sand2)),
            "circle_line__m": float(m_circle),
            "circle_line__t": float(t_circle),
            "sand_line__m": float(m_sand),
            "sand_line__t": float(t_sand)}


def mask_result_eval(path_ml_out: (str | Path)):
    """
    Multiprocessing function for extracting all image prediction images.
    Exports a csv file with all results.

    Parameters
    ----------
    path_ml_out: str | Path
        Output folder with all image-mask predictions. 3 masks per image must be generated as a prerequisite.
    """

    path_ml_out = Path(path_ml_out)

    # get image path
    set_img_tags = list(set([p.stem.split("__")[0] for p in path_ml_out.rglob("*.png")]))
    n_cores = os.cpu_count()
    with Pool(n_cores) as pool:
        result_list = pool.starmap(mp_data_generation, [(path_ml_out, chunk_img_tags) for chunk_img_tags in list_to_chunks(set_img_tags, n_cores)])
    dict_unite = {}
    for result_dict in result_list:
        dict_unite.update(result_dict)
    df_result = pd.DataFrame.from_dict(dict_unite).T
    df_result.to_csv(path_ml_out / f"{path_ml_out.name}__eval.csv")


# func for mask_result_eval multiprocessing element
def mp_data_generation(path_out: (str | Path),
                       list_img_tags: list,
                       file_type=".png",
                       visualize_pred: bool = True) -> dict:
    """
    Sub-function for the multiprocessing mask_result_eval function.

    Parameters
    ----------
    path_out: str | Path
    list_img_tags: list
    file_type: str
    visualize_pred: bool

    Returns
    -------
    dict
        Dictionary from get_line_params_from_mask_pred function.
    """

    mask_tags = ["mask_circle_chamber", "mask_circle_dot", "mask_sand", "img"]
    dict_results_collect = {}
    for unique_file in list_img_tags:
        dict_path_file_group = {(p.stem.split("__")[1] if len(p.stem.split("__")) > 1 else "img"): p for p in path_out.rglob(f"*{unique_file}*{file_type}")}
        if not mask_tags.sort() == list(dict_path_file_group.keys()).sort():
            print(f"Image sequence '{unique_file}' does not have all necessary image-elements. --> skip")
            continue
        header_string = f"Evaluation of '{unique_file}'"
        print(f"""
{header_string}""")
        print("-"*len(header_string))
        dict_pred = get_line_params_from_mask_pred(dict_path_file_group["mask_circle_chamber"],
                                                   dict_path_file_group["mask_circle_dot"],
                                                   dict_path_file_group["mask_sand"])
        if visualize_pred:
            visualize_pred_img(dict_path_file_group["img"], dict_pred)
        dict_results_collect.update({unique_file: dict_pred})
    return dict_results_collect


# debugging
if __name__ == "__main__":
    path_data = Path("/Users/werne/PycharmProjects/sand_vision_project/input_data/data_test__ml_ready")
    dict_out = get_line_params_from_mask_pred(path_data/"G36-6400-1600-nr16_1765067248000__mask_circle_chamber.png",
                                         path_data/"G36-6400-1600-nr16_1765067248000__mask_circle_dot.png",
                                         path_data/"G36-6400-1600-nr16_1765067248000__mask_sand.png")
    print(dict_out)
    visualize_pred_img("/Users/werne/PycharmProjects/sand_vision_project/input_data/data_test__ml_ready/G36-6400-1600-nr16_1765067248000.png", dict_out)


