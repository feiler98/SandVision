# imports
# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path
from PIL import Image, ImageEnhance
import random
import numpy as np
import shutil
# ----------------------------------------------------------------------------------------------------------------------


def pillow_img_transform(dict_img: dict, out_path: (str | Path) = Path.cwd()):
    """
    Random transformation in contrast, value-range, noise, blur(opt), and crop-slice of original image + 4 rotation variants.
    Designed to increase a hand curated dataset with new instances for a more stable prediction.
    Dict keys: [img, mask_circle, mask_sand]
    Dict values: Paths to the respective files

    Parameters
    ----------
    dict_img: dict
        Path of original b/w image and both masks
    out_path = str | Path
    """
    dict_contains_keys_list = list(dict_img.keys())
    dict_contains_keys_list.sort()
    dict_must_contain_list = ["img", "mask_circle_chamber", "mask_circle_dot", "mask_sand"]
    dict_must_contain_list.sort()
    if dict_contains_keys_list != dict_must_contain_list:
        raise ValueError(f"Provided dictionary must contain {dict_must_contain_list}, got {dict_contains_keys_list} instead.")

    out_path = Path(out_path).resolve()
    if not out_path.is_dir():
        raise ValueError("Path out_path must be directory!")
    out_path.mkdir(parents=True, exist_ok=True)

    # import images
    img = Image.open(dict_img["img"])
    mask_circle_chamber = Image.open(dict_img["mask_circle_chamber"])
    mask_circle_dot = Image.open(dict_img["mask_circle_dot"])
    mask_sand = Image.open(dict_img["mask_sand"])

    # crop image by random number
    w_img, h_img = img.size

    # random float number between 0.3 and 1.0
    rand_w = int(random.uniform(0.3, 1.0)*w_img)
    rand_h = int(random.uniform(0.3, 1.0)*h_img)

    diff_w = w_img-rand_w
    diff_h = h_img-rand_h

    # crop pos
    # --------
    # x,y for top point of the image slice
    rand_left_pos = int(random.uniform(0, 1.0)*diff_w)
    rand_top_pos = int(random.uniform(0, 1.0)*diff_h)
    # x,y for bottom point of the image slice
    rand_right_pos = rand_left_pos+rand_w
    rand_bottom_pos = rand_top_pos+rand_h

    # crop imports
    img_c = img.crop((rand_left_pos, rand_top_pos, rand_right_pos, rand_bottom_pos))
    mask_circle_chamber_c = mask_circle_chamber.crop((rand_left_pos, rand_top_pos, rand_right_pos, rand_bottom_pos))
    mask_circle_dot_c = mask_circle_dot.crop((rand_left_pos, rand_top_pos, rand_right_pos, rand_bottom_pos))
    mask_sand_c = mask_sand.crop((rand_left_pos, rand_top_pos, rand_right_pos, rand_bottom_pos))

    # img_c transform by random effects
    # ---------------------------------
    # brightness | 0.5 to 2
    brightness_lvl = random.uniform(0.5, 2.0)
    img_c_b = ImageEnhance.Brightness(img_c).enhance(brightness_lvl)

    # contrast | 0.5 to 2
    contrast_lvl = random.uniform(0.5, 2.0)
    img_c_bc = ImageEnhance.Brightness(img_c_b).enhance(contrast_lvl)

    # add noise to image
    # genearte noise with same shape as that of the image
    noise_max_lvl = random.randint(30, 60)
    noise_matrix = np.random.normal(0, noise_max_lvl, list(img_c_bc.size)[::-1].append(3))
    # Add the noise to the image
    rand_noise_state = random.randint(0, 1)
    if rand_noise_state == 1:
        img_c_bcn_pre = np.asarray(img_c_bc) + noise_matrix
        noise_tag = ""
    else:
        img_c_bcn_pre = np.asarray(img_c_bc) - noise_matrix
        noise_tag = "-"
    # Clip the pixel values to be between 0 and 255.
    img_c_bcn = np.clip(img_c_bcn_pre, 0, 255).astype(np.uint8)

    # export & rotate images + masks
    for deg in range(0, 360, 90):
        Image.fromarray(img_c_bcn).rotate(deg, expand=True).save(out_path / f"{dict_img["img"].stem}__rot{deg}_c_l{rand_left_pos}t{rand_top_pos}r{rand_right_pos}b{rand_bottom_pos}_b{brightness_lvl}_c{contrast_lvl}_n{noise_tag}{noise_max_lvl}.png")
        img_c_bc.rotate(deg, expand=True).save(out_path / f"{dict_img["img"].stem}__rot{deg}_c_l{rand_left_pos}t{rand_top_pos}r{rand_right_pos}b{rand_bottom_pos}_b{brightness_lvl}_c{contrast_lvl}.png")
        img_c.rotate(deg, expand=True).save(out_path / f"{dict_img["img"].stem}__rot{deg}_c_l{rand_left_pos}t{rand_top_pos}r{rand_right_pos}b{rand_bottom_pos}.png")
        mask_circle_chamber_c.rotate(deg, expand=True).save(out_path / f"{dict_img["mask_circle_chamber"].stem}__rot{deg}_c_l{rand_left_pos}t{rand_top_pos}r{rand_right_pos}b{rand_bottom_pos}.png")
        mask_circle_dot_c.rotate(deg, expand=True).save(out_path / f"{dict_img["mask_circle_dot"].stem}__rot{deg}_c_l{rand_left_pos}t{rand_top_pos}r{rand_right_pos}b{rand_bottom_pos}.png")
        mask_sand_c.rotate(deg, expand=True).save(out_path / f"{dict_img["mask_sand"].stem}__rot{deg}_c_l{rand_left_pos}t{rand_top_pos}r{rand_right_pos}b{rand_bottom_pos}.png")


def rgb_channel_to_binary_matrix(path_mask: (str | Path), channel: str) -> Image:
    # select rgb-channel with mask
    rgb_tuple = ("r", "g", "b")
    if channel not in rgb_tuple:
        raise ValueError(f"Given channel is not valid! Expected {rgb_tuple} instead.")
    idx = rgb_tuple.index(channel)
    path_mask = Path(path_mask)

    # check paths
    if not path_mask.exists():
        raise ValueError(f"Given path {path_mask} does not exist!")
    if not path_mask.is_file():
        raise ValueError(f"Given path {path_mask} is not an image-file!")

    # mask import
    img_mask = Image.open(path_mask)
    # transform image to numpy array
    data = np.asarray(img_mask, dtype="int32")[:,:,idx]
    # function over slices of 2d numpy array
    def slice_to_binary(arr: np.array):
        data_list = np.asarray(list(map(lambda x: np.asarray(255 if x > 200 else 0), list(arr))))
        return np.asarray(data_list).astype(np.uint8)
    single_channel_arr = np.apply_along_axis(slice_to_binary, axis=0, arr=data)
    # transform back to image
    img_mask_binary = Image.fromarray(single_channel_arr)
    return img_mask_binary

def gen_stdized_ml_set(path_dir_images: (str | Path), n_transform: int = 0, file_type: str = ".png"):
    # check pathing
    path_dir_images = Path(path_dir_images)
    if not path_dir_images.exists() or not path_dir_images.is_dir():
        raise ValueError(f"Given path {path_dir_images} is not a valid directory!")
    # export path
    path_out = path_dir_images.parent / f"{path_dir_images.stem}__ml_ready"
    path_out.mkdir(parents=True, exist_ok=True)

    # get unique set of taged-files for qaulity control filtering
    set_files = list(set([p.stem.split("__")[0] for p in path_dir_images.rglob(f"*{file_type}")]))

    # get dictionary with files
    list_required_elements = ["img", "mask_circle", "mask_sand"]
    for unique_file in set_files:
        dict_path_file_group = {(p.stem.split("__")[1] if len(p.stem.split("__")) > 1 else "img"):p for p in path_dir_images.rglob(f"*{unique_file}*{file_type}")}
        # list of required keys must be identical
        len_compare_keys = len([k for k in list_required_elements if k in dict_path_file_group.keys()])
        # otherwise skip execution below
        if len_compare_keys != 3:
            print(f"File-group '{unique_file}' did not have all required elements: {list_required_elements}")
            continue
        print(f"Valid file-group >> {unique_file}")
        # if list key matches, prepare masks
        # og img
        img_out_path = path_out / dict_path_file_group["img"].name
        shutil.copy(dict_path_file_group["img"], img_out_path)
        # rgb --> b chamber
        chamber_mask_out_path = path_out / f"{dict_path_file_group["mask_circle"].stem}_chamber{file_type}"
        rgb_channel_to_binary_matrix(dict_path_file_group["mask_circle"], "b").save(chamber_mask_out_path)
        # rgb --> r dot
        dot_mask_out_path = path_out / f"{dict_path_file_group["mask_circle"].stem}_dot{file_type}"
        rgb_channel_to_binary_matrix(dict_path_file_group["mask_circle"], "r").save(dot_mask_out_path)
        # rgb --> g sand
        sand_mask_out_path = path_out / dict_path_file_group["mask_sand"].name
        rgb_channel_to_binary_matrix(dict_path_file_group["mask_sand"], "g").save(sand_mask_out_path)

        n = 0
        while n < n_transform:
            pillow_img_transform({"img": img_out_path,
                                  "mask_circle_chamber": chamber_mask_out_path,
                                  "mask_circle_dot": dot_mask_out_path,
                                  "mask_sand": sand_mask_out_path}, path_out)
            n+=1




if __name__ == "__main__":
    # rgb_channel_to_binary_matrix("/Users/werne/PycharmProjects/SandVision/input_data/data/G36-6400-1600-nr16_1765067383880__mask_circle.png", "r")
    gen_stdized_ml_set("/Users/werne/PycharmProjects/sand_vision_project/input_data/data_test", 3)






