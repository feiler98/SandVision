# imports
# ----------------------------------------------------------------------------------------------------------------------
import random
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader

# local lib import
from .img_seg_ml import SandDataLoader
# ----------------------------------------------------------------------------------------------------------------------


def list_to_chunks(input_list: list, n_chunks: int) -> list:
    """
    Multiprocessing list to list-chunks.

    Parameters
    ----------
    input_list: list
    n_chunks: int

    Returns
    -------
    list
        A list of lists as generator for memory-saving.
    """
    n_chunks = n_chunks if n_chunks <= len(input_list) else len(input_list)
    step_size = int(len(input_list)/n_chunks-1)
    for i in range(0, len(input_list), step_size):
        yield input_list[i:i+step_size]


# utility for handling model training
# ----------------------------------------------------------------------------------------------------------------------

# saving path for model
path_checkpoint = Path(__file__).parent.parent / "ml_model"


def save_checkpoint(state: torch,
                    filename: str = "model.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, path_checkpoint/filename)


def load_checkpoint(checkpoint: dict,
                    model: torch.nn.Module):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(data_dir: (str | Path),
                mask_tag: str,
                train_tag_list: list,
                val_tag_list: list,
                train_transform,
                val_transform,
                batch_size: int,
                n_jobs: int = 4,
                pin_memory: bool = True):
    """
    Shuffel true for the training-set, not the validation-set --> no effect on performance of the model.
    """

    train_ds = SandDataLoader(
        img_dir=data_dir,
        mask_tag=mask_tag,
        img_tags=train_tag_list,
        transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=n_jobs,
        pin_memory=pin_memory,
        shuffle=True)

    val_ds = SandDataLoader(
        img_dir=data_dir,
        mask_tag=mask_tag,
        img_tags=val_tag_list,
        transform=val_transform)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=n_jobs,
        pin_memory=pin_memory,
        shuffle=False)

    return train_loader, val_loader, val_tag_list


def check_accuracy(loader: tuple,
                   model: torch.nn.Module,
                   device: str = "cuda"):

    # variables for evaluation
    num_correct_pixel = 0
    num_pixels = 0
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct_pixel += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    # final metrics
    accuracy = round(float(num_correct_pixel/num_pixels*100), 2)
    avg_dice_score = round(float(dice_score/len(loader)), 2)

    # evaluation output
    print(f"{num_correct_pixel}/{num_pixels} correctly predicted pixels | Accuracy: {accuracy} %")
    print(f"Dice score: {avg_dice_score}")
    model.train()
    return dict(accuracy=accuracy,
                avg_dice_score=avg_dice_score,
                num_correct_pixel=int(num_correct_pixel),
                num_pixels=int(num_pixels))


def save_predictions_as_imgs(loader: tuple,
                             loader_tag_list: list,
                             mask_tag: str,
                             model: torch.nn.Module,
                             path_out: (str | Path),
                             device: str = "cuda"):

    path_out = Path(path_out)
    if not path_out.exists():
        path_out.mkdir(exist_ok=True, parents=True)
    model.eval()
    for tag, (x, y) in zip(loader_tag_list, loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, path_out / f"{tag}{mask_tag}.png")
    model.train()


def ml_ready_data_cv(img_dir: (str, Path), data_suffix: str = ".png", n_split: int = 3) -> dict:
    list_unique_files = list(set([p.stem.split("__")[0] for p in Path(img_dir).rglob(f"*{data_suffix}")]))
    random.shuffle(list_unique_files)
    slices = list(list_to_chunks(list_unique_files, n_split))
    dict_cv = {}
    for i in range(0, len(slices)):
        slice_copy = slices.copy()
        list_train = []
        list_val = slice_copy.pop(0)
        [list_train.extend(sub_slice) for sub_slice in slice_copy]
        dict_cv.update({i+1: {"train_set": list_train, "val_set": list_val}})
    return dict_cv


