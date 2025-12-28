# imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

# local imports --> lib
from lib.img_seg_ml import VisTransformer
from lib.general_utils import (load_checkpoint,
                               save_checkpoint,
                               get_loaders,
                               check_accuracy,
                               save_predictions_as_imgs,
                               ml_ready_data_cv)
# ----------------------------------------------------------------------------------------------------------------------


###################
# HYPERPARAMETERS #
###################

SEED = None
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
NUM_EPOCHS = 20
NUM_WORKERS = 10
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR_PATH = Path("/home/wernerfeiler/muenster/SandVision/input_data/data__ml_ready")
MASK_TAG = "__mask_sand"  # __mask_sand, __mask_circle_chamber, __mask_circle_dot


#####################
# ML TRAIN PIPELINE #
#####################

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # forward
        with torch.amp.autocast("cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


# ----------------------------------------------------------------------------------------------------------------------
# model training
# ==============

def train_ml():

    print("""
###########################
# VisTransformer training #
###########################
""")
    print(f"""> model for {MASK_TAG} prediction <
""")
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    # cross validation
    dict_train_eval = ml_ready_data_cv(IMG_DIR_PATH)
    for split_id, dict_train_eval_split in dict_train_eval.items():
        # optional seed setting
        if isinstance(SEED, int):
            torch.manual_seed(abs(SEED))
            print(f"Manual seed set to {abs(SEED)}")

        # model settings
        model = VisTransformer(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("")
        print_cv = f"| Cross validation set number #{split_id} |"
        print("-" * len(print_cv))
        print(print_cv)
        print("-"*len(print_cv))

        # path of training-results & performance
        path_results = IMG_DIR_PATH.parent / f"pred_out__split_{split_id}{MASK_TAG}"

        train_loader, val_loader, list_val_tags = get_loaders(IMG_DIR_PATH,
                                                              MASK_TAG,
                                                              dict_train_eval_split["train_set"],
                                                              dict_train_eval_split["val_set"],
                                                              train_transform,
                                                              val_transform,
                                                              BATCH_SIZE,
                                                              NUM_WORKERS,
                                                              PIN_MEMORY)
        if LOAD_MODEL:
            load_checkpoint(torch.load(f"model{MASK_TAG}.pth.tar"), model)

        scaler = torch.amp.GradScaler("cuda")

        # check accuracy
        accuracy_params_dict = check_accuracy(val_loader, model, device=DEVICE)
        dict_metrics = {"init": accuracy_params_dict}
        for epoch in range(NUM_EPOCHS):
            train_fn(train_loader,
                     model,
                     optimizer,
                     loss_fn,
                     scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, f"model{MASK_TAG}.pth.tar")

            # check accuracy
            accuracy_params_dict = check_accuracy(val_loader, model, device=DEVICE)
            dict_metrics.update({f"epoch {epoch+1}": accuracy_params_dict})

            # print some examples to a folder
            save_predictions_as_imgs(val_loader,
                                     list_val_tags,
                                     MASK_TAG,
                                     model,
                                     path_results,
                                     device=DEVICE)
        df_result = pd.DataFrame.from_dict(dict_metrics).T
        df_result.to_csv(path_results / f"ml_metrics__split_{split_id}{MASK_TAG}.csv")
# ----------------------------------------------------------------------------------------------------------------------


################
# RUN TRAINING #
################

if __name__ == "__main__":
    train_ml()
