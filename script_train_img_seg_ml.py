# imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 10
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR_PATH = Path("/home/wernerfeiler/muenster/SandVision/input_data/data__ml_ready")
MASK_TAG = "__mask_circle_chamber"  # __mask_sand, __mask_circle_chamber, __mask_circle_dot


#####################
# ML TRAIN PIPELINE #
#####################

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
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
# ----------------------------------------------------------------------------------------------------------------------
def train_ml():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    # model settings
    model = VisTransformer(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dict_train_eval = ml_ready_data_cv(IMG_DIR_PATH)[1]

    train_loader, val_loader, list_val_tags = get_loaders(IMG_DIR_PATH,
                                                          MASK_TAG,
                                                          dict_train_eval["train_set"],
                                                          dict_train_eval["val_set"],
                                                          train_transform,
                                                          val_transform,
                                                          BATCH_SIZE,
                                                          NUM_WORKERS,
                                                          PIN_MEMORY)

    accuracy_params = check_accuracy(val_loader, model, device=DEVICE)
    print(accuracy_params)
    scaler = torch.cuda.amp.GradScaler()

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
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader,
                                 list_val_tags,
                                 MASK_TAG,
                                 model,
                                 IMG_DIR_PATH.parent / f"pred_out{MASK_TAG}",
                                 device=DEVICE)
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    train_ml()
