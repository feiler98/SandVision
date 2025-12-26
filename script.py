# imports
# ----------------------------------------------------------------------------------------------------------------------
from lib.img_seg_ml import pred_by_model
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pred_by_model("/home/wernerfeiler/muenster/SandVision/input_data/data_validate_ml_performance",
                  "/home/wernerfeiler/muenster/SandVision/ml_model/model2.pth.tar",
                  "__mask_circle_chamber")
