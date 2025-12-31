# SandVision

SandVision aims to automate the analysis of video-recordings of the sand within a rotating chamber.
Specifically, it aims to automatically detect the chamber-rotating-angle and the sand-surface-angle relatively
to the cartesian coordinate system.
This allows the calculation of the sand-slides time point and the maximal tilt angle of the sand until instability.

<br />
<img src="./example_out/G36-6400-1600-nr16_1765067248000__pred_result.png" style="width: 100%;">
<br />

## Download
```bash
# fedora --> python3.13 is recommended for the project
sudo yum install python3.13
git clone https://github.com/feiler98/SandVision

# install requirements in project directory
sudo yum install ffmpeg
sudo yum install -r requirements.txt
```
It is recommended to use the program in combination with PyCharm or as Docker application (see Dockerfile).

<br />

## Usage
- training of the model: script_train_img_seg_ml.py 
- prediction of image-sequence data: script.py

<br />

## Authorship
**W. Feiler** | construction of data processing and machine learning pipeline  <br />
**R. Lößlein** | data production and image masking for ML-training
<br />
Credits to Aladdin Persson for his UNET implementation | https://www.youtube.com/@AladdinPersson
<br />
Usage of the program is allowed with credits to the authors. 
<br />
Please read through the LICENSE for further information in regard to download, usage, and distribution of the software.