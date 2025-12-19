# imports
# ----------------------------------------------------------------------------------------------------------------------
import shutil
from pathlib import Path
from datetime import datetime
import ffmpeg
import cv2
# ----------------------------------------------------------------------------------------------------------------------


def date_to_timestamp(dt_string: str) -> int:
    """
    Expected name convention: YYYYMMDDhhmmssfff
                              Y = Year
                              M = Month
                              D = Day
                              h = hour
                              m = minute
                              s = second
                              f = millisecond
    """
    dt_obj = datetime.strptime(dt_string, "%Y%m%d%H%M%S%f")
    return int(dt_obj.timestamp()*1000)


def date2_to_timestamp(dt_string: str) -> int:
    """
    Expected name convention: YYYY-MM-DDThh:mm:ss.ffffff
                              Y = Year
                              M = Month
                              D = Day
                              h = hour
                              m = minute
                              s = second
                              f = millisecond
    """
    dt_obj = datetime.strptime(dt_string, "%Y-%m-%dT%H:%M:%S.%f")
    return int(dt_obj.timestamp()*1000)


def rename_files_date_to_timestamp(input_folder_path: (str, Path), 
                                   output_folder_path: (str, Path), 
                                   target_tag: str = ".png", 
                                   rename: str = "img"):
    """
    Expected name convention: Pic_YYYYMMDDhhmmssfff-pic_number
                              Y = Year
                              M = Month
                              D = Day
                              h = hour
                              m = minute
                              s = second
                              f = millisecond
    
    Parameters
    ----------
    input_folder_path: Path | str
    output_folder_path: Path | str
    target_tag: str
    rename: str
    """
    
    input_folder_path = Path(input_folder_path)
    output_folder_path = Path(output_folder_path).resolve()

    if not input_folder_path.exists() or not input_folder_path.is_dir():
        raise ValueError(f"Given path {input_folder_path} is not a valid directory!")

    if not output_folder_path.exists():
        output_folder_path.mkdir(exist_ok=True, parents=True)

    list_img_paths = [p for p in input_folder_path.glob(f"*{target_tag}")]
    print(len(list_img_paths))
    for p_img in list_img_paths:
        stem_transform = p_img.stem.replace("Pic_", f"").split("-")[0]
        timestamp = date_to_timestamp(stem_transform)
        print(f"{p_img.stem} --> {timestamp}")
        shutil.copy(p_img, output_folder_path / f"{rename}_{timestamp}{p_img.suffix}")


def img_seq_to_video(img_seq_dir_path: (str, Path), 
                     video_name: str, 
                     out_path: (str, Path) = Path.cwd(), 
                     target_tag: str = ".png", 
                     fps: float = 20.0):
    """    
    Parameters
    ----------
    img_seq_dir_path: str | Path
    video_name: str
    out_path: str | Path
    target_tag: str
    fps: float
    """

    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)
        
    imgs = [p for p in Path(img_seq_dir_path).glob(f"*{target_tag}")]
    imgs.sort()  # ascending sorting by timestamp; use rename_files_date_to_timestamp prior
    frame = cv2.imread(imgs[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(out_path / video_name, fourcc, float(fps), (width,height))
    
    for img in imgs:
        video.write(cv2.imread(img))
    
    cv2.destroyAllWindows()
    video.release()


def video_to_image_seq(video_path: (str, Path),
                       img_seq_name: str,
                       out_path: (str, Path) = Path.cwd()):
    """
    Parameters
    ----------
    video_path: str | Path
    img_seq_name: str
    out_path: str | Path
    """

    # extracting essential embedded video meta data
    dict_stream = ffmpeg.probe(Path(video_path))["streams"][0]
    framerate = int(dict_stream["avg_frame_rate"].replace("/1", ""))
    frame_duration_ms = int(1000/framerate)
    if "tags" in dict_stream.keys():
        creation_time = dict_stream["tags"]["creation_time"].replace("Z", "")  # creation time in YYYY-MM-DDThh:mm:ss.ffffff
        creation_timestamp = date2_to_timestamp(creation_time)
    else:
        creation_timestamp = 0

    # new directory for images
    out_path_new_dir = Path(out_path) / img_seq_name
    out_path_new_dir.mkdir(exist_ok=True, parents=True)

    # video to image seq
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames_t_progress = creation_timestamp
    while success:
        cv2.imwrite(out_path_new_dir / f"{img_seq_name}_{frames_t_progress}.png", image)  # save frame as .png
        success, image = vidcap.read()  # bool, image
        print(f"{img_seq_name}_{frames_t_progress}.png has been created")
        frames_t_progress += frame_duration_ms


# testing
if __name__ == "__main__":
    pass


