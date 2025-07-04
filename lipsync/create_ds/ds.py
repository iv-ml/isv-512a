import tempfile

import cv2
import fastcore.all as fc
import numpy as np

from lipsync.create_ds.face_mask import get_face_mask_batch
from lipsync.create_ds.mediapipe_landmarks import FaceLandmarks
from lipsync.create_ds.video_utils import extract_frames


def process_landmarks(video_path, frames, landmarks_folder, landmark_weight_file):
    if not fc.Path(landmarks_folder / (video_path.stem + "_face_landmarks.npz")).exists():
        fl = FaceLandmarks(model_path=landmark_weight_file)
        face_landmarks, hw, frame_numbers = fl.process_frames(frames)
        np.savez(
            landmarks_folder / (video_path.stem + "_face_landmarks.npz"),
            face_landmarks=face_landmarks,
            hw=hw,
            frame_numbers=frame_numbers,
        )
    else:
        data = np.load(landmarks_folder / (video_path.stem + "_face_landmarks.npz"), allow_pickle=True)
        face_landmarks, hw, frame_numbers = data["face_landmarks"], data["hw"], data["frame_numbers"]
    return face_landmarks, hw, frame_numbers


def main(
    video_path,
    landmark_weight_file,
    landmarks_folder,
    save_crops_folder,
    batch_size=9,
    stride=5,
    expand_ratio=2.3,
    crop_size=512,
):
    # process all files and store in a temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        extract_frames(video_path, temp_dir, start_number=0, step=1, end_number=600)

        # process frames
        frames = fc.L(fc.Path(temp_dir).glob("*.png"))
        frames.sort()

        fc.Path(landmarks_folder).mkdir(parents=True, exist_ok=True)
        face_landmarks, hw, frame_numbers = process_landmarks(
            video_path, frames, landmarks_folder, landmark_weight_file
        )
        if len(face_landmarks) == 0:
            print("no face landmarks found")
            return

        batches, bboxes = get_face_mask_batch(
            face_landmarks, hw, frame_numbers, batch_size=batch_size, stride=stride, expand_ratio=expand_ratio
        )
        if len(bboxes) == 0:
            print("no valid face masks found")
            return
        fc.Path(save_crops_folder / video_path.stem).mkdir(parents=True, exist_ok=True)
        for batch, box in zip(batches, bboxes):
            if box[-1] == -1:
                continue
            folder_name = str(box[-2]).zfill(6)
            fc.Path(save_crops_folder / video_path.stem / f"{folder_name}").mkdir(parents=True, exist_ok=True)
            print(f"storing batch: {folder_name}")
            for frame_number in batch:
                frame_path = frames[frame_number]
                img = cv2.imread(str(frame_path))
                crop_img = img[box[1] : box[3], box[0] : box[2]]
                crop_img = cv2.resize(crop_img, (crop_size, crop_size))
                cv2.imwrite(
                    str(save_crops_folder / video_path.stem / folder_name / f"{str(frame_number).zfill(6)}.jpg"),
                    crop_img,
                )


if __name__ == "__main__":
    video_path = fc.L(fc.Path("/data/lipsync_raw_data/th_1kh_512").glob("*.mp4"))
    video_path = video_path
    for video in video_path:
        if video.stem != "UKc54igtdXI_0072_S812_E1223":
            continue
        main(video, "weights/face_landmarker.task", fc.Path("temp/landmarks"), fc.Path("temp/th_1kh_512_crops"))
