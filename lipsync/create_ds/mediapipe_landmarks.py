import mediapipe as mp
import numpy as np
from tqdm import tqdm


class FaceLandmarks:
    def __init__(self, model_path):
        self.model_path = model_path
        BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        # Create a face landmarker instance with the video mode:
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
        )

    def __call__(self, img_loc):
        with self.FaceLandmarker.create_from_options(self.options) as landmarker:
            image = mp.Image.create_from_file(str(img_loc))
            height, width = image.height, image.width
            face_landmarker_result = landmarker.detect(image)
            face_landmark = self.compute_face_landmarks(face_landmarker_result, height, width)
        return np.array(face_landmark), height, width

    def compute_face_landmarks(self, detection_result, h, w):
        """
        Compute face landmarks from a detection result.

        Args:
            detection_result (mediapipe.solutions.face_mesh.FaceMesh): The detection result containing face landmarks.
            h (int): The height of the video frame.
            w (int): The width of the video frame.

        Returns:
            face_landmarks_list (list): A list of face landmarks.
        """
        face_landmarks_list = detection_result.face_landmarks
        if len(face_landmarks_list) != 1:
            print("#face is invalid:", len(face_landmarks_list))
            return []
        return [[p.x * w, p.y * h] for p in face_landmarks_list[0]]

    def process_frames(self, frames):
        frames.sort()
        face_landmarks = []
        hw = []
        frame_numbers = []
        for n, frame in tqdm(enumerate(frames)):
            landmarks, h, w = self(frame)
            if len(landmarks) == 0:
                print(f"no face in frame {n}")
                continue
            face_landmarks.append(landmarks)
            hw.append([h, w])
            frame_numbers.append(n)
        if len(face_landmarks) == 0:
            return [], [], []
        return np.stack(face_landmarks), np.stack(hw), np.array(frame_numbers)


# def main(folder_path, save_dir, model_path):
#     #use multiprocessing to process videos in parallel
#     folder = fc.Path(folder_path)
#     fl = FaceLandmarks(model_path=model_path)
#     def _process_video(video_path):
#         frames = fc.L(fc.Path(f"temp/ds/{video_path.stem}").glob("*.jpg"))
#         frames.sort()
#         face_landmarks, hw, frame_numbers = fl.process_frames(frames)
#         np.savez(f"{save_dir}/{video_path.stem}/face_landmarks.npz", face_landmarks=face_landmarks, hw=hw, frame_numbers=frame_numbers)

#     videos = fc.L(folder.glob("*.mp4"))
#     #using multiprocessing to process videos in parallel
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         results = pool.map(_process_video, videos)

if __name__ == "__main__":
    import fastcore.all as fc

    video_path = fc.Path("/data/lipsync_raw_data/hdtf/rB4HJDtUI04.mp4")
    fl = FaceLandmarks(model_path="weights/face_landmarker.task")
    frames = fc.L(fc.Path(f"temp/ds/{video_path.stem}").glob("*.jpg"))
    frames.sort()
    face_landmarks, hw, frame_numbers = fl.process_frames(frames)
    if face_landmarks is None:
        print("no face landmarks found")
        exit()
    np.savez(
        f"temp/ds/{video_path.stem}/face_landmarks.npz",
        face_landmarks=face_landmarks,
        hw=hw,
        frame_numbers=frame_numbers,
    )
