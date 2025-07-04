import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger
from tqdm import tqdm


def get_face_analyzer():
    # Initialize InsightFace
    face_analyzer = FaceAnalysis(name="buffalo_l")
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return face_analyzer


def lms_68(frames: list[np.ndarray]) -> list[list[tuple[int, int]]]:
    store_landmarks = []
    face_analyzer = get_face_analyzer()

    for i in tqdm(range(len(frames)), desc="Fetching landmarks"):
        faces = face_analyzer.get(frames[i])
        if len(faces) == 0:
            logger.warning(f"No face detected in frame {i}")
            continue

        face = faces[0]
        landmarks = face.landmark_3d_68
        frame_data = landmarks.tolist()
        frame_data = [tuple(point[:2]) for point in frame_data]
        store_landmarks.append(frame_data)

    return store_landmarks
