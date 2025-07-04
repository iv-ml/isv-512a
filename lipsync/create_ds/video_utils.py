import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import wraps

import cv2
from tqdm import tqdm


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal time to run {func.__name__}: {total_time:.2f} seconds")
        return result

    return wrapper


def _extract_and_save_frame(args):
    frame, output_folder, frame_number = args
    output_path = os.path.join(output_folder, str(frame_number).zfill(6) + ".png")
    cv2.imwrite(output_path, frame)


@timeit
def extract_frames(video_path, output_folder, start_number=0, step=1, end_number=None, multithread=False):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = start_number
    frames_to_process = []

    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % step == 0:
                frames_to_process.append((frame, output_folder, frame_number))

            frame_number += 1
            pbar.update(1)
            if end_number is not None and frame_number >= end_number:
                break

    cap.release()
    if not multithread:
        for i in tqdm(frames_to_process, desc="Saving Frames"):
            _extract_and_save_frame(i)
    else:
        # Use multiprocessing to save frames
        num_processes = multiprocessing.cpu_count() // 2
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            list(
                tqdm(
                    executor.map(_extract_and_save_frame, frames_to_process),
                    total=len(frames_to_process),
                    desc="Saving Frames",
                )
            )
