import csv
import math
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from PIL import Image
from tqdm import tqdm

from live_portrait.modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer
from live_portrait.modules.utils.constants import ModelType, SamplePart

multiprocessing.set_start_method("spawn", force=True)

app = typer.Typer()


@dataclass
class ExpressionRanges:
    """Defines ranges for different facial expression parameters"""

    aaa: Tuple[float, float] = (-15.0, 15.0)  # Mouth shapes
    eee: Tuple[float, float] = (-15.0, 15.0)
    woo: Tuple[float, float] = (-15.0, 15.0)
    smile: Tuple[float, float] = (-1.3, 1.3)
    source_ratio: Tuple[float, float] = (0.0, 1.0)
    crop_factor: Tuple[float, float] = (1.5, 2.5)


def generate_random_expression() -> list:
    """Generate random values for facial expressions within defined ranges"""
    ranges = ExpressionRanges()

    params = [
        ModelType.HUMAN.value,
        0,  # Fixed values for rotation and other parameters
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        np.random.uniform(*ranges.aaa),
        np.random.uniform(*ranges.eee),
        np.random.uniform(*ranges.woo),
        np.random.uniform(*ranges.smile),
        np.random.uniform(*ranges.source_ratio),
        0.0,
        SamplePart.ALL.value,
        np.random.uniform(*ranges.crop_factor),
    ]

    return params


def process_image_with_random_expression(
    inferencer: LivePortraitInferencer,
    source_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single image with randomly generated expressions"""
    params = generate_random_expression()
    params.extend([source_image, None, None])
    processed_image = inferencer.edit_expression(*params)
    return processed_image[0]


def setup_logger(log_file: Path) -> None:
    """Configure loguru logger with file output."""
    logger.add(log_file, rotation="100 MB")


def create_error_log(gpu_id: int) -> Path:
    """Create error log CSV file in temp directory."""
    temp_dir = Path("/tmp/live_portrait_errors_gpu{gpu_id}")
    temp_dir.mkdir(exist_ok=True)

    error_file = temp_dir / f"errors_gpu{gpu_id}.csv"

    with open(error_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_id", "error_message"])

    return error_file


def get_png_files(input_file: Path, datasets: list[str]) -> list[tuple[Path, str, str, str]]:
    """Read image paths and metadata from CSV file"""
    # Read CSV file
    df = pd.read_csv(input_file)

    # Filter by datasets if specified
    if datasets:
        df = df[df["dataset"].isin(datasets)]

    # Convert to list of tuples: (image_path, dataset, clip_id, bin_id)
    # bin_id is string of length 6
    results = [(Path(row["img_path"]), row["dataset"], row["clip_id"], str(row["bin_id"]).zfill(6)) for _, row in df.iterrows()]

    return results


def process_batch(
    batch: list[tuple[Path, str, str, str]],
    output_dir: Path,
    error_file: Path,
    success_file: Path,
    device_id: int,
) -> None:
    """Process a batch of images using LivePortrait"""
    try:
        inferencer = LivePortraitInferencer(gpu_id=device_id)

        for png_file, dataset, clip_id, bin_id in batch:
            try:
                current_output_dir = output_dir / str(clip_id) / str(bin_id)
                current_output_dir.mkdir(parents=True, exist_ok=True)
                output_file = current_output_dir / png_file.name

                source_image = np.array(Image.open(png_file))
                processed_image = process_image_with_random_expression(inferencer, source_image)

                if not isinstance(processed_image, np.ndarray):
                    processed_image = processed_image.cpu().numpy()
                processed_image = processed_image.astype(np.uint8)

                Image.fromarray(processed_image).save(output_file)

                # Log successful processing
                with open(success_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(png_file), str(output_file), datetime.now().isoformat()])

            except Exception as e:
                error_message = f"Error processing {png_file}: {str(e)}"
                logger.error(error_message)
                with open(error_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(png_file), error_message])

                if "Failed to detect face!!" in str(e):
                    # copy the image file to output_dir
                    output_file = output_dir / str(clip_id) / str(bin_id) / png_file.name
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(png_file, output_file)

    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")

def parse_dataset(value: str) -> list[str]:
    """Parse comma-separated dataset names into a list"""
    return [item.strip() for item in value.split(",") if item.strip()]

def create_log_files(gpu_id: int) -> Tuple[Path, Path]:
    """Create error and success log CSV files in temp directory."""
    temp_dir = Path("/tmp/live_portrait_logs_gpus")
    temp_dir.mkdir(exist_ok=True)

    error_file = temp_dir / f"errors_gpu{gpu_id}.csv"
    success_file = temp_dir / f"success_gpu{gpu_id}.csv"

    # Initialize error log
    with open(error_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_id", "error_message"])

    # Initialize success log
    with open(success_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_id", "output_path", "timestamp"])

    return error_file, success_file

@app.command()
def process_images(
    input_file: Path = typer.Argument(
        "/data/prakash_lipsync/video_hallo_512/partition_0.csv",
        help="Input CSV file containing image paths and metadata",
    ),
    output_dir: Path = typer.Argument(
        "/data/prakash_lipsync/video_hallo_512_lp", help="Output directory for processed images"
    ),
    datasets: str = typer.Option(
        "hdtf", help="Dataset names as comma-separated values (e.g., hdtf,iv_recording)", callback=parse_dataset
    ),
    device_id: int = typer.Option(0, help="GPU device ID to run inference on"),
    fast: bool = typer.Option(False, help="Process only one clip for quick testing"),
    batch_size: int = typer.Option(32, help="Number of images to process in each batch"),
    num_workers: int = typer.Option(4, help="Number of parallel processes"),
) -> None:
    """Process images with LivePortrait, applying random expressions to each image"""
    # Setup logging and error tracking
    log_file = output_dir / "processing.log"
    error_file, success_file = create_log_files(device_id)  # Updated function call
    setup_logger(log_file)

    logger.info(f"Processing images from {input_file} to {output_dir}")
    logger.info(f"Selected datasets: {datasets}")
    logger.info(f"Fast mode: {fast}")
    logger.info(f"Using device: cuda:{device_id}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of workers: {num_workers}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of files to process
    png_files = list(get_png_files(input_file, datasets))

    if fast:
        # In fast mode, process only 10 images
        png_files = png_files[:10]

    # Create batches
    num_files = len(png_files)
    num_batches = math.ceil(num_files / batch_size)
    batches = [png_files[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]

    logger.info(f"Processing {num_files} files in {num_batches} batches")

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_batch_partial = partial(
            process_batch,
            output_dir=output_dir,
            error_file=error_file,
            success_file=success_file,
            device_id=device_id,
        )
        list(tqdm(executor.map(process_batch_partial, batches), total=len(batches), desc="Processing batches"))

    logger.info("Processing completed")


if __name__ == "__main__":
    app()
