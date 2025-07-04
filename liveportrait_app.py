from pathlib import Path
import typer
from loguru import logger
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from dataclasses import dataclass
from typing import Tuple
import json
from datetime import datetime

from live_portrait.modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer
from live_portrait.modules.utils.paths import MODELS_DIR, OUTPUTS_DIR
from live_portrait.modules.utils.constants import ModelType, SamplePart


@dataclass
class ExpressionRanges:
    """Defines ranges for different facial expression parameters"""

    rotate_pitch: Tuple[float, float] = (-20.0, 20.0)
    rotate_yaw: Tuple[float, float] = (-20.0, 20.0)
    rotate_roll: Tuple[float, float] = (-20.0, 20.0)
    blink: Tuple[float, float] = (-20.0, 20.0)
    eyebrow: Tuple[float, float] = (-40.0, 40.0)
    wink: Tuple[float, float] = (0.0, 25.0)  # Changed from 0 to 0.0
    pupil_x: Tuple[float, float] = (-20.0, 20.0)
    pupil_y: Tuple[float, float] = (-20.0, 20.0)
    aaa: Tuple[float, float] = (-30.0, 30.0)
    eee: Tuple[float, float] = (-20.0, 20.0)
    woo: Tuple[float, float] = (-20.0, 20.0)
    smile: Tuple[float, float] = (-2.0, 2.0)  # Changed to float
    source_ratio: Tuple[float, float] = (0.0, 1.0)
    crop_factor: Tuple[float, float] = (1.5, 2.5)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to RGB format"""
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:  # RGB
        return image
    else:
        raise ValueError(f"Unexpected image format with shape {image.shape}")


def generate_expression_params(ranges: ExpressionRanges, use_sliders: bool = True) -> tuple[list, dict]:
    """Generate expression parameters either randomly or from sliders and return both list and dict formats"""
    param_names = [
        "model_type",
        "rotate_pitch",
        "rotate_yaw",
        "rotate_roll",
        "blink",
        "eyebrow",
        "wink",
        "pupil_x",
        "pupil_y",
        "aaa",
        "eee",
        "woo",
        "smile",
        "source_ratio",
        "fixed_value",
        "sample_part",
        "crop_factor",
    ]

    if use_sliders:
        with st.sidebar:
            st.subheader("Expression Parameters")
            values = [
                ModelType.HUMAN.value,
                st.slider(
                    "Rotate Pitch",
                    min_value=float(ranges.rotate_pitch[0]),
                    max_value=float(ranges.rotate_pitch[1]),
                    value=0.0,
                    step=0.1,
                ),
                st.slider(
                    "Rotate Yaw",
                    min_value=float(ranges.rotate_yaw[0]),
                    max_value=float(ranges.rotate_yaw[1]),
                    value=0.0,
                    step=0.1,
                ),
                st.slider(
                    "Rotate Roll",
                    min_value=float(ranges.rotate_roll[0]),
                    max_value=float(ranges.rotate_roll[1]),
                    value=0.0,
                    step=0.1,
                ),
                st.slider(
                    "Blink", min_value=float(ranges.blink[0]), max_value=float(ranges.blink[1]), value=0.0, step=0.1
                ),
                st.slider(
                    "Eyebrow",
                    min_value=float(ranges.eyebrow[0]),
                    max_value=float(ranges.eyebrow[1]),
                    value=0.0,
                    step=0.1,
                ),
                st.slider(
                    "Wink", min_value=float(ranges.wink[0]), max_value=float(ranges.wink[1]), value=0.0, step=0.1
                ),
                st.slider(
                    "Pupil X",
                    min_value=float(ranges.pupil_x[0]),
                    max_value=float(ranges.pupil_x[1]),
                    value=0.0,
                    step=0.1,
                ),
                st.slider(
                    "Pupil Y",
                    min_value=float(ranges.pupil_y[0]),
                    max_value=float(ranges.pupil_y[1]),
                    value=0.0,
                    step=0.1,
                ),
                st.slider("AAA", min_value=float(ranges.aaa[0]), max_value=float(ranges.aaa[1]), value=0.0, step=0.1),
                st.slider("EEE", min_value=float(ranges.eee[0]), max_value=float(ranges.eee[1]), value=0.0, step=0.1),
                st.slider("WOO", min_value=float(ranges.woo[0]), max_value=float(ranges.woo[1]), value=0.0, step=0.1),
                st.slider(
                    "Smile", min_value=float(ranges.smile[0]), max_value=float(ranges.smile[1]), value=0.0, step=0.1
                ),
                st.slider(
                    "Source Ratio",
                    min_value=float(ranges.source_ratio[0]),
                    max_value=float(ranges.source_ratio[1]),
                    value=1.0,
                    step=0.01,
                ),
                0.0,
                SamplePart.ALL.value,
                st.slider(
                    "Crop Factor",
                    min_value=float(ranges.crop_factor[0]),
                    max_value=float(ranges.crop_factor[1]),
                    value=1.7,
                    step=0.1,
                ),
            ]
    else:
        values = [
            ModelType.HUMAN.value,
            np.random.uniform(*ranges.rotate_pitch),
            np.random.uniform(*ranges.rotate_yaw),
            np.random.uniform(*ranges.rotate_roll),
            np.random.uniform(*ranges.blink),
            np.random.uniform(*ranges.eyebrow),
            np.random.uniform(*ranges.wink),
            np.random.uniform(*ranges.pupil_x),
            np.random.uniform(*ranges.pupil_y),
            np.random.uniform(*ranges.aaa),
            np.random.uniform(*ranges.eee),
            np.random.uniform(*ranges.woo),
            np.random.uniform(*ranges.smile),
            np.random.uniform(*ranges.source_ratio),
            0.0,
            SamplePart.ALL.value,
            np.random.uniform(*ranges.crop_factor),
        ]

    params_dict = dict(zip(param_names, values))
    return values, params_dict


OUTPUTS_DIR = Path("outputs")  # Define a default output directory if not imported correctly


def save_params_json(params_dict: dict, output_dir: Path | None = None) -> Path:
    """Save parameters to a JSON file"""
    try:
        # Use default output directory if none provided
        if output_dir is None:
            output_dir = OUTPUTS_DIR / "params"
        else:
            output_dir = Path(output_dir)

        # Ensure directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"params_{timestamp}.json"

        # Save parameters
        with open(output_path, "w") as f:
            json.dump(params_dict, f, indent=4)

        logger.info(f"Parameters saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving parameters: {str(e)}")
        st.error(f"Failed to save parameters: {str(e)}")
        return None


def process_image(
    inferencer: LivePortraitInferencer,
    source_image: np.ndarray,
    params: list,
) -> np.ndarray:
    """Process image with given expression parameters"""
    try:
        # Convert image to RGB format
        source_image = convert_to_rgb(source_image)

        params = params + [source_image, None, None]
        processed_image, _, _ = inferencer.edit_expression(*params)
        return processed_image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"Failed to process image: {str(e)}")
        return None


def main():
    try:
        st.title("Live Portrait Expression Editor")

        # Initialize the inferencer
        @st.cache_resource
        def load_inferencer():
            return LivePortraitInferencer()

        inferencer = load_inferencer()

        # Move controls to sidebar
        with st.sidebar:
            st.header("Controls")
            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

            # Control mode
            mode = st.radio("Control Mode", ["Manual Sliders", "Random Generation"])

            # Only show generate button for random mode
            generate_button = st.button("Generate Expression") if mode == "Random Generation" else False

        if uploaded_file is not None:
            try:
                # Convert uploaded file to image
                image = Image.open(uploaded_file)
                source_image = np.array(image)

                # Create two columns for side-by-side display
                col1, col2 = st.columns(2)

                # Display source image in left column
                with col1:
                    st.subheader("Source Image")
                    st.image(source_image, use_container_width=True)

                # Initialize ranges
                ranges = ExpressionRanges()

                # Generate parameters
                params, params_dict = generate_expression_params(ranges, use_sliders=(mode == "Manual Sliders"))

                # Determine when to process the image
                should_process = (mode == "Manual Sliders") or (mode == "Random Generation" and generate_button)

                if should_process:
                    # Process image using the current parameter values
                    processed_image = process_image(inferencer, source_image, params)

                    if processed_image is not None:
                        # Display processed image in right column
                        with col2:
                            st.subheader("Processed Image")
                            st.image(processed_image, use_container_width=True)

                        # Display parameters as JSON below the images
                        st.subheader("Generated Parameters")
                        st.json(params_dict)

                        # Add save button
                        if st.button("Save Parameters"):
                            output_path = save_params_json(params_dict)
                            if output_path:
                                st.success(f"Parameters saved to: {output_path}")

            except Exception as e:
                logger.error(f"Error processing uploaded file: {str(e)}")
                st.error(f"Error processing uploaded file: {str(e)}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")
        
if __name__ == "__main__":
    main()
