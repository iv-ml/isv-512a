# pylint: disable=C0301
"""
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
"""

import math

import torch
import torchaudio
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from lipsync.hallo_audio.wav2vec2 import Wav2VecModel


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int,
        wav2vec_model_path: str,
        only_last_features: bool,
        device: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.device = (
            device
            if device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.audio_encoder = Wav2VecModel.from_pretrained(
            wav2vec_model_path, local_files_only=False
        )
        self.audio_encoder = self.audio_encoder.to(self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.only_last_features = only_last_features

        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_model_path, local_files_only=False
        )

    def get_audio_offset_and_num_frames(
        self, video_path: str, f_start: int, f_end: int, video_fps: float
    ) -> tuple[int, int]:
        metadata = torchaudio.info(video_path)
        audio_fps = metadata.sample_rate

        start_time = f_start / video_fps
        end_time = f_end / video_fps

        frame_offset = int(start_time * audio_fps)
        num_frames = int((end_time - start_time) * audio_fps)

        return frame_offset, num_frames

    def load_audio(
        self, audio: str, frame_offset: int = 0, num_frames: int = -1
    ) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(
            audio, frame_offset=frame_offset, num_frames=num_frames
        )

        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze().to(self.device)

    def preprocess(
        self, audio: torch.Tensor, fps: float, clip_length: int = -1
    ) -> tuple[torch.Tensor, int]:
        # Extract wav2vec features
        input_values = self.wav2vec_feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_values
        audio_feature = input_values.squeeze().to(self.device)

        seq_len = math.ceil(audio_feature.shape[0] / self.sample_rate * fps)
        audio_length = seq_len
        seq_len = seq_len if clip_length < 0 else clip_length

        audio_feature = audio_feature.unsqueeze(0)
        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=seq_len, output_hidden_states=True
            )
        assert len(embeddings) > 0, "Failed to extract audio embedding"
        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        return audio_emb, audio_length
