import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


class SileroVADSilenceDetector:
    def __init__(self):
        self.device = torch.device("cpu")  # Force CPU as recommended

        # Download and initialize Silero VAD
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=False
        )

        self.model = model.to(self.device)
        self.get_speech_timestamps = utils[0]
        self.save_audio = utils[2]  # Utility function for saving audio chunks

        # Model parameters
        self.sample_rate = 16000  # Model expects 16kHz

    def detect_silence(self, audio_path, threshold=0.5, min_silence_duration_ms=500):
        """
        Detect silence regions in audio file using Silero VAD.

        Args:
            audio_path: Path to audio file
            threshold: VAD threshold (higher = more sensitive to speech)
            min_silence_duration_ms: Minimum silence duration in milliseconds

        Returns:
            List of tuples containing (start_time, end_time) for silence regions
        """
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        # Ensure audio is on CPU
        waveform = waveform.to(self.device)

        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            waveform[0],
            self.model,
            threshold=threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )

        # Convert speech timestamps to silence regions
        silence_regions = []
        current_time = 0.0

        for speech in speech_timestamps:
            # If there's a gap between current_time and speech start, it's silence
            if speech["start"] - current_time >= min_silence_duration_ms / 1000:
                silence_regions.append((current_time, speech["start"]))
            current_time = speech["end"]

        # Add final silence region if needed
        audio_length = len(waveform[0]) / self.sample_rate
        if audio_length - current_time >= min_silence_duration_ms / 1000:
            silence_regions.append((current_time, audio_length))

        return silence_regions

    def visualize_silence_regions(self, audio_path, silence_regions):
        """
        Visualize the audio waveform and detected silence regions

        Args:
            audio_path: Path to audio file
            silence_regions: List of (start_time, end_time) tuples
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        time_axis = np.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1])

        plt.figure(figsize=(15, 5))
        plt.plot(time_axis, waveform[0].numpy(), alpha=0.5)

        # Highlight silence regions
        for start, end in silence_regions:
            plt.axvspan(start, end, color="red", alpha=0.2)

        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Audio Waveform with Detected Silence Regions")
        plt.grid(True)
        plt.show()

    def get_speech_segments(self, audio_path, threshold=0.5, min_silence_duration_ms=500):
        """
        Extract speech segments from audio file (inverse of silence detection)

        Args:
            audio_path: Path to audio file
            threshold: VAD threshold
            min_silence_duration_ms: Minimum silence duration in milliseconds

        Returns:
            List of tuples containing (start_time, end_time) for speech regions
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        # Get speech timestamps directly
        speech_timestamps = self.get_speech_timestamps(
            waveform[0],
            self.model,
            threshold=threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )

        return [(speech["start"], speech["end"]) for speech in speech_timestamps]


DETECTOR: SileroVADSilenceDetector | None = None


def get_silence_detector() -> SileroVADSilenceDetector:
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = SileroVADSilenceDetector()
    return DETECTOR
