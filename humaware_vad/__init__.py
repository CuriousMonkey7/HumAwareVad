import json
from typing import List

import torch

import numpy as np
from numpy.typing import NDArray

from huggingface_hub import hf_hub_download
from silero_vad import get_speech_timestamps

from fastrtc.utils import AudioChunk
from fastrtc.pause_detection.protocol import PauseDetectionModel
from fastrtc.pause_detection.silero import SileroVadOptions


class HumAwareVADModel(PauseDetectionModel):
    @staticmethod
    def download_and_load_model():
        config_path = hf_hub_download(repo_id="CuriousMonkey7/HumAware-VAD", filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        model_path = hf_hub_download(repo_id="CuriousMonkey7/HumAware-VAD", filename=config["model_file"])
        model = torch.jit.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        return model

    def __init__(self):
        self.model = self.download_and_load_model()
        self.warmup()

    collected_audio: list[np.ndarray] = []  # Global buffer to store all audio chunks

    @staticmethod
    def collect_chunks(audio: np.ndarray, chunks: List[AudioChunk]) -> np.ndarray:
        """Collects and concatenates audio chunks."""
        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(
            [audio[chunk["start"] : chunk["end"]] for chunk in chunks]
        )    

    def warmup(self):
        for _ in range(10):
            dummy_audio = np.zeros(102400, dtype=np.float32)
            self.vad((24000, dummy_audio), None)

    def vad(
        self,
        audio: tuple[int, NDArray[np.float32] | NDArray[np.int16]],
        options: None | SileroVadOptions,
    ) -> tuple[float, list[AudioChunk]]:
        
        sampling_rate, audio_ = audio
        if audio_.dtype != np.float32:
            audio_ = audio_.astype(np.float32) / 32768.0
        sr = 16000
        if sr != sampling_rate:
            try:
                import librosa  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Applying the VAD filter requires the librosa if the input sampling rate is not 16000hz"
                ) from e
            audio_ = librosa.resample(audio_, orig_sr=sampling_rate, target_sr=sr)

        speech_chunks = get_speech_timestamps(audio_, self.model, threshold=0.8)
        audio_ = self.collect_chunks(audio_, speech_chunks)
        duration_after_vad = audio_.shape[0] / sr
        return duration_after_vad, speech_chunks
