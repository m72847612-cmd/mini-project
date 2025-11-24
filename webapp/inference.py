from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from config import Config
from model_factory import build_model
from train_utils import load_checkpoint


def _resolve_device(cli_device: Optional[str]) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PredictionPayload:
    filename: str
    deforestation_rate: float
    forest_rate: float
    deforested_pixels: int
    forest_pixels: int
    total_pixels: int
    mask_data_url: str
    overlay_data_url: str


class ForestInferenceService:
    """
    Tiny wrapper that keeps the trained U-Net model in memory
    and exposes a simple `predict` method usable by the FastAPI app.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = Config.best_model_path,
        *,
        device: str | None = None,
        image_size: int = Config.image_size,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}. "
                "Train the model or update Config.best_model_path before starting the server."
            )

        self.device = _resolve_device(device)
        self.image_size = image_size
        self.threshold = threshold
        self.overlay_color = np.array([214, 39, 40], dtype=np.float32)  # red tint for deforested pixels

        self.model = self._load_model()
        self.transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ]
        )

    def _load_model(self):
        model = build_model()
        model = load_checkpoint(model, str(self.checkpoint_path), self.device)
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        file_bytes: bytes,
        filename: str | None = None,
        *,
        threshold_override: float | None = None,
    ) -> PredictionPayload:
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc

        original_size = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        threshold = self._sanitize_threshold(threshold_override)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)
            mask_tensor = (probs > threshold).float()

        mask_np = mask_tensor.squeeze().cpu().numpy()
        if mask_np.ndim != 2:
            raise ValueError(f"Unexpected prediction shape: {mask_np.shape}")

        # Resize mask back to original dimensions for nicer visualization.
        mask_image = Image.fromarray((mask_np * 255).astype("uint8"), mode="L")
        mask_image = mask_image.resize(original_size, Image.NEAREST)

        overlay_image = self._build_overlay(image, mask_image)

        total_pixels = mask_np.size
        deforested_pixels = float(mask_np.sum())
        forest_pixels = total_pixels - deforested_pixels
        deforestation_percentage = (deforested_pixels / total_pixels) * 100.0
        forest_percentage = (forest_pixels / total_pixels) * 100.0

        filename = filename or "uploaded_image"

        return PredictionPayload(
            filename=filename,
            deforestation_rate=deforestation_percentage,
            forest_rate=forest_percentage,
            deforested_pixels=int(deforested_pixels),
            forest_pixels=int(forest_pixels),
            total_pixels=total_pixels,
            mask_data_url=self._to_data_url(mask_image),
            overlay_data_url=self._to_data_url(overlay_image),
        )

    def _build_overlay(self, original: Image.Image, mask_img: Image.Image, alpha: float = 0.55) -> Image.Image:
        """
        Colorize the predicted mask and blend it with the original image.
        """
        orig_np = np.array(original).astype(np.float32)
        mask_np = (np.array(mask_img).astype(np.float32) / 255.0)[..., None]

        tinted = orig_np * (1 - mask_np * alpha) + self.overlay_color * (mask_np * alpha)
        tinted = np.clip(tinted, 0, 255).astype(np.uint8)
        return Image.fromarray(tinted)

    @staticmethod
    def _to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"

    def _sanitize_threshold(self, override: float | None) -> float:
        if override is None:
            return self.threshold
        if not 0.05 <= override <= 0.95:
            raise ValueError("Threshold must be between 0.05 and 0.95.")
        return override


def build_service() -> ForestInferenceService:
    """
    Helper factory that can be used by FastAPI dependency injection.
    """
    return ForestInferenceService()


