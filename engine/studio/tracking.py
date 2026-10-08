"""Rastreadores de objetos em vídeo.

`Sam2Tracker` envolve o `SAM2VideoPredictor` oficial da Meta. `DemoTracker` é
um substituto geométrico (círculos em volta dos cliques, sem IA) usado nos
testes automatizados e para desenvolver a interface sem PyTorch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

FrameCallback = Callable[[int, dict[int, np.ndarray]], None]


class TrackerError(RuntimeError):
    pass


class Tracker(ABC):
    name: str
    device: str

    @abstractmethod
    def load_video(self, frames_dir: Path) -> None:
        """Carrega os quadros JPEG e zera todos os prompts."""

    @abstractmethod
    def reset(self) -> None:
        """Remove prompts e resultados, mantendo o vídeo carregado."""

    @abstractmethod
    def add_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        """Usa uma máscara existente como condição neste quadro."""

    @abstractmethod
    def add_points(
        self, frame_idx: int, obj_id: int, points: Sequence[Sequence[float]], labels: Sequence[int]
    ) -> np.ndarray:
        """Define os cliques do objeto neste quadro e devolve a máscara resultante (H, W) bool.

        `points` substitui os cliques anteriores do objeto neste quadro; a máscara
        já existente no quadro (de `add_mask`, de cliques anteriores ou do
        rastreamento) é usada como ponto de partida para a correção.
        """

    @abstractmethod
    def propagate(self, start_frame: int, on_frame: FrameCallback) -> None:
        """Rastreia para frente a partir de `start_frame` e depois para trás até o quadro 0."""

    def unload(self) -> None:
        """Libera a memória do vídeo carregado (o modelo continua carregado)."""


def resolve_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Sam2Tracker(Tracker):
    name = "sam2"

    def __init__(self, config: str, checkpoint: Path | None, device: str = "auto"):
        """`checkpoint=None` cria a rede com pesos aleatórios: só serve para testar o encanamento."""
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise TrackerError(
                "SAM 2 não está instalado neste ambiente Python. Rode scripts/setup.sh."
            ) from exc
        if checkpoint is not None and not checkpoint.exists():
            raise TrackerError(
                f"Checkpoint do SAM 2.1 não encontrado em {checkpoint}. "
                "Baixe com: studio models download"
            )
        self._torch = torch
        dev = resolve_device(device)
        self.device = str(dev)
        self._predictor = build_sam2_video_predictor(
            config, str(checkpoint) if checkpoint is not None else None, device=dev
        )
        self._state = None

    def _require_state(self):
        if self._state is None:
            raise TrackerError("Nenhum vídeo carregado no rastreador.")
        return self._state

    def load_video(self, frames_dir: Path) -> None:
        self._state = None  # libera a memória do vídeo anterior antes de carregar o novo
        self._state = self._predictor.init_state(
            video_path=str(frames_dir), offload_video_to_cpu=True, async_loading_frames=False
        )

    def reset(self) -> None:
        self._predictor.reset_state(self._require_state())

    def unload(self) -> None:
        self._state = None

    def add_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        self._predictor.add_new_mask(
            inference_state=self._require_state(), frame_idx=frame_idx, obj_id=obj_id,
            mask=np.asarray(mask, dtype=bool),
        )

    def add_points(self, frame_idx, obj_id, points, labels) -> np.ndarray:
        _, obj_ids, logits = self._predictor.add_new_points_or_box(
            inference_state=self._require_state(),
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.asarray(points, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int32),
            clear_old_points=True,
        )
        return self._masks_by_obj(obj_ids, logits)[obj_id]

    def propagate(self, start_frame: int, on_frame: FrameCallback) -> None:
        state = self._require_state()
        for reverse in (False, True):
            if reverse and start_frame == 0:
                break
            for frame_idx, obj_ids, logits in self._predictor.propagate_in_video(
                state, start_frame_idx=start_frame, reverse=reverse
            ):
                on_frame(frame_idx, self._masks_by_obj(obj_ids, logits))

    @staticmethod
    def _masks_by_obj(obj_ids, logits) -> dict[int, np.ndarray]:
        masks = (logits > 0.0).cpu().numpy()
        return {int(obj_id): masks[i, 0] for i, obj_id in enumerate(obj_ids)}


class DemoTracker(Tracker):
    """Rastreador falso: discos em volta dos cliques, copiados para os quadros vizinhos."""

    name = "demo"
    device = "cpu"

    def __init__(self, radius_ratio: float = 0.12):
        self.radius_ratio = radius_ratio
        self._shape: tuple[int, int] | None = None
        self._num_frames = 0
        self.reset()

    def load_video(self, frames_dir: Path) -> None:
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            raise TrackerError(f"Nenhum quadro em {frames_dir}")
        with Image.open(frames[0]) as img:
            self._shape = (img.height, img.width)
        self._num_frames = len(frames)
        self.reset()

    def reset(self) -> None:
        self._seed: dict[tuple[int, int], np.ndarray] = {}
        self._cond: dict[tuple[int, int], np.ndarray] = {}

    def _disc(self, x: float, y: float) -> np.ndarray:
        h, w = self._shape
        yy, xx = np.ogrid[:h, :w]
        r = self.radius_ratio * min(h, w)
        return (xx - x) ** 2 + (yy - y) ** 2 <= r * r

    def add_mask(self, frame_idx, obj_id, mask) -> None:
        mask = np.asarray(mask, dtype=bool)
        self._seed[(frame_idx, obj_id)] = mask
        self._cond[(frame_idx, obj_id)] = mask

    def add_points(self, frame_idx, obj_id, points, labels) -> np.ndarray:
        if self._shape is None:
            raise TrackerError("Nenhum vídeo carregado no rastreador.")
        key = (frame_idx, obj_id)
        mask = self._seed.setdefault(key, np.zeros(self._shape, dtype=bool)).copy()
        for (x, y), label in zip(points, labels):
            mask = mask | self._disc(x, y) if label == 1 else mask & ~self._disc(x, y)
        self._cond[key] = mask
        return mask

    def propagate(self, start_frame: int, on_frame: FrameCallback) -> None:
        obj_ids = sorted({obj for _, obj in self._cond})
        order = list(range(start_frame, self._num_frames)) + list(range(start_frame - 1, -1, -1))
        for frame_idx in order:
            out = {}
            for obj in obj_ids:
                frames = [f for f, o in self._cond if o == obj]
                nearest = min(frames, key=lambda f: abs(f - frame_idx))
                out[obj] = self._cond[(nearest, obj)]
            on_frame(frame_idx, out)
