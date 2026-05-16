"""
FLIR ADAS v2 Paired Dataset
============================
Loads synchronised thermal–RGB image pairs from the FLIR ADAS v2 dataset.

Pairing strategy:
    The ONLY reliable pairing mechanism in FLIR ADAS v2 is the file
    `rgb_to_thermal_vid_map.json`, which maps 3,749 RGB frame filenames to
    their co-registered thermal counterparts from the video test split.
    These were captured simultaneously by the DualCapture rig.

    The training image splits (images_rgb_train / images_thermal_train) do
    NOT have frame-level pairing — their coco.json IDs are independent
    sequential counters, not scene-level correspondences.

    We use the 3,749 truly paired frames and split them into train/val
    ourselves (by video, to prevent leakage from temporally adjacent frames).

Key design choice — thermal channel replication:
    Thermal images are single-channel 8-bit grayscale.  We replicate to 3
    channels (H, W) → (3, H, W) so they can pass through a standard ResNet.
    We do NOT use ImageNet normalisation on the thermal branch because
    ImageNet statistics are physically meaningless for LWIR thermal emission.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Transforms ───────────────────────────────────────────────────────────

def get_rgb_transform(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Standard ImageNet-normalised RGB transforms."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_thermal_transform(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Thermal-specific transforms.

    No ImageNet normalisation — those statistics are computed from reflected
    visible light in natural images and have no physical relevance for LWIR
    thermal emission.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# ── Filename Parsing ─────────────────────────────────────────────────────

_FRAME_RE = re.compile(r"video-([A-Za-z0-9]+)-frame-(\d+)-[A-Za-z0-9]+\.jpg")


def _extract_video_id(fname: str):
    """Extract video_id from a FLIR ADAS v2 filename."""
    m = _FRAME_RE.match(fname)
    return m.group(1) if m else None


# ── Dataset ──────────────────────────────────────────────────────────────

class FLIRPairedDataset(Dataset):
    """
    Loads paired thermal + RGB images from FLIR ADAS v2.

    Uses rgb_to_thermal_vid_map.json for true frame-level pairing, split
    by video to prevent temporal leakage between train and val.

    Each __getitem__ returns:
        rgb_image    : (3, 224, 224) tensor, ImageNet-normalised
        thermal_image: (3, 224, 224) tensor, [0, 1] range, replicated channels
        thermal_raw  : (1, 224, 224) tensor, for physics temperature extraction
        index        : int, for retrieval evaluation bookkeeping
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        val_fraction: float = 0.2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        is_train = (split == "train")
        self.rgb_transform = get_rgb_transform(image_size, is_train)
        self.thermal_transform = get_thermal_transform(image_size, is_train)

        self.pairs = self._load_pairs(val_fraction)
        print(f"[FLIRPairedDataset] {split}: {len(self.pairs)} valid pairs")

    def _load_pairs(self, val_fraction: float):
        """
        Build list of (rgb_path, thermal_path) tuples from the vid map.

        The rgb_to_thermal_vid_map.json contains the ONLY true frame-level
        pairing in the dataset.  Files are in the video_rgb_test/ and
        video_thermal_test/ directories.

        We split by video ID (not by frame) to prevent temporal leakage:
        all frames from the same video go into the same split.
        """
        map_file = self.data_dir / "rgb_to_thermal_vid_map.json"
        if not map_file.exists():
            print("[Warning] rgb_to_thermal_vid_map.json not found")
            return []

        with open(map_file) as f:
            frame_map = json.load(f)

        # Possible directories for the paired images
        rgb_dir_candidates = [
            self.data_dir / "video_rgb_test" / "data",
            self.data_dir / "video_rgb_test",
        ]
        thermal_dir_candidates = [
            self.data_dir / "video_thermal_test" / "data",
            self.data_dir / "video_thermal_test",
        ]

        rgb_dir = next((d for d in rgb_dir_candidates if d.exists()), None)
        thermal_dir = next((d for d in thermal_dir_candidates if d.exists()), None)

        if rgb_dir is None or thermal_dir is None:
            print("[Warning] Video test directories not found")
            return self._fallback_from_train_dirs(frame_map)

        # Group frames by RGB video ID for video-level splitting
        video_frames = defaultdict(list)
        for rgb_fname, thermal_fname in frame_map.items():
            rgb_path = rgb_dir / rgb_fname
            thermal_path = thermal_dir / thermal_fname

            if rgb_path.exists() and thermal_path.exists():
                vid_id = _extract_video_id(rgb_fname) or "unknown"
                video_frames[vid_id].append((rgb_path, thermal_path))

        if not video_frames:
            print("[Warning] No valid pairs found in video test directories")
            return self._fallback_from_train_dirs(frame_map)

        # Split by video: ~80% train, ~20% val (by video, not by frame)
        video_ids = sorted(video_frames.keys())
        n_val_videos = max(1, int(len(video_ids) * val_fraction))
        val_videos = set(video_ids[-n_val_videos:])
        train_videos = set(video_ids) - val_videos

        target_videos = val_videos if self.split == "val" else train_videos

        pairs = []
        for vid_id in sorted(target_videos):
            pairs.extend(video_frames[vid_id])

        total = sum(len(v) for v in video_frames.values())
        print(f"  Vid-map pairing: {total} total pairs from {len(video_ids)} videos")
        print(f"  Split: {len(train_videos)} train videos, {len(val_videos)} val videos")

        return pairs

    def _fallback_from_train_dirs(self, frame_map):
        """
        Fallback: look for vid-map filenames in the training image directories.

        Some dataset layouts put all images in the train/val folders.
        """
        rgb_dirs = [
            self.data_dir / "images_rgb_train" / "data",
            self.data_dir / "images_rgb_val" / "data",
            self.data_dir / "images_rgb_train",
            self.data_dir / "images_rgb_val",
        ]
        thermal_dirs = [
            self.data_dir / "images_thermal_train" / "data",
            self.data_dir / "images_thermal_val" / "data",
            self.data_dir / "images_thermal_train",
            self.data_dir / "images_thermal_val",
        ]

        # Build filename → path index for all available images
        rgb_index = {}
        for d in rgb_dirs:
            if d.exists():
                for f in d.glob("*.jpg"):
                    rgb_index[f.name] = f

        thermal_index = {}
        for d in thermal_dirs:
            if d.exists():
                for f in d.glob("*.jpg"):
                    thermal_index[f.name] = f

        # Match via vid map
        video_frames = defaultdict(list)
        for rgb_fname, thermal_fname in frame_map.items():
            rgb_path = rgb_index.get(rgb_fname)
            thermal_path = thermal_index.get(thermal_fname)
            if rgb_path and thermal_path:
                vid_id = _extract_video_id(rgb_fname) or "unknown"
                video_frames[vid_id].append((rgb_path, thermal_path))

        if not video_frames:
            print("  [Warning] No vid-map pairs found in any directory")
            return []

        # Same video-level split
        video_ids = sorted(video_frames.keys())
        n_val = max(1, int(len(video_ids) * 0.2))
        val_videos = set(video_ids[-n_val:])
        train_videos = set(video_ids) - val_videos
        target = val_videos if self.split == "val" else train_videos

        pairs = []
        for vid_id in sorted(target):
            pairs.extend(video_frames[vid_id])

        total = sum(len(v) for v in video_frames.values())
        print(f"  Fallback pairing: {total} pairs from {len(video_ids)} videos")

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        rgb_path, thermal_path = self.pairs[idx]

        # Load RGB (3-channel)
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_tensor = self.rgb_transform(rgb_img)

        # Load thermal (grayscale)
        thermal_img = Image.open(thermal_path).convert("L")

        # Raw copy for physics temperature extraction
        thermal_raw = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])(thermal_img)

        # Augmented thermal for contrastive training
        thermal_tensor = self.thermal_transform(thermal_img)

        # Replicate single channel → 3 channels for ResNet compatibility.
        # We do NOT use ImageNet pretrained weights for the thermal encoder,
        # so there is no channel semantics mismatch.
        thermal_tensor = thermal_tensor.repeat(3, 1, 1)

        return rgb_tensor, thermal_tensor, thermal_raw, idx
