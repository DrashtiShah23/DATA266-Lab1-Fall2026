"""Measure Task 3 metrics that were not retained during training.

Works with the notebook checkpoint format used by Task3_gan.ipynb
(keys like G_A2B / G_B2A, optional EMA and discriminator weights).

What it fills even when training never saved them:

- KID (polynomial MMD, mean and std)
- Generative precision and recall
- LPIPS (AlexNet, if the lpips package is installed)
- Content-preservation cosine similarity (VGG16 features of input vs translation)
- Cycle reconstruction L1
- Gradient norms (recomputed on a few live steps if D weights exist, else recovered from logs)
- NaN count, training time, images/sec, peak memory (recovered from JSONL epoch logs when present)

Does not invent Kaggle or human-audit scores.

Example, from the folder that contains task3_gan/data/:

    python3 -m pip install scipy tqdm pillow pytorch-fid lpips
    python3 task3_gan/drashti/src/measure_missing_metrics.py \\
        --checkpoint path/to/best_checkpoint.pt \\
        --update-csv task3_gan/drashti/metrics_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Architecture matching Task3_gan.ipynb (9 residual blocks, InstanceNorm)
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images + self.block(images)


class Generator(nn.Module):
    def __init__(self, residual_blocks: int = 9, base_channels: int = 64):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, base_channels, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.ReLU(inplace=True),
        ]
        channels = base_channels
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(channels * 2, affine=False),
                    nn.ReLU(inplace=True),
                ]
            )
            channels *= 2
        for _ in range(residual_blocks):
            layers.append(ResidualBlock(channels))
        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                    nn.InstanceNorm2d(channels // 2, affine=False),
                    nn.ReLU(inplace=True),
                ]
            )
            channels //= 2
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(channels, 3, kernel_size=7), nn.Tanh()])
        self.model = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()

        def block(in_channels: int, out_channels: int, stride: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels, affine=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3, base_channels, 2, False),
            *block(base_channels, base_channels * 2, 2),
            *block(base_channels * 2, base_channels * 4, 2),
            *block(base_channels * 4, base_channels * 8, 1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_rgb_tensor(path: Path, resolution: int = 256) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (resolution, resolution):
            rgb = rgb.resize((resolution, resolution), Image.Resampling.BICUBIC)
        array = np.asarray(rgb, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    return tensor * 2.0 - 1.0


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pick_state(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, torch.Tensor] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    return None


def load_generators(checkpoint: Path, residual_blocks: int, base_channels: int, prefer_ema: bool) -> tuple[Generator, Generator, dict[str, Any]]:
    payload = torch_load(checkpoint)
    if not isinstance(payload, dict):
        raise SystemExit(f"Checkpoint is not a dict: {checkpoint}")

    a2b_keys = ("G_A2B_EMA", "G_A2B", "generator_a2b") if prefer_ema else ("G_A2B", "G_A2B_EMA", "generator_a2b")
    b2a_keys = ("G_B2A_EMA", "G_B2A", "generator_b2a") if prefer_ema else ("G_B2A", "G_B2A_EMA", "generator_b2a")
    state_a2b = pick_state(payload, a2b_keys)
    state_b2a = pick_state(payload, b2a_keys)
    if state_a2b is None or state_b2a is None:
        raise SystemExit(f"Could not find generator weights in {checkpoint}. Keys: {sorted(payload.keys())}")

    generator_a2b = Generator(residual_blocks=residual_blocks, base_channels=base_channels)
    generator_b2a = Generator(residual_blocks=residual_blocks, base_channels=base_channels)
    generator_a2b.load_state_dict(state_a2b, strict=True)
    generator_b2a.load_state_dict(state_b2a, strict=True)
    return generator_a2b, generator_b2a, payload


# ---------------------------------------------------------------------------
# Feature metrics
# ---------------------------------------------------------------------------


def build_inception(device: torch.device):
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as error:
        raise SystemExit("Install pytorch-fid: python3 -m pip install pytorch-fid") from error
    network = InceptionV3(output_blocks=[3], resize_input=True, normalize_input=True).to(device)
    network.eval()

    @torch.no_grad()
    def extract(images: torch.Tensor) -> np.ndarray:
        unit = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
        features = network(unit)[0].squeeze(-1).squeeze(-1)
        return features.detach().cpu().numpy()

    return extract


def _matrix_square_root(matrix: np.ndarray) -> np.ndarray:
    from scipy import linalg

    result = linalg.sqrtm(matrix)
    return result[0] if isinstance(result, tuple) else result


def frechet_distance(real: np.ndarray, fake: np.ndarray) -> float:
    mu_r, mu_f = real.mean(axis=0), fake.mean(axis=0)
    sigma_r, sigma_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    diff = mu_r - mu_f
    product = _matrix_square_root(sigma_r.dot(sigma_f))
    if not np.isfinite(product).all():
        offset = np.eye(sigma_r.shape[0]) * 1e-6
        product = _matrix_square_root((sigma_r + offset).dot(sigma_f + offset))
    if np.iscomplexobj(product):
        product = product.real
    return float(diff.dot(diff) + np.trace(sigma_r + sigma_f - 2.0 * product))


def kernel_inception_distance(real: np.ndarray, fake: np.ndarray, subsets: int, subset_size: int, seed: int) -> dict[str, float]:
    dimension = real.shape[1]
    size = min(subset_size, real.shape[0], fake.shape[0])
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(subsets):
        real_subset = real[generator.choice(real.shape[0], size, replace=False)]
        fake_subset = fake[generator.choice(fake.shape[0], size, replace=False)]
        kernel_rr = (real_subset.dot(real_subset.T) / dimension + 1.0) ** 3
        kernel_ff = (fake_subset.dot(fake_subset.T) / dimension + 1.0) ** 3
        kernel_rf = (real_subset.dot(fake_subset.T) / dimension + 1.0) ** 3
        off = size * (size - 1)
        scores.append((kernel_rr.sum() - np.trace(kernel_rr)) / off + (kernel_ff.sum() - np.trace(kernel_ff)) / off - 2.0 * kernel_rf.mean())
    return {"kid_mean": float(np.mean(scores)), "kid_std": float(np.std(scores)), "kid_subset_size": float(size)}


def _pairwise_distances(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    squared = (left * left).sum(dim=1, keepdim=True) + (right * right).sum(dim=1) - 2.0 * left.matmul(right.T)
    return squared.clamp_min(0.0).sqrt()


def precision_and_recall(real: np.ndarray, fake: np.ndarray, neighbours: int, device: torch.device) -> dict[str, float]:
    real_t = torch.from_numpy(real).to(device=device, dtype=torch.float32)
    fake_t = torch.from_numpy(fake).to(device=device, dtype=torch.float32)

    def radii(points: torch.Tensor) -> torch.Tensor:
        rank = min(neighbours, points.size(0) - 1)
        found = []
        for start in range(0, points.size(0), 512):
            distances = _pairwise_distances(points[start : start + 512], points)
            found.append(distances.kthvalue(rank + 1, dim=1).values)
        return torch.cat(found)

    def covered(queries: torch.Tensor, manifold: torch.Tensor, manifold_radii: torch.Tensor) -> float:
        hits = []
        for start in range(0, queries.size(0), 512):
            distances = _pairwise_distances(queries[start : start + 512], manifold)
            hits.append((distances <= manifold_radii.unsqueeze(0)).any(dim=1))
        return float(torch.cat(hits).float().mean())

    return {
        "precision": covered(fake_t, real_t, radii(real_t)),
        "recall": covered(real_t, fake_t, radii(fake_t)),
        "precision_recall_neighbours": float(neighbours),
    }


def memorisation_distance(real: np.ndarray, fake: np.ndarray) -> float:
    real_u = real / np.clip(np.linalg.norm(real, axis=1, keepdims=True), 1e-12, None)
    fake_u = fake / np.clip(np.linalg.norm(fake, axis=1, keepdims=True), 1e-12, None)
    nearest = []
    for start in range(0, fake_u.shape[0], 512):
        block = 1.0 - fake_u[start : start + 512].dot(real_u.T)
        nearest.append(block.min(axis=1))
    return float(np.concatenate(nearest).mean())


# ---------------------------------------------------------------------------
# LPIPS and content cosine
# ---------------------------------------------------------------------------


class ContentEncoder(nn.Module):
    """VGG16 features for content-preservation cosine similarity."""

    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision import models

        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features
        self.network = nn.Sequential(*list(vgg.children())[:23]).to(device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        unit = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
        normalized = (unit - self.mean) / self.std
        features = self.network(normalized)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)


def build_lpips(device: torch.device):
    try:
        import lpips
    except ImportError as error:
        raise SystemExit("Install lpips: python3 -m pip install lpips") from error
    network = lpips.LPIPS(net="alex").to(device)
    network.eval()
    return network


@torch.no_grad()
def mean_lpips(lpips_net, sources: torch.Tensor, translations: torch.Tensor) -> float:
    # lpips expects [-1, 1]
    return float(lpips_net(sources, translations).mean())


@torch.no_grad()
def mean_content_cosine(encoder: ContentEncoder, sources: torch.Tensor, translations: torch.Tensor) -> float:
    left = F.normalize(encoder(sources), dim=1)
    right = F.normalize(encoder(translations), dim=1)
    return float((left * right).sum(dim=1).mean())


# ---------------------------------------------------------------------------
# Translation + batching
# ---------------------------------------------------------------------------


@torch.no_grad()
def translate_paths(
    generator: nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    resolution: int,
) -> tuple[torch.Tensor, list[Path]]:
    generator.eval()
    outputs = []
    used = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        batch = torch.stack([load_rgb_tensor(path, resolution) for path in chunk]).to(device)
        outputs.append(generator(batch).cpu())
        used.extend(chunk)
    return torch.cat(outputs, dim=0), used


@torch.no_grad()
def features_from_tensors(extract, images: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, images.size(0), batch_size):
        chunks.append(extract(images[start : start + batch_size].to(device)))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def real_features(extract, paths: list[Path], device: torch.device, batch_size: int, resolution: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([load_rgb_tensor(path, resolution) for path in paths[start : start + batch_size]]).to(device)
        chunks.append(extract(batch))
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Recover retained training stats from logs
# ---------------------------------------------------------------------------


def recover_from_logs(log_globs: list[Path]) -> dict[str, Any]:
    """Pull epoch-level fields that training already wrote to JSONL."""
    recovered: dict[str, Any] = {
        "gradient_norm_mean_last_epoch": None,
        "gradient_norm_mean_all_epochs": None,
        "nan_count_total": None,
        "training_elapsed_seconds": None,
        "images_per_second_last_epoch": None,
        "images_per_second_mean": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
        "source_log": None,
        "epochs_found": 0,
    }
    epoch_rows: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    chosen_log = None
    for path in log_globs:
        if not path.is_file():
            continue
        local_epochs = []
        local_completed = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("record_type") == "epoch":
                        local_epochs.append(row)
                    elif row.get("record_type") == "completed":
                        local_completed.append(row)
        except OSError:
            continue
        if len(local_epochs) >= len(epoch_rows):
            epoch_rows = local_epochs
            completed_rows = local_completed
            chosen_log = path

    if not epoch_rows:
        return recovered

    recovered["source_log"] = chosen_log.as_posix() if chosen_log else None
    recovered["epochs_found"] = len(epoch_rows)
    grad_keys = ("generator_grad_norm_mean", "generator_grad_norm", "grad_norm_mean", "grad_norm")
    grads = []
    ips = []
    nan_total = 0.0
    peak_alloc = []
    peak_reserved = []
    for row in epoch_rows:
        for key in grad_keys:
            if key in row and row[key] is not None:
                grads.append(float(row[key]))
                break
        if row.get("images_per_second") is not None:
            ips.append(float(row["images_per_second"]))
        if row.get("nan_count") is not None:
            nan_total += float(row["nan_count"])
        if row.get("cuda_peak_allocated_bytes") is not None:
            peak_alloc.append(int(row["cuda_peak_allocated_bytes"]))
        if row.get("cuda_peak_reserved_bytes") is not None:
            peak_reserved.append(int(row["cuda_peak_reserved_bytes"]))
        if row.get("elapsed_seconds") is not None:
            recovered["training_elapsed_seconds"] = float(row["elapsed_seconds"])

    if grads:
        recovered["gradient_norm_mean_last_epoch"] = grads[-1]
        recovered["gradient_norm_mean_all_epochs"] = float(np.mean(grads))
    if ips:
        recovered["images_per_second_last_epoch"] = ips[-1]
        recovered["images_per_second_mean"] = float(np.mean(ips))
    recovered["nan_count_total"] = nan_total
    if peak_alloc:
        recovered["cuda_peak_allocated_bytes"] = max(peak_alloc)
    if peak_reserved:
        recovered["cuda_peak_reserved_bytes"] = max(peak_reserved)
    if completed_rows and completed_rows[-1].get("elapsed_seconds") is not None:
        recovered["training_elapsed_seconds"] = float(completed_rows[-1]["elapsed_seconds"])
    return recovered


def recompute_gradient_norm(
    payload: dict[str, Any],
    generator_a2b: Generator,
    generator_b2a: Generator,
    monet_paths: list[Path],
    photo_paths: list[Path],
    device: torch.device,
    residual_blocks: int,
    base_channels: int,
    steps: int,
) -> dict[str, Any] | None:
    """If discriminators are in the checkpoint, run a few LSGAN steps and record grad norms."""
    state_d_a = pick_state(payload, ("D_A", "discriminator_a"))
    state_d_b = pick_state(payload, ("D_B", "discriminator_b"))
    if state_d_a is None or state_d_b is None:
        return None
    if not monet_paths or not photo_paths:
        return None

    discriminator_a = PatchDiscriminator(base_channels).to(device)
    discriminator_b = PatchDiscriminator(base_channels).to(device)
    discriminator_a.load_state_dict(state_d_a, strict=False)
    discriminator_b.load_state_dict(state_d_b, strict=False)
    generator_a2b = generator_a2b.to(device).train()
    generator_b2a = generator_b2a.to(device).train()
    discriminator_a.train()
    discriminator_b.train()

    opt_g = torch.optim.Adam(list(generator_a2b.parameters()) + list(generator_b2a.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_da = torch.optim.Adam(discriminator_a.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_db = torch.optim.Adam(discriminator_b.parameters(), lr=2e-4, betas=(0.5, 0.999))
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    norms = []
    nan_count = 0
    for step in range(steps):
        real_a = load_rgb_tensor(monet_paths[step % len(monet_paths)]).unsqueeze(0).to(device)
        real_b = load_rgb_tensor(photo_paths[step % len(photo_paths)]).unsqueeze(0).to(device)
        with torch.no_grad():
            fake_b = generator_a2b(real_a)
            fake_a = generator_b2a(real_b)
        opt_da.zero_grad(set_to_none=True)
        loss_da = 0.5 * (mse(discriminator_a(real_a), torch.ones_like(discriminator_a(real_a))) + mse(discriminator_a(fake_a.detach()), torch.zeros_like(discriminator_a(fake_a))))
        loss_da.backward()
        opt_da.step()
        opt_db.zero_grad(set_to_none=True)
        loss_db = 0.5 * (mse(discriminator_b(real_b), torch.ones_like(discriminator_b(real_b))) + mse(discriminator_b(fake_b.detach()), torch.zeros_like(discriminator_b(fake_b))))
        loss_db.backward()
        opt_db.step()
        opt_g.zero_grad(set_to_none=True)
        fake_b = generator_a2b(real_a)
        fake_a = generator_b2a(real_b)
        rec_a = generator_b2a(fake_b)
        rec_b = generator_a2b(fake_a)
        loss_g = (
            mse(discriminator_b(fake_b), torch.ones_like(discriminator_b(fake_b)))
            + mse(discriminator_a(fake_a), torch.ones_like(discriminator_a(fake_a)))
            + 10.0 * (l1(rec_a, real_a) + l1(rec_b, real_b))
        )
        if not torch.isfinite(loss_g):
            nan_count += 1
            continue
        loss_g.backward()
        total = 0.0
        for module in (generator_a2b, generator_b2a):
            for parameter in module.parameters():
                if parameter.grad is not None:
                    total += float(parameter.grad.detach().pow(2).sum().cpu())
        norms.append(total ** 0.5)
        opt_g.step()

    generator_a2b.eval()
    generator_b2a.eval()
    if not norms:
        return {"recomputed_gradient_norm_mean": None, "recomputed_nan_count": nan_count, "recomputed_steps": steps}
    return {
        "recomputed_gradient_norm_mean": float(np.mean(norms)),
        "recomputed_gradient_norm_last": float(norms[-1]),
        "recomputed_nan_count": nan_count,
        "recomputed_steps": steps,
        "note": "Live probe after loading the checkpoint. Not the historical training mean unless logs were missing.",
    }


# ---------------------------------------------------------------------------
# CSV update
# ---------------------------------------------------------------------------


PENDING_MARKERS = ("PENDING", "pending", "")


def _is_pending(value: str | None) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.upper().startswith("PENDING")


def update_metrics_csv(path: Path, measured: dict[str, Any]) -> Path:
    """Fill PENDING cells in metrics_report.csv style tables; write a sibling filled file too."""
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else ["metric", "photo_to_monet", "monet_to_photo", "overall", "status"]

    mapping = {
        "KID": ("kid_photo_to_monet", "kid_monet_to_photo", "kid_mean_overall"),
        "Generative precision": ("precision_photo_to_monet", "precision_monet_to_photo", "precision_overall"),
        "Generative recall": ("recall_photo_to_monet", "recall_monet_to_photo", "recall_overall"),
        "LPIPS": ("lpips_photo_to_monet", "lpips_monet_to_photo", "lpips_overall"),
        "Content preservation cosine similarity": (
            "content_cosine_photo_to_monet",
            "content_cosine_monet_to_photo",
            "content_cosine_overall",
        ),
        "Gradient norm": (None, None, "gradient_norm"),
        "NaN count": (None, None, "nan_count"),
        "Training time": (None, None, "training_time_seconds"),
        "Images per second": (None, None, "images_per_second"),
        "Peak memory": (None, None, "peak_memory_bytes"),
        "Cycle reconstruction L1": ("cycle_l1_photo_to_monet", "cycle_l1_monet_to_photo", "cycle_l1_overall"),
    }

    for row in rows:
        metric = row.get("metric", "")
        if metric not in mapping:
            continue
        left_key, right_key, overall_key = mapping[metric]
        if left_key and _is_pending(row.get("photo_to_monet")) and measured.get(left_key) is not None:
            row["photo_to_monet"] = f"{measured[left_key]:.6f}"
        if right_key and _is_pending(row.get("monet_to_photo")) and measured.get(right_key) is not None:
            row["monet_to_photo"] = f"{measured[right_key]:.6f}"
        if overall_key and _is_pending(row.get("overall")) and measured.get(overall_key) is not None:
            value = measured[overall_key]
            row["overall"] = f"{value:.6f}" if isinstance(value, float) else str(value)
            row["status"] = "measured"
        elif left_key and measured.get(left_key) is not None:
            row["status"] = "measured"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filled = path.with_name(f"{path.stem}_filled_{stamp}_{uuid.uuid4().hex[:8]}.csv")
    with filled.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Also overwrite the original pending file with filled values.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return filled


def write_json_report(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"task3_missing_metrics_{stamp}_{uuid.uuid4().hex[:8]}.json"
    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python3 task3_gan/drashti/src/measure_missing_metrics.py")
    parser.add_argument("--checkpoint", required=True, help="Notebook-style .pt with G_A2B / G_B2A (EMA optional).")
    parser.add_argument("--root", default=".", help="Project root that contains task3_gan/data/.")
    parser.add_argument("--monet-dir", default=None)
    parser.add_argument("--photo-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Where to write the JSON report.")
    parser.add_argument("--update-csv", action="append", default=[], help="metrics_report.csv path(s) to fill. Repeatable.")
    parser.add_argument("--generated-count", type=int, default=300, help="Photos to translate for photo→Monet. 0 = all.")
    parser.add_argument("--monet-count", type=int, default=300, help="Monet files for Monet→photo. 0 = all.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--residual-blocks", type=int, default=9)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--neighbours", type=int, default=3)
    parser.add_argument("--seed", type=int, default=266)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--skip-opposite-direction", action="store_true", help="Only score photo→Monet.")
    parser.add_argument("--grad-probe-steps", type=int, default=8, help="Live gradient probe steps if D weights exist.")
    parser.add_argument("--log-glob", action="append", default=[], help="Extra raw log files to scan. Repeatable.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    monet_dir = Path(args.monet_dir) if args.monet_dir else root / "task3_gan" / "data" / "monet_jpg"
    photo_dir = Path(args.photo_dir) if args.photo_dir else root / "task3_gan" / "data" / "photo_jpg"
    output_dir = Path(args.output_dir) if args.output_dir else root / "task3_gan" / "drashti" / "outputs"
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"CHECKPOINT_MISSING {checkpoint}")
        return 2

    monet_files = list_images(monet_dir)
    photo_files = list_images(photo_dir)
    if not monet_files or not photo_files:
        print(f"DATASET_MISSING monet={len(monet_files)} photo={len(photo_files)}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint}")
    print(f"monet files: {len(monet_files)}  photo files: {len(photo_files)}")

    generator_a2b, generator_b2a, payload = load_generators(
        checkpoint, args.residual_blocks, args.base_channels, args.prefer_ema
    )
    generator_a2b = generator_a2b.to(device).eval()
    generator_b2a = generator_b2a.to(device).eval()

    photo_subset = photo_files if args.generated_count < 1 else photo_files[: args.generated_count]
    monet_subset = monet_files if args.monet_count < 1 else monet_files[: args.monet_count]

    extract = build_inception(device)
    content_encoder = ContentEncoder(device)
    lpips_net = None if args.skip_lpips else build_lpips(device)

    started = time.perf_counter()
    print("Extracting real Monet features...")
    feat_real_monet = real_features(extract, monet_files[: max(len(monet_subset), 300)], device, args.batch_size, 256)

    print(f"Translating {len(photo_subset)} photos -> Monet...")
    fake_monet, used_photos = translate_paths(generator_b2a, photo_subset, device, args.batch_size, 256)
    # paired sources for LPIPS / content / cycle
    source_photos = torch.stack([load_rgb_tensor(path) for path in used_photos])
    with torch.no_grad():
        cycle_photos = generator_a2b(fake_monet.to(device)).cpu()
    feat_fake_monet = features_from_tensors(extract, fake_monet, device, args.batch_size)

    photo_to_monet = {
        "fid": frechet_distance(feat_real_monet, feat_fake_monet),
        **kernel_inception_distance(feat_real_monet, feat_fake_monet, args.kid_subsets, args.kid_subset_size, args.seed),
        **precision_and_recall(feat_real_monet, feat_fake_monet, args.neighbours, device),
        "memorisation_distance": memorisation_distance(feat_real_monet, feat_fake_monet),
        "cycle_l1": float((cycle_photos - source_photos).abs().mean()),
        "content_cosine": mean_content_cosine(content_encoder, source_photos.to(device), fake_monet.to(device)),
        "count_real": int(feat_real_monet.shape[0]),
        "count_generated": int(feat_fake_monet.shape[0]),
    }
    if lpips_net is not None:
        # LPIPS in chunks to limit VRAM
        scores = []
        for start in range(0, source_photos.size(0), args.batch_size):
            scores.append(
                mean_lpips(
                    lpips_net,
                    source_photos[start : start + args.batch_size].to(device),
                    fake_monet[start : start + args.batch_size].to(device),
                )
            )
        photo_to_monet["lpips"] = float(np.mean(scores))

    monet_to_photo: dict[str, Any] | None = None
    if not args.skip_opposite_direction:
        print("Extracting real photo features...")
        feat_real_photo = real_features(extract, photo_files[: max(len(photo_subset), 300)], device, args.batch_size, 256)
        print(f"Translating {len(monet_subset)} Monet -> photo...")
        fake_photo, used_monet = translate_paths(generator_a2b, monet_subset, device, args.batch_size, 256)
        source_monet = torch.stack([load_rgb_tensor(path) for path in used_monet])
        with torch.no_grad():
            cycle_monet = generator_b2a(fake_photo.to(device)).cpu()
        feat_fake_photo = features_from_tensors(extract, fake_photo, device, args.batch_size)
        monet_to_photo = {
            "fid": frechet_distance(feat_real_photo, feat_fake_photo),
            **kernel_inception_distance(feat_real_photo, feat_fake_photo, args.kid_subsets, args.kid_subset_size, args.seed + 1),
            **precision_and_recall(feat_real_photo, feat_fake_photo, args.neighbours, device),
            "memorisation_distance": memorisation_distance(feat_real_photo, feat_fake_photo),
            "cycle_l1": float((cycle_monet - source_monet).abs().mean()),
            "content_cosine": mean_content_cosine(content_encoder, source_monet.to(device), fake_photo.to(device)),
            "count_real": int(feat_real_photo.shape[0]),
            "count_generated": int(feat_fake_photo.shape[0]),
        }
        if lpips_net is not None:
            scores = []
            for start in range(0, source_monet.size(0), args.batch_size):
                scores.append(
                    mean_lpips(
                        lpips_net,
                        source_monet[start : start + args.batch_size].to(device),
                        fake_photo[start : start + args.batch_size].to(device),
                    )
                )
            monet_to_photo["lpips"] = float(np.mean(scores))

    # Recover training telemetry
    default_logs = list((root / "reproducibility" / "raw_logs").glob("task3*.log")) if (root / "reproducibility" / "raw_logs").is_dir() else []
    extra_logs = [Path(item) for item in args.log_glob]
    recovered = recover_from_logs(default_logs + extra_logs)
    grad_probe = recompute_gradient_norm(
        payload,
        generator_a2b,
        generator_b2a,
        monet_files,
        photo_files,
        device,
        args.residual_blocks,
        args.base_channels,
        args.grad_probe_steps,
    )

    parameter_count = sum(parameter.numel() for parameter in list(generator_a2b.parameters()) + list(generator_b2a.parameters()))
    # include D if present for reporting parity with notebook total
    for key in ("D_A", "D_B", "discriminator_a", "discriminator_b"):
        state = pick_state(payload, (key,))
        if state is not None:
            parameter_count += sum(tensor.numel() for tensor in state.values() if torch.is_tensor(tensor))

    gradient_norm = recovered.get("gradient_norm_mean_all_epochs")
    if gradient_norm is None and grad_probe and grad_probe.get("recomputed_gradient_norm_mean") is not None:
        gradient_norm = grad_probe["recomputed_gradient_norm_mean"]

    nan_count = recovered.get("nan_count_total")
    if nan_count is None and grad_probe is not None:
        nan_count = grad_probe.get("recomputed_nan_count")

    flat = {
        "kid_photo_to_monet": photo_to_monet["kid_mean"],
        "precision_photo_to_monet": photo_to_monet["precision"],
        "recall_photo_to_monet": photo_to_monet["recall"],
        "lpips_photo_to_monet": photo_to_monet.get("lpips"),
        "content_cosine_photo_to_monet": photo_to_monet["content_cosine"],
        "cycle_l1_photo_to_monet": photo_to_monet["cycle_l1"],
        "gradient_norm": gradient_norm,
        "nan_count": nan_count,
        "training_time_seconds": recovered.get("training_elapsed_seconds"),
        "images_per_second": recovered.get("images_per_second_mean") or recovered.get("images_per_second_last_epoch"),
        "peak_memory_bytes": recovered.get("cuda_peak_allocated_bytes"),
    }
    if monet_to_photo is not None:
        flat.update(
            {
                "kid_monet_to_photo": monet_to_photo["kid_mean"],
                "precision_monet_to_photo": monet_to_photo["precision"],
                "recall_monet_to_photo": monet_to_photo["recall"],
                "lpips_monet_to_photo": monet_to_photo.get("lpips"),
                "content_cosine_monet_to_photo": monet_to_photo["content_cosine"],
                "cycle_l1_monet_to_photo": monet_to_photo["cycle_l1"],
                "kid_mean_overall": (photo_to_monet["kid_mean"] + monet_to_photo["kid_mean"]) / 2.0,
                "precision_overall": (photo_to_monet["precision"] + monet_to_photo["precision"]) / 2.0,
                "recall_overall": (photo_to_monet["recall"] + monet_to_photo["recall"]) / 2.0,
                "content_cosine_overall": (photo_to_monet["content_cosine"] + monet_to_photo["content_cosine"]) / 2.0,
                "cycle_l1_overall": (photo_to_monet["cycle_l1"] + monet_to_photo["cycle_l1"]) / 2.0,
            }
        )
        if photo_to_monet.get("lpips") is not None and monet_to_photo.get("lpips") is not None:
            flat["lpips_overall"] = (photo_to_monet["lpips"] + monet_to_photo["lpips"]) / 2.0
    else:
        flat["kid_mean_overall"] = photo_to_monet["kid_mean"]
        flat["precision_overall"] = photo_to_monet["precision"]
        flat["recall_overall"] = photo_to_monet["recall"]
        flat["content_cosine_overall"] = photo_to_monet["content_cosine"]
        flat["cycle_l1_overall"] = photo_to_monet["cycle_l1"]
        if photo_to_monet.get("lpips") is not None:
            flat["lpips_overall"] = photo_to_monet["lpips"]

    report = {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_keys": sorted(str(key) for key in payload.keys()),
        "device": str(device),
        "prefer_ema": bool(args.prefer_ema),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count_loaded_modules": parameter_count,
        "photo_to_monet": photo_to_monet,
        "monet_to_photo": monet_to_photo,
        "recovered_from_logs": recovered,
        "gradient_probe": grad_probe,
        "flat_for_csv": flat,
        "notes": [
            "KID / precision / recall / LPIPS / content cosine are computed now from the checkpoint.",
            "Gradient norm prefers historical log means; falls back to a short live probe if discriminators are stored.",
            "Training time, images/sec, peak memory, and NaN count come from raw logs when those files still exist.",
            "Human audit and Kaggle rows are left untouched.",
        ],
    }
    report_path = write_json_report(output_dir, report)

    csv_paths = [Path(item) for item in args.update_csv]
    if not csv_paths:
        for candidate in (
            root / "task3_gan" / "drashti" / "metrics_report.csv",
            root / "task3_gan" / "drashti" / "full_metrics_report.csv",
        ):
            if candidate.is_file():
                csv_paths.append(candidate)
    filled_paths = []
    for csv_path in csv_paths:
        filled_paths.append(update_metrics_csv(csv_path, flat).as_posix())

    print()
    print("=" * 60)
    print("PHOTO TO MONET")
    print("=" * 60)
    print(f"FID                 {photo_to_monet['fid']:.4f}")
    print(f"KID mean            {photo_to_monet['kid_mean']:.6f}  std {photo_to_monet['kid_std']:.6f}")
    print(f"precision           {photo_to_monet['precision']:.4f}")
    print(f"recall              {photo_to_monet['recall']:.4f}")
    print(f"content cosine      {photo_to_monet['content_cosine']:.4f}")
    print(f"cycle L1            {photo_to_monet['cycle_l1']:.6f}")
    if "lpips" in photo_to_monet:
        print(f"LPIPS               {photo_to_monet['lpips']:.4f}")
    if monet_to_photo is not None:
        print()
        print("MONET TO PHOTO")
        print(f"FID                 {monet_to_photo['fid']:.4f}")
        print(f"KID mean            {monet_to_photo['kid_mean']:.6f}")
        print(f"precision           {monet_to_photo['precision']:.4f}")
        print(f"recall              {monet_to_photo['recall']:.4f}")
        print(f"content cosine      {monet_to_photo['content_cosine']:.4f}")
        if "lpips" in monet_to_photo:
            print(f"LPIPS               {monet_to_photo['lpips']:.4f}")
    print()
    print("RECOVERED / PROBED")
    print(f"gradient norm       {gradient_norm}")
    print(f"nan count           {nan_count}")
    print(f"training seconds    {recovered.get('training_elapsed_seconds')}")
    print(f"images / second     {flat.get('images_per_second')}")
    print(f"peak alloc bytes    {recovered.get('cuda_peak_allocated_bytes')}")
    print(f"source log          {recovered.get('source_log')}")
    print()
    print(f"json report: {report_path}")
    for path in filled_paths:
        print(f"filled csv:  {path}")
    print("MISSING_METRICS=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
