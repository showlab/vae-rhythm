"""Training augmentations.

* Artifact path: random resized crop (area 0.8-1.0 of the 256 x 256 cache frame, shared by
  the 8 frames of a clip) to 224, horizontal flip with p = 0.5. Test: centre crop 224.
* Semantic path: the 11 per-frame transforms of Effort's released DeepfakeBench training code,
  in the same order with the same probabilities and default parameters, re-implemented with
  OpenCV / NumPy so no particular albumentations version is needed:
  flip .5 | brightness-contrast .5 | hue-saturation-value .3 | JPEG 40-100 .1 | Gaussian noise .1
  | motion blur .1 | CLAHE .1 | channel shuffle .1 | cutout .1 | gamma .3 | glass blur .3
"""
from __future__ import annotations

import math

import numpy as np


def rrc_clip(x, train: bool):
    """x: float tensor (T, 3, 256, 256) in [0, 1] -> (T, 3, 224, 224)."""
    import torch.nn.functional as F

    if not train:
        return x[:, :, 16:240, 16:240]
    s = np.random.uniform(0.8, 1.0)
    c = int(round(256 * math.sqrt(s)))
    y0, x0 = np.random.randint(0, 256 - c + 1), np.random.randint(0, 256 - c + 1)
    x = x[:, :, y0:y0 + c, x0:x0 + c]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    if np.random.rand() < 0.5:
        x = x.flip(-1)
    return x


def dfb_frame(im: np.ndarray) -> np.ndarray:
    """One HWC uint8 RGB frame -> augmented frame (independent random parameters per call)."""
    import cv2

    R = np.random
    im = np.ascontiguousarray(im)
    H, W = im.shape[:2]
    if R.rand() < 0.5:                                                   # HorizontalFlip
        im = np.ascontiguousarray(im[:, ::-1])
    if R.rand() < 0.5:                                                   # RandomBrightnessContrast(0.2, 0.2)
        a = 1 + R.uniform(-0.2, 0.2)
        b = R.uniform(-0.2, 0.2) * 255
        im = np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if R.rand() < 0.3:                                                   # HueSaturationValue(20, 30, 20)
        h, s_, v = R.randint(-20, 21), R.randint(-30, 31), R.randint(-20, 21)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + h) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + s_, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + v, 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if R.rand() < 0.1:                                                   # ImageCompression(40, 100)
        q = int(R.randint(40, 101))
        _, buf = cv2.imencode(".jpg", np.ascontiguousarray(im[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
        im = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1])
    if R.rand() < 0.1:                                                   # GaussNoise(var 10-50)
        sig = R.uniform(10, 50) ** 0.5
        im = np.clip(im.astype(np.float32) + R.normal(0, sig, im.shape), 0, 255).astype(np.uint8)
    if R.rand() < 0.1:                                                   # MotionBlur(7)
        k = int(R.choice([3, 5, 7]))
        ker = np.zeros((k, k), np.float32)
        xs, xe = R.randint(0, k), R.randint(0, k)
        if xs == xe:
            ys, ye = R.choice(k, 2, replace=False)
        else:
            ys, ye = R.randint(0, k), R.randint(0, k)
        cv2.line(ker, (int(xs), int(ys)), (int(xe), int(ye)), 1, thickness=1)
        ker /= ker.sum()
        im = cv2.filter2D(im, -1, ker)
    if R.rand() < 0.1:                                                   # CLAHE(1-4, 8x8) on L
        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=float(R.uniform(1, 4)), tileGridSize=(8, 8)).apply(lab[..., 0])
        im = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if R.rand() < 0.1:                                                   # ChannelShuffle
        im = np.ascontiguousarray(im[..., R.permutation(3)])
    if R.rand() < 0.1:                                                   # Cutout(8 holes, 8 x 8)
        im = im.copy()
        for _ in range(8):
            y, x = R.randint(0, H), R.randint(0, W)
            im[max(0, y - 4):min(H, y + 4), max(0, x - 4):min(W, x + 4)] = 0
    if R.rand() < 0.3:                                                   # RandomGamma(80, 120)
        g = R.uniform(80, 120) / 100.0
        lut = (np.linspace(0, 1, 256) ** g * 255).round().astype(np.uint8)
        im = lut[im]
    if R.rand() < 0.3:                                                   # GlassBlur(0.7, 4, 2), fast mode
        sigma, md, it = 0.7, 4, 2
        x = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=sigma)
        hs, ws = np.arange(H - md, md, -1), np.arange(W - md, md, -1)
        hh, ww = np.tile(hs, ws.shape[0]), np.repeat(ws, hs.shape[0])
        dxy = R.randint(-md, md, size=(hh.shape[0], it, 2))
        for i in range(it):
            dy, dx = dxy[:, i, 0], dxy[:, i, 1]
            x[hh, ww], x[hh + dy, ww + dx] = x[hh + dy, ww + dx], x[hh, ww]
        im = cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=sigma)
    return im
