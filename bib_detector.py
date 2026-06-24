"""Phát hiện vùng chứa số BIB trong ảnh bằng YOLOv8.

Hai chế độ:

1. **Mặc định (không cần train)**: dùng model phát hiện người `yolov8n.pt` (tự tải),
   sau đó khoanh vùng thân trên (torso) — nơi số bib thường được gắn — để OCR.
   Giảm nhiễu từ nền, biển quảng cáo, số áo... so với OCR toàn ảnh.

2. **Model BIB tự train (tùy chọn)**: nếu bạn fine-tune YOLO để phát hiện trực tiếp
   vùng bib, đặt `is_custom=True` và trỏ `weights` tới file .pt đó; khi đó box phát
   hiện được dùng nguyên làm vùng OCR.
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

_PERSON_CLASS = 0  # class "person" trong COCO

# Vùng torso theo tỉ lệ chiều cao bbox người (bib thường nằm ngực–bụng).
_TORSO_TOP = 0.25
_TORSO_BOTTOM = 0.75


class BibRegionDetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        is_custom: bool = False,
    ) -> None:
        self._model = YOLO(weights)
        self._conf = conf
        self._is_custom = is_custom

    def detect_regions(self, img_bgr: np.ndarray) -> list[np.ndarray]:
        """Trả về danh sách ảnh crop (BGR) là các vùng nghi chứa số BIB.

        Rỗng nếu không phát hiện được vùng nào → bên gọi nên fallback OCR toàn ảnh.
        """
        height, width = img_bgr.shape[:2]
        results = self._model(img_bgr, conf=self._conf, verbose=False)

        crops: list[np.ndarray] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                if self._is_custom:
                    region = (x1, y1, x2, y2)
                else:
                    if int(box.cls[0]) != _PERSON_CLASS:
                        continue
                    region = self._torso_region(x1, y1, x2, y2)
                crop = self._crop(img_bgr, region, width, height)
                if crop is not None:
                    crops.append(crop)
        return crops

    @staticmethod
    def _torso_region(
        x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float, float, float]:
        person_height = y2 - y1
        return (x1, y1 + _TORSO_TOP * person_height, x2, y1 + _TORSO_BOTTOM * person_height)

    @staticmethod
    def _crop(
        img_bgr: np.ndarray,
        region: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> np.ndarray | None:
        x1 = max(0, int(region[0]))
        y1 = max(0, int(region[1]))
        x2 = min(width, int(region[2]))
        y2 = min(height, int(region[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img_bgr[y1:y2, x1:x2]
        return crop if crop.size else None
