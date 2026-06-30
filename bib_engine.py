"""Đọc số BIB trong ảnh bằng PaddleOCR.

- Nếu được cấp `BibRegionDetector`: chỉ OCR các vùng nghi chứa bib (torso người, hoặc
  box từ model bib tự train) → ít nhiễu, chính xác hơn. Khi detector không tìm thấy
  vùng nào thì tự động fallback OCR toàn ảnh.
- Nếu không có detector: OCR toàn ảnh và lọc token chữ số.

Token được giữ lại nếu độ tin cậy OCR đủ cao và độ dài hợp lệ (bib thường 2–6 chữ số).

Tối ưu chất lượng (không cần train):
- Upscale crop nhỏ lên tối thiểu 800px cạnh dài (bicubic) trước khi OCR.
- CLAHE trên kênh L (LAB) để tăng tương phản cục bộ, giúp đọc bib trong bóng/ngược nắng.
- Dùng PP-OCRv4 + nới `det_db_*` để bắt text mờ/nhỏ.
- Lọc token nhiễu phổ biến (năm sự kiện, toàn 0).
"""
from __future__ import annotations

import re

import cv2
import numpy as np
from paddleocr import PaddleOCR

from bib_detector import BibRegionDetector

_DIGITS = re.compile(r"\d+")
_MIN_OCR_SIDE = 800
_NOISE_TOKENS = {"2023", "2024", "2025", "2026", "2027"}


class BibEngine:
    def __init__(
        self,
        lang: str = "en",
        min_conf: float = 0.6,
        min_len: int = 2,
        max_len: int = 6,
        detector: BibRegionDetector | None = None,
    ) -> None:
        self._min_conf = min_conf
        self._min_len = min_len
        self._max_len = max_len
        self._detector = detector
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            ocr_version="PP-OCRv4",
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.8,
        )

    def read_bibs(self, img_bgr: np.ndarray) -> list[str]:
        """Trả về danh sách số BIB (chuỗi chữ số) đọc được, đã loại trùng và sắp xếp."""
        targets = [img_bgr]
        if self._detector is not None:
            regions = self._detector.detect_regions(img_bgr)
            if regions:  # không thấy vùng nào → giữ fallback toàn ảnh
                targets = regions

        bibs: set[str] = set()
        for target in targets:
            prepared = _prepare_for_ocr(target)
            bibs.update(self._ocr_numbers(prepared))
        return sorted(bibs)

    def _ocr_numbers(self, img_bgr: np.ndarray) -> set[str]:
        result = self._ocr.ocr(img_bgr, cls=True)
        if not result or not result[0]:
            return set()

        numbers: set[str] = set()
        for line in result[0]:
            text, confidence = line[1]
            if float(confidence) < self._min_conf:
                continue
            for token in _DIGITS.findall(text):
                if not (self._min_len <= len(token) <= self._max_len):
                    continue
                if token in _NOISE_TOKENS or set(token) == {"0"}:
                    continue
                numbers.add(token)
        return numbers


def _prepare_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Tăng chất lượng ảnh trước OCR: upscale nếu nhỏ, CLAHE tăng tương phản."""
    h, w = img_bgr.shape[:2]
    if max(h, w) < _MIN_OCR_SIDE:
        scale = _MIN_OCR_SIDE / max(h, w)
        img_bgr = cv2.resize(
            img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
