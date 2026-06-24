"""Đọc số BIB trong ảnh bằng PaddleOCR.

- Nếu được cấp `BibRegionDetector`: chỉ OCR các vùng nghi chứa bib (torso người, hoặc
  box từ model bib tự train) → ít nhiễu, chính xác hơn. Khi detector không tìm thấy
  vùng nào thì tự động fallback OCR toàn ảnh.
- Nếu không có detector: OCR toàn ảnh và lọc token chữ số.

Token được giữ lại nếu độ tin cậy OCR đủ cao và độ dài hợp lệ (bib thường 2–6 chữ số).
"""
from __future__ import annotations

import re

import numpy as np
from paddleocr import PaddleOCR

from bib_detector import BibRegionDetector

_DIGITS = re.compile(r"\d+")


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
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def read_bibs(self, img_bgr: np.ndarray) -> list[str]:
        """Trả về danh sách số BIB (chuỗi chữ số) đọc được, đã loại trùng và sắp xếp."""
        targets = [img_bgr]
        if self._detector is not None:
            regions = self._detector.detect_regions(img_bgr)
            if regions:  # không thấy vùng nào → giữ fallback toàn ảnh
                targets = regions

        bibs: set[str] = set()
        for target in targets:
            bibs.update(self._ocr_numbers(target))
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
                if self._min_len <= len(token) <= self._max_len:
                    numbers.add(token)
        return numbers
