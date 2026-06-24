"""Cấu hình cho Phần 1 (indexer). Đọc từ biến môi trường / file .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    """Lấy biến môi trường bắt buộc, báo lỗi rõ ràng nếu thiếu."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Thiếu biến môi trường bắt buộc: {name}. "
            f"Xem file .env.example để biết cách cấu hình."
        )
    return value


@dataclass
class IndexerConfig:
    # --- Google Drive ---
    drive_folder_id: str           # bắt buộc
    service_account_file: str      # optional khi chạy Colab dùng OAuth user
    # --- Đầu ra ---
    output_dir: str
    # --- Model khuôn mặt ---
    face_model: str
    det_size: int
    ctx_id: int            # >=0: GPU, -1: CPU
    min_det_score: float
    # --- OCR số BIB ---
    enable_bib: bool
    bib_lang: str
    bib_min_conf: float
    bib_min_len: int
    bib_max_len: int
    # --- Phát hiện vùng BIB (YOLO) ---
    bib_use_detector: bool
    bib_detector_weights: str
    bib_detector_conf: float
    bib_detector_is_custom: bool

    @classmethod
    def from_env(cls) -> "IndexerConfig":
        return cls(
            drive_folder_id=_require("DRIVE_FOLDER_ID"),
            service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", ""),
            output_dir=os.getenv("INDEX_OUTPUT_DIR", "data/index"),
            face_model=os.getenv("FACE_MODEL", "buffalo_l"),
            det_size=int(os.getenv("FACE_DET_SIZE", "640")),
            ctx_id=int(os.getenv("FACE_CTX_ID", "0")),
            min_det_score=float(os.getenv("FACE_MIN_DET_SCORE", "0.5")),
            enable_bib=os.getenv("ENABLE_BIB", "true").lower() == "true",
            bib_lang=os.getenv("BIB_LANG", "en"),
            bib_min_conf=float(os.getenv("BIB_MIN_CONF", "0.6")),
            bib_min_len=int(os.getenv("BIB_MIN_LEN", "2")),
            bib_max_len=int(os.getenv("BIB_MAX_LEN", "6")),
            bib_use_detector=os.getenv("BIB_USE_DETECTOR", "true").lower() == "true",
            bib_detector_weights=os.getenv("BIB_DETECTOR_WEIGHTS", "yolov8n.pt"),
            bib_detector_conf=float(os.getenv("BIB_DETECTOR_CONF", "0.25")),
            bib_detector_is_custom=os.getenv("BIB_DETECTOR_IS_CUSTOM", "false").lower()
            == "true",
        )
