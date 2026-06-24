"""Pipeline chính của App 1 (indexer).

Quét toàn bộ ảnh trong folder Google Drive → phát hiện khuôn mặt (embedding) và
đọc số BIB → ghi ra index dùng cho App 2.

Chạy (từ trong thư mục app1_indexer/):
    python build_index.py            # xử lý toàn bộ
    python build_index.py --limit 50 # xử lý thử 50 ảnh đầu

Xem đặc tả đầu ra tại index_format.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from bib_detector import BibRegionDetector
from bib_engine import BibEngine
from config import IndexerConfig
from drive_source import DriveImage, DriveSource
from face_engine import FaceEngine

_EMBED_DIM = 512


def _decode_image(raw: bytes) -> np.ndarray | None:
    """Giải mã bytes ảnh sang ndarray BGR. Trả về None nếu hỏng."""
    array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _build_bib_engine(config: IndexerConfig) -> BibEngine:
    detector = (
        BibRegionDetector(
            weights=config.bib_detector_weights,
            conf=config.bib_detector_conf,
            is_custom=config.bib_detector_is_custom,
        )
        if config.bib_use_detector
        else None
    )
    return BibEngine(
        lang=config.bib_lang,
        min_conf=config.bib_min_conf,
        min_len=config.bib_min_len,
        max_len=config.bib_max_len,
        detector=detector,
    )


def build_index(
    config: IndexerConfig,
    limit: int | None = None,
    drive: DriveSource | None = None,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if drive is None:
        drive = DriveSource(service_account_file=config.service_account_file)
    face_engine = FaceEngine(
        model_name=config.face_model,
        det_size=config.det_size,
        ctx_id=config.ctx_id,
        min_det_score=config.min_det_score,
    )
    bib_engine = _build_bib_engine(config) if config.enable_bib else None

    print(f"Đang liệt kê ảnh trong folder Drive {config.drive_folder_id} ...")
    images = drive.list_images(config.drive_folder_id)
    if limit is not None:
        images = images[:limit]
    print(f"Tìm thấy {len(images)} ảnh. Bắt đầu xử lý.")

    embeddings: list[np.ndarray] = []
    face_rows: list[dict] = []
    image_rows: list[dict] = []
    bib_lookup: dict[str, list[str]] = {}
    failed = 0

    for image in tqdm(images, desc="Xử lý ảnh", unit="ảnh"):
        try:
            faces, bibs = _process_one(image, drive, face_engine, bib_engine)
        except Exception as exc:  # bỏ qua ảnh lỗi, không dừng cả pipeline
            failed += 1
            tqdm.write(f"[Lỗi] {image.name} ({image.file_id}): {exc}")
            continue

        for face in faces:
            face_rows.append(
                {
                    "face_idx": len(embeddings),
                    "image_id": image.file_id,
                    "bbox": face.bbox,
                    "det_score": face.det_score,
                }
            )
            embeddings.append(face.embedding)

        image_rows.append(
            {
                "image_id": image.file_id,
                "drive_file_id": image.file_id,
                "filename": image.name,
                "view_url": image.view_url,
                "download_url": image.download_url,
                "num_faces": len(faces),
                "bib_numbers": bibs,
            }
        )
        for bib in bibs:
            bib_lookup.setdefault(bib, []).append(image.file_id)

    _write_outputs(output_dir, embeddings, face_rows, image_rows, bib_lookup)
    print(
        f"Xong. {len(image_rows)} ảnh, {len(embeddings)} khuôn mặt, "
        f"{len(bib_lookup)} số BIB. Bỏ qua {failed} ảnh lỗi."
    )
    print(f"Index đã ghi vào: {output_dir.resolve()}")


def _process_one(
    image: DriveImage,
    drive: DriveSource,
    face_engine: FaceEngine,
    bib_engine: BibEngine | None,
):
    raw = drive.download_bytes(image.file_id)
    img_bgr = _decode_image(raw)
    if img_bgr is None:
        raise ValueError("không giải mã được ảnh")
    faces = face_engine.extract(img_bgr)
    bibs = bib_engine.read_bibs(img_bgr) if bib_engine is not None else []
    return faces, bibs


def _write_outputs(
    output_dir: Path,
    embeddings: list[np.ndarray],
    face_rows: list[dict],
    image_rows: list[dict],
    bib_lookup: dict[str, list[str]],
) -> None:
    index = faiss.IndexFlatIP(_EMBED_DIM)
    if embeddings:
        matrix = np.vstack(embeddings).astype(np.float32)
        index.add(matrix)
    faiss.write_index(index, str(output_dir / "faces.faiss"))

    pd.DataFrame(
        face_rows, columns=["face_idx", "image_id", "bbox", "det_score"]
    ).to_parquet(output_dir / "faces_meta.parquet", index=False)

    pd.DataFrame(
        image_rows,
        columns=[
            "image_id",
            "drive_file_id",
            "filename",
            "view_url",
            "download_url",
            "num_faces",
            "bib_numbers",
        ],
    ).to_parquet(output_dir / "images.parquet", index=False)

    (output_dir / "bib_lookup.json").write_text(
        json.dumps(bib_lookup, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo index khuôn mặt + BIB từ Google Drive.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu (để chạy thử)."
    )
    args = parser.parse_args()
    build_index(IndexerConfig.from_env(), limit=args.limit)


if __name__ == "__main__":
    main()
