"""Bộ trích xuất khuôn mặt dùng chung cho Phần 1 (indexer) và Phần 2 (web app).

Bọc InsightFace (ArcFace `buffalo_l`): phát hiện mặt + sinh embedding 512 chiều
đã chuẩn hoá L2. Nhờ chuẩn hoá L2, inner product giữa hai embedding = cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class DetectedFace:
    """Một khuôn mặt được phát hiện trong ảnh."""

    bbox: list[float]          # [x1, y1, x2, y2]
    det_score: float           # độ tin cậy phát hiện
    embedding: np.ndarray      # vector 512-d, đã chuẩn hoá L2 (float32)


class FaceEngine:
    """Phát hiện khuôn mặt và sinh embedding."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        ctx_id: int = 0,
        min_det_score: float = 0.5,
    ) -> None:
        """ctx_id >= 0 dùng GPU tương ứng, ctx_id = -1 dùng CPU."""
        self._min_det_score = min_det_score
        self._app = FaceAnalysis(name=model_name)
        self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def extract(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        """Trả về toàn bộ khuôn mặt vượt ngưỡng tin cậy trong ảnh (định dạng BGR)."""
        faces = self._app.get(img_bgr)
        result: list[DetectedFace] = []
        for face in faces:
            if float(face.det_score) < self._min_det_score:
                continue
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            result.append(
                DetectedFace(
                    bbox=[float(v) for v in face.bbox],
                    det_score=float(face.det_score),
                    embedding=embedding,
                )
            )
        return result

    def embed_largest(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Dùng cho ảnh selfie người dùng upload: trả về embedding của khuôn mặt lớn nhất.

        Trả về None nếu không phát hiện khuôn mặt nào.
        """
        faces = self.extract(img_bgr)
        if not faces:
            return None
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces[0].embedding
