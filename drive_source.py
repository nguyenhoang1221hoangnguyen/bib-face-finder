"""Đọc ảnh từ Google Drive qua Drive API v3.

Hỗ trợ 2 cách xác thực:
  1. **Service account JSON** (chạy local/headless): folder phải share cho email service account.
  2. **Credentials có sẵn** (Colab OAuth user): tự xác thực bằng tài khoản Google của bạn,
     folder chỉ cần thuộc về bạn hoặc đã share cho bạn.

Liệt kê đệ quy mọi ảnh trong một folder (kể cả folder con) và tải nội dung dạng bytes.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_FOLDER_MIME = "application/vnd.google-apps.folder"


@dataclass
class DriveImage:
    file_id: str
    name: str
    mime_type: str

    @property
    def view_url(self) -> str:
        return f"https://drive.google.com/thumbnail?id={self.file_id}&sz=w1000"

    @property
    def download_url(self) -> str:
        return f"https://drive.google.com/uc?export=download&id={self.file_id}"


class DriveSource:
    def __init__(
        self,
        service_account_file: str | None = None,
        credentials: Any = None,
    ) -> None:
        """Truyền `credentials` (object đã build sẵn) hoặc `service_account_file`."""
        if credentials is None:
            if not service_account_file:
                raise ValueError(
                    "Cần truyền service_account_file hoặc credentials cho DriveSource."
                )
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file, scopes=_SCOPES
            )
        self._service = build("drive", "v3", credentials=credentials)

    def list_images(self, folder_id: str) -> list[DriveImage]:
        """Liệt kê đệ quy tất cả file ảnh trong folder và các folder con."""
        images: list[DriveImage] = []
        self._walk_folder(folder_id, images)
        return images

    def _walk_folder(self, folder_id: str, images: list[DriveImage]) -> None:
        page_token: str | None = None
        while True:
            response = (
                self._service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for item in response.get("files", []):
                mime = item["mimeType"]
                if mime == _FOLDER_MIME:
                    self._walk_folder(item["id"], images)
                elif mime.startswith("image/"):
                    images.append(
                        DriveImage(
                            file_id=item["id"], name=item["name"], mime_type=mime
                        )
                    )
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def download_bytes(self, file_id: str) -> bytes:
        """Tải nội dung một file về dạng bytes."""
        request = self._service.files().get_media(
            fileId=file_id, supportsAllDrives=True
        )
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()
