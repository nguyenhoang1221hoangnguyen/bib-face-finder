# bib-face-finder — Indexer (App 1)

Pipeline tạo index khuôn mặt + số BIB từ ảnh trên Google Drive.

- **InsightFace** (`buffalo_l`) — phát hiện mặt + sinh embedding 512-d (chuẩn hoá L2).
- **PaddleOCR** — đọc số BIB; có thể kết hợp **YOLOv8** để khoanh vùng torso trước khi OCR.
- **Google Drive API** — đọc đệ quy mọi ảnh trong folder.
- Xuất ra `data/index/` với 4 file (FAISS + parquet + JSON). Web app đọc đúng định dạng này
  (xem `index_format.md`).

## Cách nhanh nhất — chạy trên Colab (không cần service account)

Mở `colab_indexer.ipynb` trên Colab → Runtime → GPU → chạy lần lượt 5 cell code:

1. Clone repo + cài thư viện.
2. Dán `DRIVE_FOLDER_ID` + bấm cho phép Drive.
3. Chạy thử 50 ảnh.
4. Chạy full.
5. Tải `index.zip` về.

## Chạy local (cần service account)

```bash
cp .env.example .env       # điền DRIVE_FOLDER_ID + GOOGLE_SERVICE_ACCOUNT_FILE
pip install -r requirements.txt
python build_index.py --limit 50
python build_index.py
```

Chi tiết: xem `index_format.md` và `.env.example`.
