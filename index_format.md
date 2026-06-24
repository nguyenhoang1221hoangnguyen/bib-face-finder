# Đặc tả định dạng Index (hợp đồng giữa Phần 1 và Phần 2)

Phần 1 (indexer) **ghi ra** thư mục `data/index/`. Phần 2 (web app) **đọc** đúng thư mục này.
Không thay đổi định dạng ở một phía mà không cập nhật phía còn lại.

## Thư mục `data/index/`

| File | Định dạng | Nội dung |
|------|-----------|----------|
| `faces.faiss` | FAISS `IndexFlatIP` | Vector embedding khuôn mặt, **đã chuẩn hoá L2** (dim = 512). Inner product = cosine similarity. |
| `faces_meta.parquet` | Parquet | Mỗi dòng = 1 khuôn mặt. Thứ tự dòng **trùng** vị trí vector trong FAISS. |
| `images.parquet` | Parquet | Mỗi dòng = 1 ảnh gốc trên Drive. |
| `bib_lookup.json` | JSON | `{ "<số_bib>": ["<image_id>", ...] }` — tra ảnh theo số BIB không cần model. |

### `faces_meta.parquet`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `face_idx` | int64 | Vị trí của vector trong `faces.faiss` (0-based). |
| `image_id` | string | Khoá ngoại trỏ về `images.image_id`. |
| `bbox` | list[float] | `[x1, y1, x2, y2]` toạ độ khung mặt trên ảnh gốc. |
| `det_score` | float32 | Điểm tin cậy của bộ phát hiện mặt. |

### `images.parquet`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `image_id` | string | Định danh ảnh = Google Drive file id. |
| `drive_file_id` | string | Drive file id (giống `image_id`). |
| `filename` | string | Tên file gốc trên Drive. |
| `view_url` | string | Link xem/preview (Drive thumbnail). |
| `download_url` | string | Link tải về trực tiếp. |
| `num_faces` | int32 | Số khuôn mặt phát hiện trong ảnh. |
| `bib_numbers` | list[string] | Danh sách số BIB đọc được trong ảnh (có thể rỗng). |

## Quy ước

- **Embedding chuẩn hoá L2** ở cả lúc index và lúc query → cosine = inner product.
- `image_id == drive_file_id` (duy nhất theo Drive).
- Một ảnh nhiều mặt → nhiều dòng trong `faces_meta`, cùng `image_id`.
- Link Drive (ảnh phải ở chế độ public/anyone-with-link):
  - `view_url`  = `https://drive.google.com/thumbnail?id=<id>&sz=w1000`
  - `download_url` = `https://drive.google.com/uc?export=download&id=<id>`
