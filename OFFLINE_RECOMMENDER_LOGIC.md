# Logic hiện tại của Offline Recommender

## 1. Mục tiêu

Notebook `Offline_recommender.ipynb` xây dựng hệ thống gợi ý offline theo flow:

1. Setup môi trường và đường dẫn dữ liệu.
2. Load địa điểm trực tiếp từ `Places.csv`.
3. Tính Content-Based dựa trên embedding nội dung địa điểm.
4. Train Collaborative Filtering bằng Funk SVD.
5. Kết hợp CB + CF theo Union.
6. Re-rank kết quả bằng khoảng cách từ địa điểm đang xem.
7. Test, đánh giá RMSE và trích xuất dữ liệu mẫu ra CSV.

Rating được chuẩn hóa về thang 5 sao trước khi train và đánh giá.

## 2. Input

Hệ thống dùng 4 file đầu vào, đặt cùng thư mục `BASE`:

| File | Dùng cho | Nội dung |
|------|----------|----------|
| `Places.csv` | CB + metadata | Thông tin địa điểm: `id, name, latitude, longitude, category_name, type_name, city_name, ...` |
| `rating_matrix_foody.npz` | CF | Ma trận thưa SciPy CSR `(users × items)` chứa rating gốc |
| `rating_matrix_foody_users.csv` | CF | Danh sách `UserID` (số nguyên) theo từng hàng của ma trận |
| `rating_matrix_foody_items.csv` | CF | Danh sách place `id` (UUID) theo từng cột của ma trận |

**Lưu ý quan trọng về định danh:**

- Định danh địa điểm là cột **`id` dạng UUID chuỗi** (ví dụ `00003344-d133-5b88-8e68-8054e99f55bb`), không còn là số nguyên.
- `id` trong `rating_matrix_foody_items.csv` khớp với cột `id` trong `Places.csv`.
- `UserID` vẫn là số nguyên.

## 3. Thành phố lấy từ đâu?

Hệ thống **không dùng `city_id`**.

Tên thành phố được lấy trực tiếp từ cột `city_name` trong `Places.csv`.

## 4. Dữ liệu địa điểm

Notebook **đọc thẳng `Places.csv`** vào DataFrame `places` và dùng luôn các cột gốc của file — **không tạo `places_meta` và không ghi file `.pkl`** nữa (Places.csv đã đủ thông tin).

Chỉ chuẩn hóa nhẹ ngay trên các cột gốc (không đổi tên cột):

- `id`: ép về chuỗi, `strip`, bỏ rỗng, khử trùng lặp.
- `city_name`: `strip` qua `_normalize_city_name`.
- `latitude`, `longitude`: ép về số (`to_numeric`, lỗi → `NaN`).
- `name`, `category_name`, `type_name`: `fillna('')` + `strip`.

Các dict tra cứu (theo `id` chuỗi):

- `res2city`: `id` → `city_name`
- `res2lat`: `id` → `latitude`
- `res2lon`: `id` → `longitude`
- `res2content_idx`: `id` → chỉ số hàng embedding

## 5. Content-Based

Content-Based tạo text từ các cột gốc của `Places.csv`:

```text
name - city_name - category_name - type_name
```

Sau đó dùng model `paraphrase-multilingual-MiniLM-L12-v2` để encode embedding đã normalize (cosine = dot product). Embedding và lookup được cache riêng cho dữ liệu foody:

- `content_embeddings_foody.npy`
- `cb_lookup_foody.pkl`
- `embedding_model_foody.txt`

Cache chỉ được dùng lại nếu số dòng embedding khớp với `places` hiện tại, ngược lại sẽ encode lại.

Hàm CB hiện tại:

```python
get_top_k_items_by_item(item_id, city_name=None, k=50)
```

- `item_id`: `id` (UUID) của địa điểm đang xem.
- `city_name`: tên thành phố để lọc ứng viên.
- `k`: số lượng item cần lấy.

Nếu không truyền `city_name`, hàm tự lấy thành phố của `item_id` từ `res2city`. Trả về danh sách `id` (chuỗi).

## 6. Collaborative Filtering

Collaborative Filtering dùng Funk SVD từ thư viện `surprise`.

Rating được chuẩn hóa:

- `stars` gốc đã ở thang 0–5; nếu dữ liệu ở thang 10 thì chia 2.
- Sau đó clip về khoảng `[0.5, 5.0]`.

Hàm CF hiện tại:

```python
get_top_k_items_for_user_svd(user_id, city_name=None, k=50)
```

- `user_id`: UserID (số nguyên).
- `city_name`: tên thành phố cần lọc.
- `k`: số lượng item, mặc định 50.

Output là danh sách `id` (UUID) top 50 trong thành phố đó, đã loại các địa điểm user từng tương tác trong `R_csr`.

Các mapping chính:

- `user_id_to_cf_idx`: `UserID (int)` → chỉ số hàng.
- `cf_item_id_to_idx`: `id (str)` → chỉ số cột.
- `cf_item_ids`: mảng `id` (chuỗi) theo cột ma trận.

## 7. Hybrid Union + khoảng cách

Hàm chính:

```python
recommend_hybrid(user_id, current_item_id, city_name, k=10)
```

Input online hiện tại gồm:

- `user_id` (số nguyên)
- `current_item_id` (`id` dạng UUID chuỗi)
- `city_name`

Quy trình:

1. Lấy top CB theo `current_item_id` và `city_name`.
2. Lấy top CF theo `user_id` và `city_name`.
3. Union hai danh sách:
   - `BOTH`: item xuất hiện ở cả CB và CF.
   - `CB`: chỉ xuất hiện ở CB.
   - `CF`: chỉ xuất hiện ở CF.
4. Tính `ModelScore` dựa trên thứ hạng trong CB/CF. Item `BOTH` được cộng ưu tiên nhẹ.
5. Tính khoảng cách Haversine giữa địa điểm đang xem và địa điểm gợi ý.
6. Chuyển khoảng cách thành `DistanceScore`:

```python
DistanceScore = exp(-DistanceKm / distance_decay_km)
```

Địa điểm càng gần thì `DistanceScore` càng cao.

7. Tính điểm cuối:

```python
FinalScore = model_weight * ModelScore + distance_weight * DistanceScore
```

Mặc định:

- `model_weight = 0.75`
- `distance_weight = 0.25`
- `distance_decay_km = 5.0`

Kết quả được sort giảm dần theo `FinalScore`, `ModelScore`, `DistanceScore`.

## 8. Output của hàm gợi ý

`recommend_hybrid` trả về DataFrame dùng đúng tên cột gốc của `Places.csv`:

- `id`
- `name`
- `city_name`
- `latitude`
- `longitude`
- `category_name`
- `type_name`
- `DistanceKm`
- `ModelScore`
- `DistanceScore`
- `FinalScore`
- `Rank`
- `Source`

## 9. Lưu ý đánh giá

RMSE hiện dùng để đánh giá rating prediction:

- CF dùng model Funk SVD.
- CB dự đoán rating bằng item-based kNN trên embedding.
- CB chỉ dùng `R_train_csr` để tránh lấy rating validation/test làm lịch sử user.

Hybrid Union + khoảng cách là logic phục vụ ranking online, không phải trực tiếp tối ưu RMSE.

Nếu dùng `recommend_hybrid` để đánh giá ranking holdout nghiêm ngặt, nên đảm bảo phần loại bỏ item đã tương tác chỉ dùng lịch sử train, tránh vô tình dùng lịch sử validation/test.

## 10. Giả định dữ liệu

- `id` trong `rating_matrix_foody_items.csv` phải khớp với cột `id` trong `Places.csv`. Item nào không có trong `Places.csv` sẽ thiếu metadata (city/tọa độ) nên bị loại khỏi bước lọc theo thành phố — CF vẫn dự đoán được rating nhưng không lọt vào danh sách gợi ý theo `city_name`.
