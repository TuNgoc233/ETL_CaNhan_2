# Logic hiện tại của Offline Recommender

## 1. Mục tiêu

Notebook `Offline_recommender.ipynb` xây dựng hệ thống gợi ý offline theo flow:

1. Setup môi trường và đường dẫn dữ liệu.
2. Load và merge metadata địa điểm.
3. Tính Content-Based dựa trên embedding nội dung địa điểm.
4. Train Collaborative Filtering bằng Funk SVD.
5. Kết hợp CB + CF theo Union.
6. Re-rank kết quả bằng khoảng cách từ địa điểm đang xem.
7. Test, đánh giá RMSE và trích xuất dữ liệu mẫu ra CSV.

Rating được chuẩn hóa về thang 5 sao trước khi train và đánh giá.

## 2. Thành phố lấy từ đâu?

Hệ thống **không dùng `city_id`**.

Tên thành phố được lấy trực tiếp từ file:

```text
/content/drive/MyDrive/Recommender System/places_lookup.csv
```

File này có cấu trúc:

```text
ResID,PlaceName,City
1057377,Đức Luyện Mobile,Hải Dương
140097,Linh Nhi Spa - Bùi Thị Xuân,Hải Dương
```

Trong notebook, cột `City` được dùng như `city_name`.

## 3. Metadata địa điểm

Notebook merge metadata từ:

- `places_lookup.csv`: lấy `ResID`, `PlaceName`, `City`.
- `POI_DIR = BASE / 'DATN-ETL' / 'Filtered_Data_Foody'`: lấy category và tọa độ POI.
- `RES_DIR = BASE / 'DATN-ETL' / 'merged_foody_restaurant'`: lấy tọa độ nhà hàng.

Các trường chính trong `places_meta`:

- `ResID`
- `PlaceName`
- `City`
- `Latitude`
- `Longitude`
- `CategoryGroup`
- `CategorySub`
- `Kind`

Với POI, tọa độ thường nằm ở `latitude` và `longitude`.

Với nhà hàng, tọa độ thường nằm ở `Latitude` và `Longitude`.

## 4. Content-Based

Content-Based tạo text từ:

```text
PlaceName - City - CategoryGroup - CategorySub
```

Sau đó dùng model `paraphrase-multilingual-MiniLM-L12-v2` để encode embedding đã normalize.

Hàm CB hiện tại:

```python
get_top_k_items_by_item(item_id, city_name=None, k=50)
```

Ý nghĩa:

- `item_id`: địa điểm đang xem.
- `city_name`: tên thành phố dùng để lọc ứng viên.
- `k`: số lượng item cần lấy.

Nếu không truyền `city_name`, hàm sẽ tự lấy thành phố của `item_id` từ `places_lookup.csv`.

## 5. Collaborative Filtering

Collaborative Filtering dùng Funk SVD từ thư viện `surprise`.

Rating được chuẩn hóa:

- Nếu rating gốc lớn hơn 5 thì chia 2.
- Sau đó clip về khoảng `[0.5, 5.0]`.

Hàm CF hiện tại:

```python
get_top_k_items_for_user_svd(user_id, city_name=None, k=50)
```

Input:

- `user_id`: user cần gợi ý.
- `city_name`: tên thành phố cần lọc.
- `k`: số lượng item cần lấy, mặc định 50.

Output là danh sách `ResID` top 50 trong thành phố đó, đã loại các địa điểm user từng tương tác trong `R_csr`.

## 6. Hybrid Union + khoảng cách

Hàm chính:

```python
recommend_hybrid(user_id, current_item_id, city_name, k=10)
```

Input online hiện tại gồm:

- `user_id`
- `current_item_id`
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

Kết quả được sort giảm dần theo:

1. `FinalScore`
2. `ModelScore`
3. `DistanceScore`

## 7. Output của hàm gợi ý

`recommend_hybrid` trả về DataFrame gồm:

- `ResID`
- `PlaceName`
- `City`
- `Latitude`
- `Longitude`
- `CategoryGroup`
- `CategorySub`
- `DistanceKm`
- `ModelScore`
- `DistanceScore`
- `FinalScore`
- `Rank`
- `Source`

## 8. Lưu ý đánh giá

RMSE hiện dùng để đánh giá rating prediction:

- CF dùng model Funk SVD.
- CB dự đoán rating bằng item-based kNN trên embedding.
- CB chỉ dùng `R_train_csr` để tránh lấy rating validation/test làm lịch sử user.

Hybrid Union + khoảng cách là logic phục vụ ranking online, không phải trực tiếp tối ưu RMSE.

Nếu dùng `recommend_hybrid` để đánh giá ranking holdout nghiêm ngặt, nên đảm bảo phần loại bỏ item đã tương tác chỉ dùng lịch sử train, tránh vô tình dùng lịch sử validation/test.
