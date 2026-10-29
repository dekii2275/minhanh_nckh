# Dữ liệu công bố, code snapshot và tái lập

Cập nhật 29/10/2026. Lần cập nhật này viết lại cách diễn giải và công bố bằng chứng đã chạy ở Bước 1–3; không train hoặc đánh giá official test mới.

Bộ bằng chứng gồm:

- `outputs/alignment_step1/`: mẫu 30 ID, review từng clip, screening từng frame, snapshot heuristic, hình GT/YOLO ba mapping và provenance.
- `outputs/alignment_step2/`: 253 dự đoán/video cho C1–C4, posterior CSV 15 lớp, metric, CI, cờ swap, split, checkpoint hash và kiểm tra tái lập GT.
- `outputs/calibration_step3/`: 8 dự đoán/video A–H, posterior CSV, 12 marginal contrasts, 7 interaction contrasts, 28 pairwise contrasts, danh sách đổi dự đoán, transformation diagnostics và kiểm chứng A/H.
- `outputs/research_snapshot/src/` và `tests/`: bản chụp nguyên byte code dùng trong các thực nghiệm, không thay đổi code production. [source_manifest.json](../outputs/research_snapshot/source_manifest.json) lưu SHA256 theo đường dẫn gốc. Có thể đọc code trực tiếp trên GitHub.

Repository đang có những sửa đổi source từ các công việc khác chưa được commit cùng đợt tài liệu. Vì thế bản chụp ghi chính xác source đã dùng khi đo, bao gồm model/metric helper của working tree. Không tự coi source ở `src/` trên một checkout bất kỳ là cùng phiên bản với các hash thực nghiệm.

Để chạy lại mô hình, dùng một **bản sao/checkout riêng**, khôi phục thư mục `src` và `tests` từ snapshot vào đúng đường dẫn gốc. Ví dụ khi đang đứng ở root của bản sao dành cho replay:

```bash
cp -R outputs/research_snapshot/src/. src/
cp -R outputs/research_snapshot/tests/. tests/
```

Các lệnh tái lập trong report Bước 1–3 sau đó dùng các đường dẫn `src/...` gốc. Snapshot là source thực nghiệm, không phải module cài đặt trực tiếp bằng đường dẫn nằm trong `outputs/`. Các script tạo report có thể tạo lại link tới source gốc; những report viết lại và liên kết snapshot là bản diễn giải công bố hiện tại.

Cần cung cấp cục bộ Penn Action, trọng số YOLO và checkpoint BiLSTM đúng hash, cùng cache raw nếu muốn bỏ qua inference. Dataset, file `.pt`, `.npz` và cache lớn vẫn ngoài Git theo `.gitignore`. Không force-add checkpoint hoặc ảnh dataset gốc trong đợt này. Hình skeleton audit đã có overlay được công bố phục vụ đối chiếu; không thay thế dataset.

Bước 2/3 dùng checkpoint:

```text
outputs/action_recognition/protocol_v2/multi_seed/baseline/seed0/best_bilstm_model.pt
SHA256 dc01f24210e57ccb7a52611b0b37fe91a40b35e938718fdb123847267fb74347
```

Thư viện thực nghiệm: PyTorch 2.6.0+cu124, Ultralytics 8.4.183, NumPy 2.2.6, pandas 2.3.3; replay classifier CPU với 8 luồng như lúc YOLO cấu hình ở Bước 2. Chi tiết còn lại ở các provenance JSON. Không hứa tái lập từng bit trên phiên bản/phần cứng khác.

Không cần dataset, checkpoint hay PyTorch để kiểm tra bảng công bố từ CSV:

```bash
python outputs/research_snapshot/verify_published_tables.py
```

Script này dùng thư viện chuẩn Python để tính lại Top-1, Macro-F1 và số sai→đúng/đúng→sai từ dự đoán Bước 2–3, xác minh 253 ID và cờ swap trùng nhau, A/H khớp C3/C4. Nó không chạy lại classifier hoặc bootstrap. Các verifier snapshot thực nghiệm có kiểm tra mạnh hơn trên cache cục bộ, gồm posterior và bootstrap; JSON kiểm chứng được lưu cùng artifact.

Các result JSON nhiều seed, KD, AQA và J-HMDB là artifact **lịch sử được đọc lại**, không phải phép xác nhận mới. Report riêng ghi rõ phạm vi và các diễn giải rút lại. Bản review khoa học trước audit được giữ như tài liệu lịch sử có chú thích trạng thái mới. Không coi các report phụ là bằng chứng độc lập xác nhận đóng góp calibration.
