# Phạm vi công bố và tái lập dữ liệu cục bộ

Nhánh `docs/research-audit-reports` chỉ công bố README và các report Markdown. Không đẩy CSV mới, hình audit mới, code snapshot, dataset, checkpoint hoặc cache lên GitHub trong đợt tài liệu này. Bảng metric, CI, ID và diễn giải nằm trong các report; các đường dẫn bằng mã trong report chỉ bằng chứng đang giữ ở working tree cục bộ.

Các file dữ liệu lịch sử đã có trên nhánh gốc vẫn có thể mở bằng liên kết tương đối. Với bằng chứng mới chưa có trên GitHub, report hiển thị đường dẫn cục bộ thay cho liên kết hỏng. Không coi việc có đường dẫn là artifact đã được công bố.

Working tree cục bộ đã lưu đầy đủ:

- `outputs/alignment_step1/`: 30 ID, review từng clip, screening từng frame, hình GT/YOLO và provenance.
- `outputs/alignment_step2/`: dự đoán C1–C4 trên 253 video, posterior CSV/npz, metric, paired CI, cờ swap và protocol.
- `outputs/calibration_step3/`: dự đoán A–H, CI, 12 marginal contrasts, 7 interaction contrasts, 28 pairwise contrasts, danh sách 42 video đổi nhãn và kiểm chứng tái lập A/H.
- `outputs/research_snapshot/`: source nguyên byte dùng trong thực nghiệm, SHA256 manifest và script kiểm tra bảng chỉ dùng Python chuẩn. Bản chụp tách riêng với các thay đổi source huấn luyện chưa công bố.

Tại working tree cục bộ có đủ bằng chứng, kiểm tra bảng bằng:

```bash
python outputs/research_snapshot/verify_published_tables.py
```

Script tính lại Top-1/Macro-F1 và số sai→đúng/đúng→sai, kiểm tra cùng 253 ID và cờ swap, A=C3/H=C4; không cần dataset hoặc chạy classifier. Lệnh này chưa chạy được trên một clone chỉ có tài liệu vì CSV và code snapshot mới chưa được công bố.

Để replay classifier ở máy có đủ dữ liệu, dùng một bản sao riêng; khôi phục `src` và `tests` từ snapshot vào đường dẫn gốc rồi dùng lệnh trong report Bước 1–3. Snapshot không phải module chạy trực tiếp từ bên trong `outputs/`. Cần Penn Action, trọng số YOLO, checkpoint GT-trained đúng hash và cache raw tương ứng nếu bỏ qua inference.

```text
outputs/action_recognition/protocol_v2/multi_seed/baseline/seed0/best_bilstm_model.pt
SHA256 dc01f24210e57ccb7a52611b0b37fe91a40b35e938718fdb123847267fb74347
```

Môi trường đã đo: PyTorch 2.6.0+cu124, Ultralytics 8.4.183, NumPy 2.2.6, pandas 2.3.3, classifier CPU 8 luồng. Không hứa tái lập từng bit trên phiên bản/phần cứng khác. Các verifier thực nghiệm cục bộ đã kiểm tra metric, posterior A/H, bootstrap và hash trước lần viết lại tài liệu.

Không có thực nghiệm train/test mới khi viết lại report. Dữ liệu nhiều seed, KD, AQA và J-HMDB được đọc từ artifact lịch sử và ghi rõ giới hạn; không thay thế đánh giá độc lập tiếp theo.
