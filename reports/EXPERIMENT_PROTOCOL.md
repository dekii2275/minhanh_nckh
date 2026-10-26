# Quy trình thực nghiệm và trạng thái kết quả

## Quy trình Penn Action phiên bản 2

Baseline và bốn thử nghiệm nhận diện hành động dùng cùng một quy trình:

1. Giữ nguyên tập test chính thức, tách validation theo từng lớp từ tập train.
   Đơn vị chia là video, không phải frame. Mặc định `seed=42`, tỷ lệ validation 20%.
2. Kiểm tra ID video duy nhất, train/test không giao nhau và ba modality test có
   cùng ID cùng nhãn. Dừng với lỗi rõ ràng nếu cache không đáp ứng điều kiện.
3. Chỉ dùng validation GT để chọn checkpoint. Các phép biến đổi tăng cường dữ
   liệu chỉ áp dụng cho train.
4. Nạp checkpoint tốt nhất và đánh giá một lần trên mỗi modality test: GT,
   YOLO thô và YOLO hiệu chỉnh.
5. Lưu seed, tỷ lệ validation, danh sách ID của từng tập, epoch tốt nhất và
   validation accuracy trong JSON kết quả.

Với cache hiện tại, cấu hình mặc định tạo **1.005 video train / 253 video
validation / 1.068 video test**. Các lớp ít hơn hai video train bị từ chối vì
không thể xuất hiện ở cả train và validation.

Baseline đánh giá validation mỗi epoch; các thử nghiệm đánh giá ở epoch 1,
mỗi 5 epoch và epoch cuối. Điểm khởi tạo chọn checkpoint là -1 để checkpoint
đầu tiên luôn được lưu, kể cả accuracy bằng 0.

Kết quả mới mặc định nằm trong:

- `outputs/action_recognition/protocol_v2/`
- `outputs/action_recognition/experiments/protocol_v2/`

Có thể thay đường dẫn bằng `--output_dir`; dùng thư mục riêng cho từng seed.
Cache cũ vẫn được đọc nhưng không bị thay đổi bởi bước chia validation.

## Cách chạy lại

Chạy các lệnh từ thư mục gốc repository, trong môi trường đã cài dependencies
và có dữ liệu/cache theo README:

```bash
python src/train_action_recognition.py --epochs 50 --seed 42 --validation_fraction 0.2
python src/run_action_experiments.py --mode exp1_kinematics --epochs 50 --seed 42
python src/run_action_experiments.py --mode exp2_augmentation --epochs 50 --seed 42
python src/run_action_experiments.py --mode exp3_spatiotemporal --epochs 50 --seed 42
python src/run_action_experiments.py --mode exp4_combined --epochs 50 --seed 42
```

Để đo độ biến thiên, chạy nhiều seed và lưu mỗi lần vào thư mục khác nhau:

```bash
python src/train_action_recognition.py --seed 7 --output_dir outputs/action_recognition/protocol_v2/seed7
```

Seed giúp tái lập cách chia dữ liệu và khởi tạo ngẫu nhiên; không đảm bảo kết
quả giống từng bit giữa các GPU, phiên bản thư viện hoặc hệ điều hành.

## Giới hạn của kết quả đã lưu trước phiên bản 2

Các JSON và hình cũ được giữ để truy vết. Chúng **chưa phải kết quả chạy lại
với quy trình validation mới**. Code Penn Action trước đây chọn checkpoint
dựa vào accuracy trên test GT, nên cần chạy lại trước khi dùng số liệu để
khẳng định khả năng tổng quát hóa.

Một số nội dung trong báo cáo tổng hợp chưa khớp artifact gốc:

| Nội dung | Giá trị trong artifact |
| --- | --- |
| Penn baseline Top-1, raw → calibrated | 45,51% → 67,23% (+21,72 điểm phần trăm), theo quy trình cũ |
| Penn baseline các chỉ số top-k đã lưu | Top-1 và Top-3; JSON baseline không có Top-5 |
| Penn jitter, raw → full combined | 19,72 → 9,88 px, theo `outputs/calibration/ablation_summary.csv` |
| Penn bone-length std, raw → full combined | 7,32 → 5,64 px |
| Penn PCK@20px, chỉ căn trái/phải → full combined | 64,06% → 63,21%; thêm các module không tăng chỉ số này |
| J-HMDB, GT-trained: raw → calibrated Top-1 | 40,30% → 37,69% |
| J-HMDB, calibrated-trained: raw → calibrated Top-1 | 46,27% → 44,03% |

Do đó, 46,27% trên J-HMDB là kết quả **test raw** của mô hình được train trên
calibrated, không phải test calibrated. Hiệu chỉnh làm mượt hơn chưa chứng
minh được nhận diện hành động tốt hơn trên J-HMDB. Thử nghiệm J-HMDB hiện
train và test trong split 1 của chính J-HMDB; đây là kiểm tra trên bộ dữ liệu
thứ hai, chưa phải chuyển mô hình train trên Penn sang J-HMDB.

Con số Penn domain-matched 88,81% trong báo cáo tổng hợp chưa có JSON kết quả
và luồng huấn luyện tương ứng trong repository hiện tại để đối chiếu. Không
dùng nó như kết quả đã được tái lập. Tốc độ của riêng bước YOLO cũng không
đại diện cho tốc độ toàn pipeline có hiệu chỉnh và nhận diện hành động.

## Kiểm tra đã thực hiện khi chuẩn hóa

- Ruff lint, format và biên dịch cú pháp Python.
- 14 test về tính độc lập, tái lập và tính hợp lệ của các tập dữ liệu.
- 5 test chạy một epoch trên CPU với dữ liệu tổng hợp cho baseline và bốn
  cấu hình, kiểm tra validation được dùng trước ba lần đánh giá test cuối.
- Kiểm tra bước chia validation trên cache Penn Action hiện có.

Chưa huấn luyện lại benchmark đầy đủ 50 epoch, chưa tạo số liệu khoa học mới
và chưa đo lại tốc độ hoặc xác minh các so sánh công trình quốc tế.
