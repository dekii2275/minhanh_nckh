# Bản mã gốc của các thí nghiệm đến 13/11/2026

Thư mục `src/gt_baselines/` là bản sao nguyên byte của các script nghiên cứu trước khi chuẩn hóa định dạng để đưa vào Git. Các giá trị SHA-256 trong protocol và artifact thí nghiệm được tính từ những bản gốc này. Ví dụ, protocol official test ghi hash của `official_test_abrupt_interpolation_20261113.py`, `evaluate_abrupt_joint_interpolation_20261113.py` và `compare_person_selection.py` tại đây.

Mã phát triển hiện hành nằm ở `src/gt_baselines/` của repository. Bản snapshot phục vụ đối chiếu nguồn gốc, không được thiết kế làm một bản cài độc lập; nhiều script giả định vị trí `src/gt_baselines/` và cần dữ liệu, cache hoặc checkpoint không nằm trong Git. Chuẩn hóa làm thay đổi hash tệp ngay cả khi hành vi không đổi: không thay hash đã ghi trong kết quả cũ bằng hash của mã mới.
