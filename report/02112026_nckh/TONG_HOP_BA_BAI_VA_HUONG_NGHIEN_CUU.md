# Tổng hợp ba bài báo và đề xuất hướng nghiên cứu

Ngày 02/11/2026. Tổng hợp từ ba báo cáo đã đối chiếu với bài gốc; các con số bên dưới là kết quả tác giả báo, không phải kết quả chạy mới của dự án.

## So sánh ba bài

| Nội dung | [CascadeFormer](https://arxiv.org/html/2509.00692v1) | [Movement profiles](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1925220/full) | [Joints-Pooled 3D Descriptors](https://www.ijcai.org/Proceedings/16/Papers/470.pdf) |
|---|---|---|---|
| Mục tiêu | Nhận diện hành động bằng Transformer, tiền huấn luyện che khớp | So sánh classifier và mô tả đặc trưng vận động | Nhận diện bằng đặc trưng video lấy quanh vị trí khớp |
| Dữ liệu liên quan | Penn Action 15 lớp; không dùng J-HMDB | Penn Action 14 lớp, không dùng J-HMDB | Penn Action, sub-JHMDB và thí nghiệm bổ sung J-HMDB |
| Input | Skeleton 2D annotation; mã đưa khớp không hiển thị về 0 | Skeleton và các đặc trưng động học; full feature không phải bốn điểm profile tổng hợp | RGB 16 frame cùng khớp GT hoặc ước lượng |
| Kết quả minh họa | Penn Top-1 94,66% cho 1.0, theo điều kiện của bài | Penn macro-F1 0,893 → 0,905 khi bổ sung bộ đặc trưng động học, p hiệu chỉnh 0,079 | sub-JHMDB JDD conv5b: GT 81,9%, estimated 77,7% |
| Giá trị cho dự án | Classifier tham khảo; đã chạy smoke test một clip train | Cách mô tả vận động và kiểm tra thông tin có thể bị mất | Đối chứng ảnh hưởng nguồn pose khi hệ có thêm RGB |
| Giới hạn | Không chứng minh pose đã được sửa hoặc bền với lỗi pose thực | Chưa xác nhận chất lượng động tác, cũng chưa khảo sát có kiểm soát lỗi pose | Không phải skeleton-only; code JDD chính thức chưa xác nhận |

Không xếp hạng bằng các số này: khác tập lớp, nguồn pose, input RGB, protocol và metric. Các tác giả CascadeFormer báo 94,66% trên Penn theo tiền xử lý của họ, không phải độ chính xác trên đầu vào YOLO; Movement profiles dùng 14 lớp và macro-F1, không phải accuracy; JDD có RGB.

## Nhận xét chung

1. Các bài cùng sử dụng vị trí khớp, nhưng giải quyết các mục tiêu khác nhau: nhận diện, mô tả vận động và khai thác vùng hình ảnh quanh khớp. Không bài nào trong ba bài kiểm định trực tiếp bộ hiệu chỉnh thời gian đã đề xuất cho dự án.
2. Nguồn và chất lượng pose phải là một biến thí nghiệm. Trên sub-JHMDB, JDD báo chênh lệch 4,2 điểm phần trăm giữa GT và predicted cho cấu hình conv5b; đây là thiết lập RGB+pose cụ thể, không phải ước lượng hiệu ứng chung.
3. Chất lượng hình học, độ mượt và kết quả nhận diện cần được đo riêng. Đặc trưng vận động như khoảng góc và vận tốc có thể thay đổi nếu lọc quá mạnh; pose ít rung hơn không tự chứng minh pose đúng hơn hay hành động nhận diện tốt hơn.
4. Các bài không xác nhận novelty của một bộ lọc thích ứng. Các nghiên cứu trực tiếp hơn như SmoothNet, K-FPN, RPSTN và công trình imputation năm 2026 đã được liệt kê trong báo cáo khảo sát; cần đối chiếu trước khi tuyên bố đóng góp.

## Hướng đề xuất

**Câu hỏi làm việc:** Trên Penn Action và J-HMDB, hiệu chỉnh skeleton 2D ước lượng từ video theo thời gian có ảnh hưởng thế nào đến độ chính xác pose, sự bảo toàn chuyển động và độ chính xác nhận diện hành động so với các baseline được kiểm soát?

**Giả thuyết có thể bị bác bỏ:** Một bộ hiệu chỉnh lựa chọn can thiệp theo độ tin cậy và mức chuyển động có thể giảm sai lệch pose ở khớp không tin cậy mà ít làm mất chuyển động nhanh hơn lọc cố định; liệu điều đó cải thiện recognition phải kiểm định riêng. Không giả định confidence được hiệu chuẩn hoặc là ground truth.

**Thực nghiệm đề xuất:**

- Audit trước: kiểm kê dữ liệu, video/frame, split, checkpoint, mapping khớp và correspondence trái/phải; dùng tracker/chọn người không phụ thuộc GT trong pipeline triển khai. GT chỉ để đo, không tự dùng visibility GT để sửa predicted trong inference.
- Dùng chung video và cùng pose ước lượng cho các điều kiện: raw; nội suy khớp thiếu; một bộ lọc thời gian cố định; bộ lọc thích ứng đề xuất. Thiết lập GT là tham chiếu có nhãn, không coi như pose thực tế hoàn hảo. Khóa quy tắc trước khi test.
- Với classifier, xét cả (a) **giữ nguyên checkpoint** để đo tác động thay input và (b) **train/test cùng loại pose** cho raw và các điều kiện hiệu chỉnh để phân biệt tác động xử lý với chênh lệch phân phối train–test. Cùng kiến trúc, split, ngân sách và nhiều seed. Dùng classifier hiện có làm baseline, CascadeFormer bổ sung nếu audit được mapping và padding; checkpoint tác giả không mặc định độc lập với validation của dự án.
- Đo pose bằng PCK chuẩn hóa và sai số khớp trên tập hợp lệ cùng mẫu giữa điều kiện; mô tả thiếu và coverage riêng. Đo sự bảo toàn chuyển động bằng sai lệch góc/biên độ/vận tốc so với annotation khi tính được, xét frame rate, 2D projection và visibility. Báo jitter bổ sung, không dùng làm bằng chứng fidelity.
- Đo nhận diện bằng Top-1, macro-F1, chênh lệch theo cặp video (điểm phần trăm), khoảng bất định và biến thiên seed; phân tích theo lớp, mức thiếu và chuyển động nhanh. Ghi chi phí, sai lệch và kết quả âm. Phân tích có thể bị nhiễu bởi cùng người/nguồn quay.

**Phân chia và điều kiện kết luận:** Chọn thiết kế, ngưỡng, checkpoint bằng train/validation; test chỉ báo cáo sau khi khóa. Test và validation cũ đã tham gia lịch sử phát triển nên không tự xem là kiểm định độc lập tuyệt đối; nếu muốn khẳng định mạnh cần bộ dữ liệu/holdout mới độc lập. Chạy Penn trước, sau đó đánh giá riêng trong J-HMDB với topology và split phù hợp; không gọi là transfer giữa bộ nếu huấn luyện và đánh giá riêng trong từng bộ.

## Thứ tự thực hiện

1. Kiểm kê khả dụng và xác thực protocol; chọn một classifier, quy tắc đo và một mẫu nhỏ để kiểm tra pipeline.
2. Xây baseline raw/nội suy/lọc cố định và đánh giá validation; bổ sung lọc thích ứng chỉ khi đã thấy loại lỗi có thể xử lý.
3. Khóa protocol, chạy các seed và đánh giá đã quy định; sau đó mở rộng J-HMDB. Không xây hệ AQA hoặc đưa ra phản hồi chất lượng kỹ thuật từ các chỉ số nếu không có nhãn và kiểm định thích hợp.

## Tài liệu chi tiết

- [CascadeFormer](CASCADEFORMER_DANH_GIA_VA_CHAY_THU.md).
- [Movement profiles](PHAN_TICH_BAI_BAO_MOVEMENT_PROFILE_2026.md).
- [Joints-Pooled 3D](PHAN_TICH_BAI_BAO_JOINTS_POOLED_IJCAI_2016.md).
- [Khảo sát hai hướng hiệu chỉnh và prior work](KHAO_SAT_HIEU_CHINH_SKELETON_2026-11-02.md).
