"""
Module: movement_profiling.py
Purpose: Automated Movement Quality Profiling & Action Quality Assessment (AQA)
         using Kinematic and Biomechanical Indicators derived from KineCalib.
         Grounded in recent sports analytics literature:
         - Ma et al. (Frontiers in Public Health, 2026): Kinematic Smoothness & Coordination.
         - Dai & Zhao (2025): Formative Kinematic Feedback in Sports Coaching.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

# 8 Anatomical Bone Segments for 12 Canonical Joints (Parent, Child, Name)
# 0: left_shoulder, 1: right_shoulder, 2: left_elbow, 3: right_elbow,
# 4: left_wrist, 5: right_wrist, 6: left_hip, 7: right_hip,
# 8: left_knee, 9: right_knee, 10: left_ankle, 11: right_ankle
CANONICAL_BONE_EDGES = [
    (0, 2, "left_upper_arm"),
    (1, 3, "right_upper_arm"),
    (2, 4, "left_forearm"),
    (3, 5, "right_forearm"),
    (6, 8, "left_thigh"),
    (7, 9, "right_thigh"),
    (8, 10, "left_shin"),
    (9, 11, "right_shin"),
]

# Symmetric Joint Pairs: (left_idx, right_idx, joint_name)
BILATERAL_PAIRS = [
    (0, 1, "shoulder"),
    (2, 3, "elbow"),
    (4, 5, "wrist"),
    (6, 7, "hip"),
    (8, 9, "knee"),
    (10, 11, "ankle"),
]

BILATERAL_ACTIONS = {
    "squat",
    "bench_press",
    "pullup",
    "pushup",
    "jumping_jacks",
    "jump_rope",
    "clean_and_jerk",
    "situp",
}

BALLISTIC_ACTIONS = {
    "baseball_pitch",
    "baseball_swing",
    "golf_swing",
    "tennis_serve",
    "tennis_forehand",
    "bowl",
}


class MovementQualityProfiler:
    """
    Evaluates athlete movement dynamics and computes multi-dimensional AQA metrics:
    1. Temporal Stability Score (Tremor / Jitter suppression)
    2. Kinematic Smoothness Score (Dimensionless Jerk & Velocity continuity)
    3. Anthropometric Postural Integrity (Limb length rigidity)
    4. Bilateral Coordination & Symmetry Index (Left-Right balance)
    5. Formative Feedback Diagnostics (Actionable coaching recommendations)
    """

    def __init__(
        self,
        fps: float = 30.0,
        ref_jitter_scale: float = 12.0,
        ref_bone_var_scale: float = 0.15,
        ref_jerk_scale: float = 25.0,
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps
        self.ref_jitter_scale = ref_jitter_scale
        self.ref_bone_var_scale = ref_bone_var_scale
        self.ref_jerk_scale = ref_jerk_scale

    def compute_trajectory_stability(self, coords: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Computes high-frequency temporal jitter as mean acceleration norm:
        jitter(t) = || p(t+1) - 2*p(t) + p(t-1) ||_2
        Returns:
            stability_score (0-100), mean_jitter_px, per_joint_jitter
        """
        T, J, _ = coords.shape
        if T < 3:
            return 100.0, 0.0, np.zeros(J)

        # Second temporal finite difference
        acc = coords[2:] - 2.0 * coords[1:-1] + coords[:-2]  # [T-2, J, 2]
        jitter_per_joint_frame = np.linalg.norm(acc, axis=-1)  # [T-2, J]
        per_joint_jitter = np.nanmean(jitter_per_joint_frame, axis=0)  # [J]
        mean_jitter = float(np.nanmean(per_joint_jitter))

        stability_score = float(
            np.clip(100.0 * np.exp(-mean_jitter / self.ref_jitter_scale), 0.0, 100.0)
        )
        return stability_score, mean_jitter, per_joint_jitter

    def compute_kinematic_smoothness(self, coords: np.ndarray) -> Tuple[float, float, List[int]]:
        """
        Computes movement smoothness based on dimensionless jerk:
        jerk(t) = || (v(t+1) - 2*v(t) + v(t-1)) / dt^2 ||
        Identifies frame indices of abrupt velocity spikes (jerk anomalies).
        """
        T, J, _ = coords.shape
        if T < 4:
            return 100.0, 0.0, []

        vel = (coords[1:] - coords[:-1]) / self.dt  # [T-1, J, 2]
        jerk = (vel[2:] - 2.0 * vel[1:-1] + vel[:-2]) / (self.dt**2)  # [T-3, J, 2]
        jerk_norms = np.linalg.norm(jerk, axis=-1)  # [T-3, J]
        mean_jerk = float(np.nanmean(jerk_norms))

        # Identify anomalous jerk frames (> 2.5 * std above mean)
        frame_jerk = np.nanmean(jerk_norms, axis=1)
        thresh = np.mean(frame_jerk) + 2.5 * np.std(frame_jerk)
        spike_frames = [int(i + 2) for i, val in enumerate(frame_jerk) if val > thresh]

        # Normalized smoothness score (higher is smoother)
        # Normalized dimensionless jerk relative to movement dynamics
        v_peak = max(10.0, float(np.nanmax(np.linalg.norm(vel, axis=-1))))
        norm_jerk = mean_jerk / (v_peak * (self.fps**2) + 1e-4)
        smoothness_score = float(np.clip(100.0 * np.exp(-norm_jerk / 0.45), 0.0, 100.0))
        return smoothness_score, mean_jerk, spike_frames

    def compute_bone_integrity(self, coords: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """
        Computes anthropometric bone length variance:
        Measures how well physical limb rigidity is preserved across the motion.
        """
        T, J, _ = coords.shape
        per_bone_cov = {}

        for p_idx, c_idx, b_name in CANONICAL_BONE_EDGES:
            bone_vec = coords[:, c_idx, :] - coords[:, p_idx, :]
            lens = np.linalg.norm(bone_vec, axis=-1)
            valid_lens = lens[~np.isnan(lens)]
            if len(valid_lens) > 1 and np.mean(valid_lens) > 1e-3:
                cov = float(np.std(valid_lens) / np.mean(valid_lens))
            else:
                cov = 0.0
            per_bone_cov[b_name] = cov

        mean_bone_cov = float(np.mean(list(per_bone_cov.values()))) if per_bone_cov else 0.0
        integrity_score = float(
            np.clip(100.0 * np.exp(-mean_bone_cov / self.ref_bone_var_scale), 0.0, 100.0)
        )
        return integrity_score, mean_bone_cov, per_bone_cov

    def compute_bilateral_coordination(
        self, coords: np.ndarray, action_name: str = ""
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes bilateral limb symmetry for paired limbs (arms and legs):
        Symmetry Index (SI) = 1 - (|V_left - V_right| / (V_left + V_right))
        """
        T, J, _ = coords.shape
        if T < 2:
            return 100.0, {}

        vel = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)  # [T-1, J]
        pair_symmetry = {}

        for l_idx, r_idx, name in BILATERAL_PAIRS:
            v_l = np.nanmean(vel[:, l_idx])
            v_r = np.nanmean(vel[:, r_idx])
            denom = v_l + v_r + 1e-5
            diff = abs(v_l - v_r)
            sym = max(0.0, 1.0 - (diff / denom)) * 100.0
            pair_symmetry[name] = float(sym)

        mean_symmetry = float(np.mean(list(pair_symmetry.values())))
        return mean_symmetry, pair_symmetry

    def generate_formative_feedback(
        self,
        action_name: str,
        total_score: float,
        stability_score: float,
        smoothness_score: float,
        integrity_score: float,
        symmetry_score: float,
        spike_frames: List[int],
        per_bone_cov: Dict[str, float],
        pair_symmetry: Dict[str, float],
    ) -> List[str]:
        """
        Synthesizes biomechanical metrics into plain-language formative feedback
        for sports coaching and clinical rehabilitation.
        """
        feedback = []

        # Overall rating
        if total_score >= 85.0:
            rating = "Xuất sắc (Elite / Optimal Execution)"
        elif total_score >= 70.0:
            rating = "Tốt (Proficient Execution)"
        elif total_score >= 55.0:
            rating = "Trung bình (Moderate - Cần Tinh chỉnh Kỹ thuật)"
        else:
            rating = "Kém (Sub-optimal - Nguy cơ Chấn thương / Lỗi Kỹ thuật)"

        feedback.append(f"Đánh giá Tổng thể: {rating} (Điểm chất lượng: {total_score:.1f}/100)")

        # 1. Smoothness feedback
        if smoothness_score < 70.0:
            if spike_frames:
                frame_str = ", ".join(str(f) for f in spike_frames[:4])
                feedback.append(
                    f"⚠️ Chuyển động thiếu độ mượt: Phát hiện hiện tượng giật gia tốc đột ngột "
                    f"tại các khung hình [{frame_str}]. Cần duy trì lực phát động liên tục, "
                    f"tránh giật cục trong pha chuyển tiếp động tác."
                )
            else:
                feedback.append("⚠️ Tốc độ chuyển động chưa đồng đều qua các pha.")
        else:
            feedback.append("✓ Quỹ đạo chuyển động mượt mà, phân bố lực và vận tốc tối ưu.")

        # 2. Stability feedback
        if stability_score < 65.0:
            feedback.append(
                "⚠️ Độ ổn định tư thế thấp: Khớp chi xa có hiện tượng rung lắc/mất thăng bằng. "
                "Cần gồng chắc nhóm cơ trọng tâm (Core stability) để cố định trục cơ thể."
            )
        else:
            feedback.append("✓ Độ ổn định quỹ đạo khớp rất tốt, kiểm soát thăng bằng chuẩn xác.")

        # 3. Postural Integrity & Anthropometric distortion
        worst_bone = max(per_bone_cov.items(), key=lambda x: x[1]) if per_bone_cov else None
        if worst_bone and worst_bone[1] > 0.20:
            feedback.append(
                f"⚠️ Biến dạng tỉ lệ cơ thể: Phân đoạn xương '{worst_bone[0]}' có độ co dãn bất thường "
                f"({worst_bone[1] * 100:.1f}%). Thường xảy ra do gập xoay khớp sai góc giải phẫu hoặc che khuất tầm nhìn."
            )

        # 4. Bilateral symmetry feedback (if applicable)
        if action_name in BILATERAL_ACTIONS:
            worst_pair = min(pair_symmetry.items(), key=lambda x: x[1]) if pair_symmetry else None
            if worst_pair and worst_pair[1] < 75.0:
                feedback.append(
                    f"⚠️ Mất đối xứng chi trái/phải: Vùng '{worst_pair[0]}' chỉ đạt {worst_pair[1]:.1f}% độ cân bằng. "
                    f"Cần phân bổ tải trọng đồng đều giữa hai bên cơ thể để tránh quá tải đơn bên."
                )
            else:
                feedback.append(
                    "✓ Độ đối xứng hai bên cơ thể đồng đều, giữ cân bằng tải trọng tốt."
                )

        return feedback

    def profile_sequence(
        self, coords: np.ndarray, action_name: str = "general_motion"
    ) -> Dict[str, Any]:
        """
        Profiles a full skeleton sequence [T, 12, 2] and outputs comprehensive AQA report.
        """
        stab_score, mean_jitter, per_j_jitter = self.compute_trajectory_stability(coords)
        smooth_score, mean_jerk, spike_frames = self.compute_kinematic_smoothness(coords)
        integ_score, mean_bone_cov, per_bone_cov = self.compute_bone_integrity(coords)
        sym_score, pair_symmetry = self.compute_bilateral_coordination(coords, action_name)

        # Dynamic weighting based on action type
        if action_name in BILATERAL_ACTIONS:
            w_stab, w_smooth, w_integ, w_sym = 0.25, 0.25, 0.25, 0.25
        elif action_name in BALLISTIC_ACTIONS:
            w_stab, w_smooth, w_integ, w_sym = 0.35, 0.35, 0.20, 0.10
        else:
            w_stab, w_smooth, w_integ, w_sym = 0.30, 0.30, 0.25, 0.15

        total_score = (
            w_stab * stab_score
            + w_smooth * smooth_score
            + w_integ * integ_score
            + w_sym * sym_score
        )
        total_score = float(np.clip(total_score, 0.0, 100.0))

        feedback = self.generate_formative_feedback(
            action_name=action_name,
            total_score=total_score,
            stability_score=stab_score,
            smoothness_score=smooth_score,
            integrity_score=integ_score,
            symmetry_score=sym_score,
            spike_frames=spike_frames,
            per_bone_cov=per_bone_cov,
            pair_symmetry=pair_symmetry,
        )

        return {
            "action": action_name,
            "total_score": round(total_score, 2),
            "sub_scores": {
                "stability_score": round(stab_score, 2),
                "smoothness_score": round(smooth_score, 2),
                "anatomical_integrity_score": round(integ_score, 2),
                "bilateral_symmetry_score": round(sym_score, 2),
            },
            "raw_biomechanical_metrics": {
                "mean_jitter_px": round(mean_jitter, 3),
                "mean_jerk_px_per_s3": round(mean_jerk, 3),
                "mean_bone_cov_ratio": round(mean_bone_cov, 4),
                "num_velocity_spike_frames": len(spike_frames),
            },
            "per_bone_length_variation": {k: round(v, 4) for k, v in per_bone_cov.items()},
            "bilateral_symmetry_indices": {k: round(v, 2) for k, v in pair_symmetry.items()},
            "formative_feedback": feedback,
        }

    def compare_raw_vs_calibrated(
        self, raw_coords: np.ndarray, calib_coords: np.ndarray, action_name: str
    ) -> Dict[str, Any]:
        """
        Demonstrates the direct AQA impact of KineCalib:
        Compares movement quality profile before vs after calibration.
        """
        raw_report = self.profile_sequence(raw_coords, action_name)
        calib_report = self.profile_sequence(calib_coords, action_name)

        gain_score = calib_report["total_score"] - raw_report["total_score"]
        stability_gain = (
            calib_report["sub_scores"]["stability_score"]
            - raw_report["sub_scores"]["stability_score"]
        )
        smoothness_gain = (
            calib_report["sub_scores"]["smoothness_score"]
            - raw_report["sub_scores"]["smoothness_score"]
        )
        integrity_gain = (
            calib_report["sub_scores"]["anatomical_integrity_score"]
            - raw_report["sub_scores"]["anatomical_integrity_score"]
        )

        return {
            "action": action_name,
            "raw_profile": raw_report,
            "calibrated_profile": calib_report,
            "calibration_improvements": {
                "total_aqa_score_gain": round(gain_score, 2),
                "stability_score_gain": round(stability_gain, 2),
                "smoothness_score_gain": round(smoothness_gain, 2),
                "anatomical_integrity_gain": round(integrity_gain, 2),
                "jitter_reduction_pct": round(
                    (
                        raw_report["raw_biomechanical_metrics"]["mean_jitter_px"]
                        - calib_report["raw_biomechanical_metrics"]["mean_jitter_px"]
                    )
                    / max(1e-4, raw_report["raw_biomechanical_metrics"]["mean_jitter_px"])
                    * 100.0,
                    2,
                ),
            },
        }
