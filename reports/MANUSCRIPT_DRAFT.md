# Correspondence Alignment and Skeleton Calibration for Action Recognition: A Validation Diagnostic Study

**Draft revised after Steps 1–3, 29 October 2026. Not a final independent-test evaluation or a state-of-the-art claim.** This draft supersedes the earlier interpretation that calibration itself explains the raw-to-full accuracy gain.

## Abstract

Postprocessing estimated 2D skeletons may improve motion regularity without improving action recognition. We separate left/right correspondence alignment from coordinate calibration in an existing YOLO11n-Pose–BiLSTM pipeline. A pilot audit of 30 Penn Action official-training clips identifies failures of a clip-level shoulder-order heuristic, local correspondence reversals, and incorrect person selection. We then freeze one GT-trained 24D Protocol-v2 checkpoint and evaluate 253 held-out-training validation clips (17,747 frames). Raw YOLO achieves 44.66% Top-1, fixed swap 67.59%, production-heuristic alignment only 66.80%, and full KineCalib 64.43%. A complete 2³ ablation with the same saved alignment decisions finds no calibration configuration exceeding alignment-only Top-1. Full minus alignment-only is −2.37 percentage points (paired-bootstrap 95% CI [−5.14, +0.40]). PCHIP with One-Euro has the highest observed Macro-F1, 67.75%, but lower Top-1 than alignment-only. These conditional validation results motivate correspondence controls and task-specific evaluation; they do not establish universal calibration harm or independent-test generalization.

## Research question and scope

Does the apparent benefit of a raw-to-calibrated pipeline come from correcting joint correspondence or from PCHIP interpolation, bone-length constraints, and One-Euro smoothing? The contribution here is an empirical decomposition with inspectable per-video outcomes. Algorithmic novelty, biomechanical validity, real-time streaming, AQA, transfer learning, and state-of-the-art performance are not established by these experiments.

## Methods

The classifier is a frozen GT-trained BiLSTM with attention: 24 inputs (12 joints × xy), 128 hidden units, two bidirectional layers, four attention heads, mean pooling, 15 classes. Protocol v2 seed0 supplies 1,005 training and 253 validation IDs, disjoint at video level. The checkpoint was selected using validation GT accuracy; its stored 94.46640316% validation GT result is reproduced before the diagnostic comparison. This validation set therefore is not an independent final evaluation.

YOLO11n-Pose uses the existing weights, imgsz=640 and box threshold 0.25. The highest-confidence person is selected without GT, including cases where that person differs from the annotation target. All clips and frames are retained. All configurations share detections, features, float32 normalization, batch32, lengths, padding and classifier weights. GT joints never select the mapping.

The alignment heuristic computes the median x-coordinate difference between predicted left and right shoulders and swaps six bilateral joint pairs over the full clip if positive. It checks missing values but has no confidence-based orientation proof. In production its input follows interpolation, bone projection and smoothing. Heuristic-only uses that production decision on raw coordinates; it does not recalculate a heuristic from raw shoulders. The pilot does not establish a global viewer-centric versus anatomical convention for Penn Action.

Calibration retains the existing code and defaults: confidence 0.35, interpolation gap 15, bone ratios 0.40–1.20, 30Hz, min cutoff 1.0, beta 0.05, derivative cutoff 1.0. PCHIP fills eligible interior gaps; it does not extrapolate boundaries, and invalid confidence does not automatically convert every remaining coordinate to NaN. Bone constraints operate on projected 2D lengths relative to a clip median, not verified 3D anthropometry. Stage order is PCHIP→Bone→One-Euro. Offline future-point and clip-median operations prevent a causal-streaming claim.

For the 2³ study, each configuration skips disabled stages and applies the same saved per-video permutation last. No heuristic is invoked during this ablation. A and H reproduce the Step-2 heuristic-only and full posteriors exactly. A 4-thread replay initially gave identical labels with tiny probability differences; matching the 8-thread CPU setting selected by YOLO in Step 2 removes the difference entirely, without changing model or calibration parameters.

## Results

In the 30-clip pilot, 15 clips support fixed swap consistently on observable evidence, four show sustained or brief/local reversals, and 11 remain inconclusive. The heuristic selects the supported mapping in 11 of the 15 consistent-swap clips. No clip is established as consistently identity-only; this does not imply identity is wrong in every frame. Wrong-person and occlusion cases remain inconclusive, not automatic heuristic successes. Review used representative frames and diagnostic screening, not exhaustive independent human annotation.

The four-configuration comparison follows; all deltas are percentage points.

| Configuration | Top-1 (%) | Macro-F1 (%) | Δ vs raw (pp) |
|---|---:|---:|---:|
| C1 – Raw | 44.66 | 38.21 | +0.00 |
| C2 – Fixed Swap | 67.59 | 68.21 | +22.92 |
| C3 – Heuristic Only | 66.80 | 67.39 | +22.13 |
| C4 – Full KineCalib | 64.43 | 64.79 | +19.76 |

The path raw→heuristic-only→full yields +22.13 pp from alignment decisions and −2.37 pp from the remaining coordinate transformations, for a total +19.76 pp. These are conditional contrasts, not independent causal contribution percentages.

| Contrast | Δ Top-1 (pp) | Paired-bootstrap 95% CI |
|---|---:|---:|
| Fixed swap − raw | +22.92 | [+17.00, +28.85] |
| Heuristic only − raw | +22.13 | [+17.79, +26.48] |
| Full − heuristic only | −2.37 | [−5.14, +0.40] |
| Full − fixed swap | −3.16 | [−7.11, +0.79] |

Full factorial ablation with frozen alignment:

| Configuration | Top-1 (%) | Macro-F1 (%) | Δ vs alignment only (pp) |
|---|---:|---:|---:|
| A – Alignment Only | 66.80 | 67.39 | +0.00 |
| B – PCHIP | 66.40 | 67.16 | -0.40 |
| C – Bone | 65.61 | 65.80 | -1.19 |
| D – One-Euro | 65.22 | 65.84 | -1.58 |
| E – PCHIP + Bone | 65.22 | 65.46 | -1.58 |
| F – PCHIP + One-Euro | 66.40 | 67.75 | -0.40 |
| G – Bone + One-Euro | 64.43 | 64.81 | -2.37 |
| H – Full | 64.43 | 64.79 | -2.37 |

Standalone PCHIP corrects two and harms three clips, bone corrects zero and harms three, One-Euro corrects three and harms seven, and full corrects four and harms ten relative to A. Every primary ablation CI against A includes or touches zero. Across all eight configurations, 42 clips change predicted labels, with 18 changing correctness.

Bone has negative point estimates in all four contexts. Adding bone to PCHIP+One-Euro loses five correct clips without rescuing any: −1.98 pp, CI [−3.95, −0.40], an exploratory unadjusted comparison. Adding PCHIP to One-Euro yields +1.19 pp [−0.79, +3.16]; the PCHIP×One-Euro interaction with bone disabled is +1.58 pp [0.00, +3.56]. Neither establishes a reliable positive effect. Equal Top-1 can conceal different per-video outcomes. F improves Macro-F1 by an observed +0.36 pp over A while losing one correct clip; no inferential Macro-F1 advantage is claimed.

PCHIP changes 129 clips and fills 1,922 xy NaN values; all-zero feature frames decrease from 184 to 99. Bone changes 248 clips and One-Euro changes 253. Thus absent aggregate utility is not explained by the modules doing nothing. Filled coordinates are not automatically anatomically correct or action-informative.

## Uncertainty and limitations

Paired video bootstrap uses 20,000 stratified resamples, preserving each action's validation count, seed 20261029 and percentile 95% intervals. Identical resampled IDs are used across configurations and interaction contrasts. CIs are unadjusted for multiple comparisons, condition on one fixed checkpoint and class composition, and omit training-seed variability and possible dependence between clips from the same source. Intervals touching zero are not treated as excluding zero. Failure to show superiority is not equivalence.

The checkpoint was selected on this validation GT set. Five validation clips also appeared in the pilot; no per-video review result selects the mapping in the experiment. Historical official-test benchmarks predate this study, so the broader development history cannot be described as entirely untouched by test information. No new official-test evaluation, retraining, or parameter tuning occurs in Steps 1–3. Negative utility on a GT-trained classifier may reflect feature-distribution shift as well as trajectory distortion; this study does not identify the mechanism.

## Implications and bounded conclusion

Alignment-only is the strongest Top-1 control in the frozen eight-way comparison; fixed swap is also a necessary comparator from Step 2. Neither becomes a universal deployment policy from these data. Bone's view-dependent 2D assumptions, PCHIP gap uncertainty, and smoothing latency are hypotheses for future research rather than verified causes. A useful next contribution would characterize when correspondence correction and coordinate transformations help or harm, with fidelity, timing, person selection, uncertainty and classifier distribution treated separately.

Historical multi-seed, J-HMDB, KD and proxy-scoring measurements are retained in dedicated reports and do not establish calibration's isolated contribution here. Protocol-matched literature comparisons and independent confirmation remain outstanding; no numerical SOTA ranking or externally validated AQA claim is included in this draft.

Evidence: [Step 1](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [Step 2](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [Step 3](CALIBRATION_ABLATION_STEP3_2026-10-29.md), [protocol](EXPERIMENT_PROTOCOL.md), [reproducibility and publication scope](REPRODUCIBILITY.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
