# GAMEEMO prior-method comparison

Literature context for the GAMEEMO results. A reported GAMEEMO accuracy is only directly comparable to this repo if the label task, preprocessing, windowing, and split protocol match. Binary positive/negative studies are not directly comparable to the current four-class boring/calm/horror/funny setup.

| **Paper** | **Year** | **Task** | **Split** | **Reported result** | **Directly comparable?** |
| --------- | -------- | -------- | --------- | ------------------- | ------------------------ |
| [Alakus et al., *Database for an emotion recognition system based on EEG signals and various computer games - GAMEEMO*](https://doi.org/10.1016/j.bspc.2020.101951) | 2020 | GAMEEMO dataset introduction; EEG from 28 subjects, four games/emotions, plus arousal-valence ratings | Dataset paper reports classical pattern-recognition baselines; split details need care before reuse | Later open-access papers report Alakus et al. baseline examples such as SVM `71.64%` for positive/negative and channel-wise four-class baselines up to MLPNN `82%` | **No.** Useful dataset/baseline context, but not a matched LOSO four-class comparison. |
| [Alakus and Turkoglu, *Emotion recognition with deep learning using GAMEEMO data set*](https://doi.org/10.1049/el.2020.2460) | 2020 | Binary positive/negative emotion recognition using spectral entropy features and BiLSTM | Train/test evaluation; subject-independent separation not established from the public page | Accuracy `76.91%`, sensitivity `76.93%`, specificity `76.89%`, ROC `90%` | **No.** Binary positive/negative task, not four-class GAMEEMO. |
| [Aslan, *CNN based efficient approach for emotion recognition*](https://doi.org/10.1016/j.jksuci.2021.08.021) | 2022 | Binary positive/negative classification using CWT scalogram images, GoogLeNet deep features, and ML classifiers | 10-fold cross-validation | Best accuracy `98.78%` with SVM; kNN `98.53%`; ELM `98.41%` | **No.** Binary task and image/deep-feature pipeline; not directly comparable to four-class LOSO raw-window evaluation. |
| [Abdulrahman et al., *A Novel Approach for Emotion Recognition Based on EEG Signal Using Deep Learning*](https://doi.org/10.3390/app121910028) | 2022 | Binary and multi-class GAMEEMO classification using VMD/EMD-derived statistical features and DeepBiLSTM | 10-fold evaluation reported in public tables | Binary DeepBiLSTM mean `70.89%`; multi-class DeepBiLSTM `90.33%` | **No.** Includes multi-class GAMEEMO, but public protocol is 10-fold rather than clearly subject-independent LOSO. |
| [Kiruthiga et al., *Subject-Independent EEG Emotion Recognition Based on Genetically Optimized Projection Dictionary Pair Learning*](https://doi.org/10.3390/brainsci13070977) | 2023 | Two-class and four-class GAMEEMO subject-independent emotion recognition using log spectral power features and GA-PDPL | Subject-independent evaluation across 28 GAMEEMO subjects | Four-class GA-PDPL `49.01 +/- 1.60%`; four-class SVM baseline `46.62 +/- 1.89%` | **Closest comparison.** Same four-class subject-independent GAMEEMO framing, but feature extraction/model details differ from this repo. |
| [LEDPatNet19: Automated Emotion Recognition Model based on Nonlinear LED Pattern Feature Extraction Function using EEG Signals](https://pmc.ncbi.nlm.nih.gov/articles/PMC9279545/) | 2022 | Four-class GAMEEMO classification using TQWT, LED-pattern/statistical handcrafted features, RFIChi2 feature selection, and cubic SVM | 10-fold cross-validation | Best GAMEEMO channel accuracy `99.29%` | **No.** Four-class GAMEEMO, but 10-fold/channel-wise handcrafted-feature setup is not matched to LOSO raw-window evaluation. |
| [*Subject-independent multi-channel voting for EEG-based emotion recognition using wavelet scattering deep network and advanced signal metrics*](https://doi.org/10.1007/s10044-025-01501-1) | 2025 | GAMEEMO four-class subject-independent recognition with wavelet scattering / signal metrics and multi-channel voting | LOSO cross-validation, according to the public Springer abstract/snippet | GAMEEMO four-class accuracy `79.0179%` without multi-channel majority vote; `100%` with majority voting | **Closest but cautious.** The non-voting four-class LOSO result is protocol-relevant; the voting result needs careful scrutiny before comparison because channel-voting/aggregation can change the evaluation unit. |

## Interpretation for current results

- The safest first reference point for this repo is still the `25%` random baseline for four-class GAMEEMO.
- Binary positive/negative papers should not be compared directly against the current four-class task.
- 10-fold cross-validation papers should not be treated as subject-independent evidence unless the paper explicitly prevents subject leakage.
- The closest comparison category is subject-independent four-class GAMEEMO work, especially Kiruthiga et al. and the 2025 multi-channel voting paper, but even these are not exact matches because their feature pipelines and aggregation units differ.
- Do not claim SOTA unless the comparison protocol is matched: same four labels, same subject-independent split, same evaluation unit, and no test-subject leakage in preprocessing or normalization.

## Sources checked

- GAMEEMO dataset paper: https://doi.org/10.1016/j.bspc.2020.101951
- GAMEEMO dataset card: https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo
- Alakus and Turkoglu BiLSTM paper: https://doi.org/10.1049/el.2020.2460
- Aslan GoogLeNet/CWT paper: https://doi.org/10.1016/j.jksuci.2021.08.021
- Abdulrahman et al. DeepBiLSTM paper: https://doi.org/10.3390/app121910028
- Kiruthiga et al. GA-PDPL paper: https://doi.org/10.3390/brainsci13070977
- LEDPatNet19 paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9279545/
- Subject-independent multi-channel voting paper: https://doi.org/10.1007/s10044-025-01501-1

## Omitted or treated cautiously

Some GAMEEMO-related claims found through search snippets or reposted PDFs were not promoted into comparison claims when the public source did not expose enough protocol detail. High accuracies from binary tasks, channel-level 10-fold cross-validation, majority-vote aggregation, or unclear split protocols should be cited only as motivation for why protocol auditing matters.
