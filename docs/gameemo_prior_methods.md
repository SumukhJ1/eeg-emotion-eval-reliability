# GAMEEMO Prior-Method Comparison

This table is for literature context only. The current project should not claim direct comparison to these reported accuracies unless the label definition, preprocessing, windowing, train/test split, and subject-separation protocol are verified to match.

| Year | Source | Method / features | Reported accuracy | Task / protocol caveat |
| --- | --- | --- | ---: | --- |
| 2020 | Alakus et al., GAMEEMO dataset paper | Original ML baselines, including SVM and KNN | SVM 73%; KNN 66% | Reported as original GAMEEMO baseline context in later GAMEEMO papers; exact split and label setup still need verification before direct comparison. |
| 2020 | Alakus and Turkoglu, *Emotion recognition with deep learning using GAMEEMO data set* | Spectral entropy features + BiLSTM | 76.91% | Binary positive/negative emotion classification with train/test split; not directly comparable to this repo's current 4-class boring/calm/horror/funny setup. |
| 2022 | Abdulrahman et al., *A Novel Approach for Emotion Recognition Based on EEG Signal Using Deep Learning* | VMD/EMD-derived statistical features + DeepBiLSTM | 90.33% | Multi-class dimensional emotion setup reported on GAMEEMO; split/windowing/preprocessing details must be matched before direct comparison. |
| 2022 | Aslan, *CNN based efficient approach for emotion recognition* | CWT scalogram images + GoogLeNet deep features + SVM | 98.78% | Binary positive/negative classification using 10-fold cross-validation; high result is not directly comparable to subject-level LOSO or 4-class game-label experiments. |
| 2023 | Kiruthiga et al., *Subject-Independent EEG Emotion Recognition Based on Genetically Optimized Projection Dictionary Pair Learning* | GA-PDPL subject-independent feature representation | 49.01% | Four-class GAMEEMO result under leave-one-subject-out cross-validation; closest protocol-level context for this repo's LOSO work, but feature pipeline differs. |

## Interpretation For Current Results

- The safest first reference point for this repo is still the 25% random baseline for 4-class GAMEEMO.
- Internal comparisons should prioritize matched preprocessing, labels, and splits: time-statistical features vs. bandpower vs. EEGNet under the same protocol.
- Published GAMEEMO results are useful for motivation and method context, but several use binary labels, image-converted features, cross-validation without confirmed subject separation, or feature pipelines that are not equivalent to this repo.

## Sources Checked

- GAMEEMO dataset paper metadata: https://doi.org/10.1016/j.bspc.2020.101951
- GAMEEMO dataset card: https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo
- Alakus and Turkoglu BiLSTM paper: https://doi.org/10.1049/el.2020.2460
- Abdulrahman et al. DeepBiLSTM paper: https://www.mdpi.com/2076-3417/12/19/10028
- Aslan GoogLeNet/CWT paper: https://link.springer.com/article/10.1016/j.jksuci.2021.08.021
- Kiruthiga et al. GA-PDPL paper: https://www.mdpi.com/2076-3425/13/7/977
