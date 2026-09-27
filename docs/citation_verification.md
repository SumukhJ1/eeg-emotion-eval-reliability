# Citation verification notes

Local source files used to verify the paper's cited dataset and literature facts. This is a provenance note for the current draft, not a substitute for the final `.bib` file.

## Verified from local PDFs

| Citation used in paper | Local file checked | Verified facts relevant to the paper |
| --- | --- | --- |
| Alakus, Gonen, and Turkoglu, 2020, original GAMEEMO dataset paper | `1-s2.0-S1746809420301075-main.pdf` | Title is `Database for an emotion recognition system based on EEG signals and various computer games - GAMEEMO`; venue is `Biomedical Signal Processing and Control`; volume/page shown as `60 (2020) 101951`; dataset has 28 subjects; 14-channel EMOTIV EPOC+; four games/emotion conditions: boring, calm, horror, funny; 5 minutes per game; raw and preprocessed EEG in CSV and MAT formats. |
| Alakus and Turkoglu, 2020, GAMEEMO deep learning paper | `Electronics Letters - 2020 - Alakus - Emotion recognition with deep learning using GAMEEMO data set (2).pdf` | Title is `Emotion recognition with deep learning using GAMEEMO data set`; uses spectral entropy and bidirectional LSTM; reports 76.91% accuracy and ROC value of 90%; states GAMEEMO uses 14 Emotiv EPOC+ channels at 128 Hz and preprocessed data. |
| Su, Zhu, Song, and Chang, 2023 | `brainsci-13-00977.pdf` | Title is `Subject-Independent EEG Emotion Recognition Based on Genetically Optimized Projection Dictionary Pair Learning`; venue is `Brain Sciences`; article 13, 977; DOI text shown as `10.3390/brainsci13070977`; validates on SEED, MPED, and GAMEEMO; reports 49.01% for four-class GAMEEMO under leave-one-subject-out subject-independent evaluation. |
| Combrisson and Jerbi, 2015 | `1-s2.0-S0165027015000114-main.pdf` | Title is `Exceeding chance level by chance: The caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy`; venue is `Journal of Neuroscience Methods`; volume/page shown as `250 (2015) 126-136`; supports chance-level caution and permutation/statistical assessment framing. |
| Musgrave, Belongie, and Lim, 2020 | `2003.08505v3.pdf` | Title is `A Metric Learning Reality Check`; supports the broader evaluation-reliability framing that methodology choices can change apparent model conclusions. |
| Lotte, Congedo, Lecuyer, Lamarche, and Arnaldi, 2007 | `article2007idk.pdf` | Title is `A review of classification algorithms for EEG-based brain-computer interfaces`; venue text shows `Journal of Neural Engineering, 2007, 4`; supports EEG classifier/evaluation context. |

## EEGNet citation

The EEGNet author list and title were verified from `1611.08024v1.pdf`, the arXiv version of the paper. The final bibliography should use the canonical journal citation:

Lawhern, V. J.; Solon, A. J.; Waytowich, N. R.; Gordon, S. M.; Hung, C. P.; and Lance, B. J. 2018. EEGNet: A compact convolutional network for EEG-based brain-computer interfaces. `Journal of Neural Engineering` 15(5):056013. DOI: `10.1088/1741-2552/aace8c`.

## Dataset provenance decision

The local dataset folder name matches `Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO`.

The folder title matches the original Alakus, Gonen, and Turkoglu GAMEEMO dataset paper title. The inspected file organization also matches the original paper's description: raw and preprocessed EEG, CSV and MAT formats, 28 subjects, 14 Emotiv channels, and four game conditions. Therefore, the paper should cite Alakus et al. 2020 as the primary dataset source.

The Nasereddin et al. 2024 IEEE Access paper should not be used as the primary dataset citation unless the current analysis specifically uses that paper's release, preprocessing, or method. Current repo evidence supports Alakus et al. 2020 as the dataset provenance.

## Remaining manual checks before submission

1. Confirm the exact formatting of each BibTeX entry in the AAAI bibliography.
2. Confirm whether the final submission should include the GitHub URL in the main paper, supplemental material, or both.
3. Confirm Student Abstract eligibility with AAAI if the primary author is a high-school student rather than an undergraduate or graduate student.
