# ECG Arrhythmia Classification with MIT-BIH

**Biomedical Engineering · Machine Learning · Deep Learning**

A complete pipeline for automatic cardiac arrhythmia classification
using the MIT-BIH Arrhythmia Database. Implements classical machine
learning (Random Forest, SVM, MLP) and a 1D Convolutional Neural
Network trained directly on raw ECG signals, following the inter-patient
DS1/DS2 evaluation protocol from de Chazal et al. (2004) **[4]**.

---

## Project structure

```
ecg-arrhythmia-classifier/
│
├── data/
│   └── raw/                            ← MIT-BIH records (downloaded separately)
│
├── notebooks/
│   ├── 01_exploration.ipynb            ← Dataset analysis and visualization
│   ├── 02_preprocessing.ipynb          ← Signal filtering and beat segmentation
│   ├── 03_feature_extraction.ipynb     ← Feature engineering
│   ├── 04_classical_ml.ipynb           ← Classical ML models + SHAP
│   └── 05_deep_learning_cnn1d.ipynb    ← 1D CNN + ablation study
│
├── src/
│   ├── preprocessing.py                ← Filtering and beat segmentation
│   ├── features.py                     ← Feature extraction functions
│   ├── models.py                       ← Model definitions (RF, SVM, MLP, CNN)
│   ├── evaluation.py                   ← Metrics and visualization
│   └── utils_data.py                   ← Dataset utilities
│
├── results/
│   └── figures/                        ← All generated figures
│
├── requirements.txt
└── README.md
```

---

## Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet **[1, 2, 3]**  
48 two-channel ambulatory ECG recordings · 360 Hz · ~110,000 annotated beats

The data is not included in this repository. Download it by running
the following in any notebook or Python script:

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/raw/')
```

Beat labels follow the **AAMI EC57** standard **[5]** (5 superclasses):

| Class | Name | Description |
|---|---|---|
| N | Normal | Normal sinus rhythm beats |
| S | Supraventricular ectopic | Atrial or junctional premature beats |
| V | Ventricular ectopic (PVC) | Premature ventricular contractions |
| F | Fusion | Fusion of normal and ventricular beat |
| Q | Unknown / Paced | Paced or unclassifiable beats |

---

## Installation

```bash
git clone https://github.com/qidezhzh-coder/ecg-arrhythmia-classifier.git
cd ecg-arrhythmia-classifier
python -m venv ecg_env
source ecg_env/bin/activate
pip install -r requirements.txt
```

---
## Reproducing the results

Execute the notebooks in order:

1. Download the dataset: `wfdb.dl_database('mitdb', dl_dir='data/raw/')`
2. Run `02_preprocessing.ipynb` → generates `data/processed_beats.npz`
3. Run `03_feature_extraction.ipynb` → generates `data/features_train.csv` and `data/features_test.csv`
4. Run `04_classical_ml.ipynb` → classical ML results and figures
5. Run `05_deep_learning_cnn1d.ipynb` → CNN results and figures
6. `01_exploration.ipynb` can be run independently at any point

---

## Notebooks

| Notebook | Description | Status |
|---|---|---|
| 01_exploration | Dataset characterization, class distribution, morphological analysis | ✅ Complete |
| 02_preprocessing | Butterworth filtering, beat segmentation, DS1/DS2 split | ✅ Complete |
| 03_feature_extraction | Temporal, spectral and R-R interval feature engineering | ✅ Complete |
| 04_classical_ml | Random Forest, SVM, MLP with SHAP interpretability | ✅ Complete |
| 05_deep_learning_cnn1d | 1D CNN with ablation study on class imbalance strategies | ✅ Complete |

---

## Methodology

- **Preprocessing:** zero-phase Butterworth bandpass filter (0.5–40 Hz) **[6]**,
  per-beat z-score normalization
- **Segmentation:** 200-sample windows centered on annotated R-peaks
  (90 before + 110 after = 556 ms), capturing full P-QRS-T complex **[4]**
- **Train/test split:** inter-patient DS1/DS2 protocol from
  de Chazal et al. (2004) **[4]**, so no data leakage between patients
- **Features (Notebooks 03–04):** 24 features per beat, temporal morphology,
  R-R interval context (compensatory pause), and spectral descriptors
  via Welch's method **[7]**
- **Class imbalance (classical ML):** `class_weight='balanced'` +
  SMOTETomek inside each CV fold **[8]**
- **Class imbalance (CNN):** weighted CrossEntropyLoss only. Since in an ablation study in Notebook 05 shows WeightedRandomSampler causes overfitting with class Q (8 training samples)
- **Primary metric:** F1-macro (equal weight across all 5 classes)
- **Interpretability:** SHAP TreeExplainer on Random Forest **[9]**
- **Optimizer (CNN):** Adam with weight decay **[10]**
- **ECG component durations:** following standard clinical reference values **[11]**

---

## Results

All models evaluated on DS2 (inter-patient test set, used exactly once).

### Classical ML — F1-macro on DS2

| Model | CV F1 (mean ± std) | Test F1 | Class V Recall |
|---|---|---|---|
| Random Forest | 0.7527 ± 0.0351 | **0.4294** | 0.905 |
| MLP | 0.7718 ± 0.0427 | 0.4191 | 0.879 |
| SVM (RBF) | 0.6726 ± 0.0293 | 0.3676 | 0.904 |

### Deep Learning — CNN 1D on DS2

| Model | Test F1 | Class V Recall |
|---|---|---|
| CNN 1D | 0.3335 | 0.842 |

### Ablation study — class imbalance strategies (Notebook 05)

| Configuration | Best Val Loss | Val F1-Macro | Class V Recall |
|---|---|---|---|
| Baseline (standard CE + shuffle) | 0.2323 | 0.516 | 0.821 |
| Only Sampler (standard CE + WRS) | 0.3027 | **0.526** | **0.937** |
| Only Weighted Loss (weighted CE + shuffle) | 0.4868 | 0.451 | 0.817 |
| Combined (weighted CE + WRS) | 0.8426 | 0.331 | 0.871 |

### Key findings

- The large CV → test gap (+0.30) for classical models reflects the
  inter-patient variability inherent to ECG classification — a gap
  that is invisible with random beat-level splits, which is why many
  published papers reporting >95% accuracy on MIT-BIH are not
  clinically meaningful.
- Class V (PVC) recall is consistently high across all models (0.82–0.94),
  driven by its morphologically distinctive wide QRS and compensatory
  pause pattern.
- Class F (Fusion) and class Q (Paced) remain effectively undetectable
  by all models, consistent with their morphological ambiguity and
  negligible training sample count (8 beats for class Q).
- The Random Forest remains the best model overall: highest test F1-macro
  and interpretable via SHAP feature importance.

---

## References

[1] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia
Database," *IEEE Engineering in Medicine and Biology Magazine*, vol. 20,
no. 3, pp. 45–50, May–Jun. 2001, doi: 10.1109/51.932724.

[2] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet,"
*Circulation*, vol. 101, no. 23, pp. e215–e220, Jun. 2000,
doi: 10.1161/01.CIR.101.23.e215.

[3] G. B. Moody and R. G. Mark, "MIT-BIH Arrhythmia Database," PhysioNet.
[Online]. Available: https://physionet.org/content/mitdb/.

[4] P. de Chazal, M. O'Dwyer, and R. B. Reilly, "Automatic classification
of heartbeats using ECG morphology and heartbeat interval features,"
*IEEE Transactions on Biomedical Engineering*, vol. 51, no. 7,
pp. 1196–1206, Jul. 2004, doi: 10.1109/TBME.2004.827359.

[5] Association for the Advancement of Medical Instrumentation,
*Testing and Reporting Performance Results of Cardiac Rhythm and
ST Segment Measurement Algorithms*, AAMI EC57, Arlington, VA, 1998.

[6] P. S. Hamilton and W. J. Tompkins, "Quantitative investigation of
QRS detection rules using the MIT/BIH arrhythmia database," *IEEE
Transactions on Biomedical Engineering*, vol. 33, no. 12,
pp. 1157–1165, Dec. 1986, doi: 10.1109/TBME.1986.325695.

[7] P. D. Welch, "The use of fast Fourier transform for the estimation
of power spectra," *IEEE Transactions on Audio and Electroacoustics*,
vol. 15, no. 2, pp. 70–73, Jun. 1967, doi: 10.1109/TAU.1967.1161901.

[8] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer,
"SMOTE: Synthetic Minority Over-sampling Technique," *Journal of
Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002,
doi: 10.1613/jair.953.

[9] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting
model predictions," in *Advances in Neural Information Processing
Systems*, vol. 30, 2017. [Online]. Available:
https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

[10] D. P. Kingma and J. Ba, "Adam: A method for stochastic
optimization," in *ICLR*, 2015. [Online]. Available:
https://arxiv.org/abs/1412.6980

[11] A. L. Goldberger, Z. D. Goldberger, and A. Shvilkin,
*Goldberger's Clinical Electrocardiography: A Simplified Approach*,
9th ed. Philadelphia, PA: Elsevier, 2017.

---

## Author

**Qide Zhengzhao**  
Biomedical Engineering 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/qide-zhengzhao-083933299)
