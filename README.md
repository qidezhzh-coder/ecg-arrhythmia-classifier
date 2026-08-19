# ECG Arrhythmia Classification with MIT-BIH

**Biomedical Engineering · Machine Learning · Deep Learning**

A complete pipeline for automatic cardiac arrhythmia classification using the MIT-BIH Arrhythmia Database. Implements classical machine learning (Random Forest, SVM, MLP) and a 1D Convolutional Neural Network trained directly on raw ECG signals.

---

## Project structure

```
ecg-arrhythmia-classifier/
│
├── data/
│   └── raw/                       ← MIT-BIH records (downloaded separately)
│
├── notebook/
│   ├── 01_exploration.ipynb       ← Dataset analysis and visualization
│   ├── 02_preprocessing.ipynb     ← Signal filtering and beat segmentation
│   ├── 03_feature_extraction.ipynb ← Feature engineering
│   ├── 04_classical_ml.ipynb      ← Classical ML models
│   └── 05_deep_learning_cnn1d.ipynb ← 1D CNN
│
├── src/
│   ├── preprocessing.py           ← Filtering and beat segmentation
│   ├── features.py                ← Feature extraction functions
│   ├── models.py                  ← Model definitions
│   ├── evaluation.py              ← Metrics and visualization
│   └── utils_data.py              ← Dataset utilities
│
├── results/
│   └── figures/                   ← All generated figures
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet [1, 2, 3]  
48 two-channel ambulatory ECG recordings · 360 Hz · ~110,000 annotated beats

The data is not included in this repository. Download it by running
the following in any notebook or Python script:

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/raw/')
```

Beat labels follow the **AAMI EC57** standard [5] (5 superclasses):

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
pip install -e .
```

---

## Notebooks

| Notebook | Description | Status |
|---|---|---|
| 01_exploration | Dataset characterization, class distribution, morphological analysis | ✅ Complete |
| 02_preprocessing | Butterworth filtering, beat segmentation, DS1/DS2 split | ✅ Complete |
| 03_feature_extraction | Temporal, spectral and R-R interval feature engineering | ✅ Complete |
| 04_classical_ml | Random Forest, SVM, MLP with SHAP interpretability | 🔄 In progress |
| 05_deep_learning_cnn1d | 1D CNN trained on raw ECG signal | ⏳ Planned |

---

## Methodology

- **Preprocessing:** zero-phase Butterworth bandpass filter (0.5–40 Hz) [6],
  per-beat z-score normalization
- **Segmentation:** 200-sample windows centered on annotated R-peaks
  (90 before + 110 after = 556 ms), capturing full P-QRS-T complex [4]
- **Train/test split:** inter-patient DS1/DS2 protocol from
  de Chazal et al. (2004) [4] — no data leakage between patients
- **Features:** 24 features per beat — temporal morphology, R-R interval
  context (compensatory pause), and spectral descriptors via Welch's method [7]
- **Class imbalance:** `class_weight='balanced'` + SMOTETomek on training set only [8]
- **Primary metric:** F1-macro (equal weight across all 5 classes)
- **Interpretability:** SHAP TreeExplainer on Random Forest [9]

---

## References

[1] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, May–Jun. 2001, doi: 10.1109/51.932724.

[2] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, Jun. 2000, doi: 10.1161/01.CIR.101.23.e215.

[3] G. B. Moody and R. G. Mark, "MIT-BIH Arrhythmia Database," PhysioNet. [Online]. Available: https://physionet.org/content/mitdb/.

[4] P. de Chazal, M. O'Dwyer, and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Transactions on Biomedical Engineering*, vol. 51, no. 7, pp. 1196–1206, Jul. 2004, doi: 10.1109/TBME.2004.827359.

[5] Association for the Advancement of Medical Instrumentation, *Testing and Reporting Performance Results of Cardiac Rhythm and ST Segment Measurement Algorithms*, AAMI EC57, Arlington, VA, 1998.

[6] P. S. Hamilton and W. J. Tompkins, "Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database," *IEEE Transactions on Biomedical Engineering*, vol. 33, no. 12, pp. 1157–1165, Dec. 1986, doi: 10.1109/TBME.1986.325695.

[7] P. D. Welch, "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms," *IEEE Transactions on Audio and Electroacoustics*, vol. 15, no. 2, pp. 70–73, Jun. 1967, doi: 10.1109/TAU.1967.1161901.

[8] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, Jun. 2002, doi: 10.1613/jair.953.

[9] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

---

## Author

**Qide Zhengzhao**  
Biomedical Engineering · Universidad Carlos III de Madrid (UC3M)  
Erasmus+ Exchange · Nanyang Technological University (NTU), Singapore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/qide-zhengzhao-083933299)
