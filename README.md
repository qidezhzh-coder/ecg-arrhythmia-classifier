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
├── notebooks/
│   ├── 01_exploration.ipynb       ← Dataset analysis and visualization
│   ├── 02_preprocessing.ipynb     ← Signal filtering and beat segmentation
│   ├── 03_feature_extraction.ipynb
│   ├── 04_classical_ml.ipynb
│   └── 05_deep_learning_cnn1d.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet  
48 two-channel ambulatory ECG recordings · 360 Hz · ~110,000 annotated beats

The data is not included in this repository. Download it by running
the following in any notebook or Python script:

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/raw/')
```

Beat labels follow the **AAMI EC57** standard (5 superclasses):

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

## Notebooks

| Notebook | Description | Status |
|---|---|---|
| 01_exploration | Dataset characterization, class distribution, morphological analysis | ✅ Complete |
| 02_preprocessing | Butterworth filtering, beat segmentation, DS1/DS2 split | 🔄 In progress |
| 03_feature_extraction | Temporal and spectral feature engineering | ⏳ Planned |
| 04_classical_ml | Random Forest, SVM, MLP with SHAP interpretability | ⏳ Planned |
| 05_deep_learning_cnn1d | 1D CNN trained on raw ECG signal | ⏳ Planned |

---

## Methodology

- **Preprocessing:** zero-phase Butterworth bandpass filter (0.5–40 Hz),
  per-beat z-score normalization
- **Segmentation:** 200-sample windows centered on annotated R-peaks
  (90 before + 110 after)
- **Train/test split:** inter-patient DS1/DS2 protocol from
  de Chazal et al. (2004) — no data leakage between patients
- **Class imbalance:** `class_weight='balanced'` + SMOTE on training set only
- **Primary metric:** F1-macro (equal weight across all 5 classes)

---

## References

[1] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, May–Jun. 2001, doi: 10.1109/51.932724.

[2] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, Jun. 2000, doi: 10.1161/01.CIR.101.23.e215.

[3] G. B. Moody and R. G. Mark, "MIT-BIH Arrhythmia Database," PhysioNet. [Online]. Available: https://physionet.org/content/mitdb/.

[4] P. de Chazal, M. O'Dwyer, and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Transactions on Biomedical Engineering*, vol. 51, no. 7, pp. 1196–1206, Jul. 2004, doi: 10.1109/TBME.2004.827359.

---

## Author

**Qide Zhengzhao**  
Biomedical Engineering · Universidad Carlos III de Madrid (UC3M)  
Erasmus+ Exchange · Nanyang Technological University (NTU), Singapore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/qide-zhengzhao-083933299)