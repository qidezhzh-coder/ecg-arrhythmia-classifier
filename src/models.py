# src/models.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn


def get_classical_models(RANDOM_SEED):
    """
    Return a dictionary of scikit-learn pipelines ready for training.
    Each pipeline includes StandardScaler + classifier.
    class_weight='balanced' adjusts loss to compensate for class imbalance.
    """
    return {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators = 200,
                class_weight = 'balanced',
                random_state = RANDOM_SEED,
                n_jobs = -1
            ))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel = 'rbf',
                class_weight = 'balanced',
                probability = True,   # needed for AUC-ROC computation
                random_state = RANDOM_SEED
            ))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes = (128, 64, 32),
                max_iter = 300,
                random_state = RANDOM_SEED
            ))
        ])
    }


class ECGClassifierCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG beat classification.

    Architecture rationale:
    - Conv layers learn morphological patterns directly from raw signal
    - Kernel sizes decrease progressively (11 → 7 → 5): first layer captures
      broad QRS shape (~30ms at 360Hz), deeper layers capture finer details
    - BatchNorm stabilizes training gradients
    - AdaptiveAvgPool produces fixed output size regardless of input length
    - Dropout prevents overfitting on the small minority classes
    """
    def __init__(self, num_classes=5, input_length=200):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=6, stride=6),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))