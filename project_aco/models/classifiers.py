"""
Classifier builders for binary and multi-class tasks.

Binary (paper): LR, DT, SVM
Multi-class (paper): DT, SVM
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


@dataclass
class ModelConfig:
    random_state: int = 42


def build_binary_classifiers(cfg: ModelConfig) -> Dict[str, Any]:
    return {
        "LR": LogisticRegression(max_iter=2000),
        "DT": DecisionTreeClassifier(random_state=cfg.random_state),
        "SVM": SVC(kernel="rbf", probability=False),
    }


def build_multiclass_classifiers(cfg: ModelConfig) -> Dict[str, Any]:
    return {
        "DT": DecisionTreeClassifier(random_state=cfg.random_state),
        "SVM": SVC(kernel="rbf", probability=False),
    }
