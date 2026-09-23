"""Entrena un modelo LightGBM con validación cruzada para Loan Approval Prediction
y genera el archivo de submission para Kaggle.

Uso:
    python src/train.py
    python src/train.py --data-dir data --out submissions/submission.csv --folds 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import CATEGORICAL_COLS, ID_COL, TARGET, build_model_frame, feature_columns


def prepare_xy(df: pd.DataFrame, cols: list[str]):
    X = df[cols].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    return X


def run(data_dir: Path, out_path: Path, n_folds: int, seed: int) -> float:
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    train = build_model_frame(train_raw)
    test = build_model_frame(test_raw)

    cols = feature_columns()
    X = prepare_xy(train, cols)
    y = train[TARGET].values
    X_test = prepare_xy(test, cols)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_X=X.iloc[va_idx],
            eval_y=y[va_idx],
            categorical_feature=CATEGORICAL_COLS,
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        val_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof_preds[va_idx] = val_pred
        fold_auc = roc_auc_score(y[va_idx], val_pred)
        fold_aucs.append(fold_auc)
        print(f"Fold {fold}/{n_folds} AUC: {fold_auc:.5f}")

        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nCV AUC promedio por fold: {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})")
    print(f"CV AUC out-of-fold: {oof_auc:.5f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission[TARGET] = test_preds
    submission.to_csv(out_path, index=False)
    print(f"Submission guardado en: {out_path}")

    return oof_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "submissions" / "submission.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.data_dir, args.out, args.folds, args.seed)
