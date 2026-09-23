"""Entrena los modelos de Loan Approval Prediction, registra todo en MLflow y deja
guardado el modelo final listo para servir con `mlflow models serve` / Docker.

Uso:
    python src/train.py
    python src/train.py --data-dir data --out submissions/submission.csv --folds 5
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import TARGET, build_model_frame, feature_columns


class LoanApprovalProbaModel(mlflow.pyfunc.PythonModel):
    """Envoltorio fino para que el endpoint devuelva la probabilidad de aprobación
    (predict_proba) en vez de solo la clase 0/1, que es lo que expone por defecto
    el flavor nativo de LightGBM en MLflow."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["lgbm_model"])

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(model_input)[:, 1]

LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbosity=-1,
)


def prepare_xy(df: pd.DataFrame, cols: list[str]):
    return df[cols].copy()


def train_baseline(X, y, skf) -> float:
    """Regresión logística de referencia. Se registra como run aparte en MLflow
    para poder comparar contra LightGBM directamente en la UI."""
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]
    auc = roc_auc_score(y, oof)

    with mlflow.start_run(run_name="baseline-logreg"):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_metric("auc_oof", auc)

    print(f"Regresión logística — AUC out-of-fold: {auc:.5f}")
    return auc


def train_lightgbm_cv(X, y, X_test, skf, seed: int) -> tuple[float, np.ndarray, int]:
    """CV de LightGBM. Registra un run con la métrica por fold (para ver la curva en
    MLflow) y las predicciones out-of-fold, y devuelve también el número de iteraciones
    óptimo promedio, que se usa después para entrenar el modelo final."""
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_aucs, best_iters = [], []

    with mlflow.start_run(run_name="lightgbm-cv"):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("folds", skf.n_splits)
        mlflow.log_param("seed", seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            model = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_X=X.iloc[va_idx], eval_y=y[va_idx],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            val_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
            oof[va_idx] = val_pred
            fold_auc = roc_auc_score(y[va_idx], val_pred)
            fold_aucs.append(fold_auc)
            best_iters.append(model.best_iteration_)
            mlflow.log_metric("fold_auc", fold_auc, step=fold)
            print(f"Fold {fold}/{skf.n_splits} AUC: {fold_auc:.5f}")

            test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

        auc_oof = roc_auc_score(y, oof)
        mlflow.log_metric("auc_oof", auc_oof)
        mlflow.log_metric("auc_mean", float(np.mean(fold_aucs)))
        mlflow.log_metric("auc_std", float(np.std(fold_aucs)))

    print(f"\nCV AUC out-of-fold: {auc_oof:.5f}")
    return auc_oof, test_preds, int(np.mean(best_iters))


def train_final_model(X, y, best_n_estimators: int, seed: int, model_dir: Path):
    """Reentrena LightGBM sobre el 100% del train con el número de iteraciones que dio
    la CV, y lo deja registrado + guardado en disco como modelo standalone, listo para
    empaquetar en Docker."""
    final_params = {**LGBM_PARAMS, "n_estimators": best_n_estimators}
    model = LGBMClassifier(random_state=seed, **final_params)
    model.fit(X, y)

    model_dir = Path(model_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)

    with mlflow.start_run(run_name="lightgbm-final"):
        mlflow.log_params(final_params)
        mlflow.log_param("trained_on", "full_train_set")

        # Se registra con el flavor nativo de LightGBM (queda prolijo en el Model
        # Registry, con el booster tal cual), y por separado se guarda en model_dir
        # con el wrapper de predict_proba: ese es el que sirve Docker/mlflow serve.
        mlflow.lightgbm.log_model(model, name="model", registered_model_name="loan-approval-lgbm")

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_model_path = Path(tmp_dir) / "lgbm_model.pkl"
            joblib.dump(model, raw_model_path)
            mlflow.pyfunc.save_model(
                path=str(model_dir),
                python_model=LoanApprovalProbaModel(),
                artifacts={"lgbm_model": str(raw_model_path)},
                pip_requirements=["lightgbm", "scikit-learn", "joblib", "pandas"],
            )

    print(f"Modelo final guardado en: {model_dir} (n_estimators={best_n_estimators})")
    return model


def run(data_dir: Path, out_path: Path, n_folds: int, seed: int, model_dir: Path) -> float:
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    train = build_model_frame(train_raw)
    test = build_model_frame(test_raw)

    cols = feature_columns()
    X = prepare_xy(train, cols)
    y = train[TARGET].values
    X_test = prepare_xy(test, cols)

    # Ancla el tracking store a la raíz del proyecto (no al cwd), para que dé igual si
    # se corre este script desde src/ o el notebook desde notebooks/: ambos escriben en
    # el mismo mlflow.db / mlruns/ y se ven juntos en `mlflow ui`. Uso sqlite como
    # backend porque MLflow 3 ya no soporta el Model Registry sobre el file store plano.
    project_root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{project_root / 'mlflow.db'}")
    experiment_name = "loan-approval-prediction"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=f"file:{project_root / 'mlruns'}")
    mlflow.set_experiment(experiment_name)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_baseline(X, y, skf)
    auc_oof, test_preds, best_n_estimators = train_lightgbm_cv(X, y, X_test, skf, seed)
    train_final_model(X, y, best_n_estimators, seed, model_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission[TARGET] = test_preds
    submission.to_csv(out_path, index=False)
    print(f"Submission guardado en: {out_path}")

    return auc_oof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "submissions" / "submission.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent.parent / "model")
    args = parser.parse_args()

    run(args.data_dir, args.out, args.folds, args.seed, args.model_dir)
