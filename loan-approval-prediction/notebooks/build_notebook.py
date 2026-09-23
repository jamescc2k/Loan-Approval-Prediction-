"""Genera notebooks/loan_approval_prediction.ipynb a partir de celdas definidas aquí.

Se incluye por reproducibilidad: define la estructura completa del notebook en texto plano
(más fácil de revisar en diffs de git que un .ipynb). Para regenerarlo:

    python build_notebook.py
    python -m nbconvert --to notebook --execute --inplace loan_approval_prediction.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda src: cells.append(nbf.v4.new_markdown_cell(src))
code = lambda src: cells.append(nbf.v4.new_code_cell(src))

md("""\
# Loan Approval Prediction (Kaggle Playground S4E10)

Mi entrega para el challenge de predicción de aprobación de préstamos.
""")

code("""\
import sys
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from features import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    build_model_frame,
    feature_columns,
    load_data,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)
RANDOM_STATE = 42
""")

md("## Cargando los datos")

code("""\
train_raw, test_raw, sample_submission = load_data("../data")

print("Train:", train_raw.shape)
print("Test: ", test_raw.shape)
train_raw.head()
""")

code("""\
train_raw.info()
""")

code("""\
# Valores nulos por columna (train y test)
print("Nulos en train:")
print(train_raw.isna().sum())
print("\\nNulos en test:")
print(test_raw.isna().sum())
""")

md("## ¿Cómo está repartido el target?")

code("""\
target_counts = train_raw[TARGET].value_counts(normalize=True)
print(target_counts)

ax = sns.countplot(x=TARGET, data=train_raw)
ax.set_title("Distribución de loan_status (0 = rechazado, 1 = aprobado)")
for p in ax.patches:
    ax.annotate(f"{p.get_height()/len(train_raw):.1%}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.show()
""")

md("""\
Bastante desbalanceado: solo ~14% de las solicitudes se aprueban.
Con esto en mente, voy a usar folds estratificados para la validación
""")

md("## Variables numéricas")

code("""\
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, col in zip(axes.flat, NUMERIC_COLS):
    sns.histplot(train_raw[col], bins=40, ax=ax, kde=True)
    ax.set_title(col)
for ax in axes.flat[len(NUMERIC_COLS):]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()
""")

code("""\
train_raw[NUMERIC_COLS].describe().T
""")

md("""\
Ahí están: `person_age` tiene gente de más de 120 años (obviamente un error de carga de
datos) y `person_emp_length` también se va por encima de 60 años de experiencia, lo cual
no tiene sentido para alguien que a la vez tiene 20-30 años. Los recorto en vez de tirar
esas filas — en `clean_outliers()` dentro de `src/features.py` — porque el test set también
tiene casos así y no me puedo dar el lujo de no predecirlos.
""")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x=train_raw["person_age"], ax=axes[0])
axes[0].set_title("person_age (antes de limpiar)")
sns.boxplot(x=train_raw["person_emp_length"], ax=axes[1])
axes[1].set_title("person_emp_length (antes de limpiar)")
plt.tight_layout()
plt.show()

print("Registros con person_age > 80:", (train_raw["person_age"] > 80).sum())
print("Registros con person_emp_length > 60:", (train_raw["person_emp_length"] > 60).sum())
""")

md("## Categóricas contra el target")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, CATEGORICAL_COLS):
    rate = train_raw.groupby(col)[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=ax)
    ax.set_title(f"Tasa de aprobación por {col}")
    ax.set_ylabel("P(loan_status = 1)")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
""")

code("""\
rate_grade = train_raw.groupby("loan_grade")[TARGET].mean().sort_index()
ax = sns.barplot(x=rate_grade.index, y=rate_grade.values)
ax.set_title("Tasa de aprobación por loan_grade")
ax.set_ylabel("P(loan_status = 1)")
plt.show()
""")

md("""\
Esto es lo más claro que vi en todo el EDA: `loan_grade` sube de A a G prácticamente en
línea recta con la probabilidad de que el target sea 1. Tiene todo el sentido (la letra
ya es una medida de riesgo), así que en vez de tratarla como una categórica cualquiera la
paso a ordinal (A=0 ... G=6) para que el modelo no pierda ese orden.
""")

md("## Correlaciones entre las numéricas")

code("""\
corr = train_raw[NUMERIC_COLS + [TARGET]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación (Pearson)")
plt.show()
""")

md("""\
`loan_percent_income` y `loan_int_rate` son las que más se mueven con el target. Tiene
lógica: cuanto más pesa el préstamo sobre lo que ganas y más alta la tasa que te ponen
(que ya de por sí refleja el riesgo que te asignaron), más probable que quede como 1.
""")

md("## Features nuevas")

code("""\
train = build_model_frame(train_raw)
test = build_model_frame(test_raw)

cols = feature_columns()
print(f"Total de features usadas por el modelo: {len(cols)}")
cols
""")

code("""\
train[cols + [TARGET]].head()
""")

md("""\
Además de lo que ya venía en el dataset, agregué estas (todo vive en `src/features.py`,
para no repetir la lógica entre el notebook y el script de entrenamiento):

- `loan_grade_ordinal`: la letra A-G pasada a número 0-6.
- `income_to_loan_ratio`: ingreso sobre monto del préstamo, como proxy de capacidad de pago.
- `credit_history_ratio`: años de historial crediticio sobre edad — cuánta "vida crediticia"
  relativa tiene la persona.
- `age_emp_ratio`: parecido pero con años de empleo.
- `income_per_year_employed`: ingreso normalizado por experiencia laboral.
- `log_person_income` y `log_loan_amnt`: log1p, porque ambas tienen una cola bien larga a
  la derecha y así ayudo un poco a los modelos lineales.
""")

md("## Modelos")

md("""\
Para validar uso Stratified K-Fold con 5 folds, manteniendo la proporción de clases en
cada uno. Voy a probar dos cosas:

1. Una regresión logística de referencia, para tener un piso con el que comparar.
2. LightGBM, que suele ir bien en este tipo de datos tabulares.

Las tres categóricas (`person_home_ownership`, `loan_intent`, `cb_person_default_on_file`)
ya llegan como columnas one-hot desde `feature_columns()` (categorías fijas, ver
`src/features.py`), así que `X` queda 100% numérico desde el principio — nada de
`categorical_feature` de LightGBM ni `OneHotEncoder` acá. Lo hice así a propósito: el
tipo `category` de pandas no sobrevive bien el viaje por JSON cuando el modelo se sirve
como API más adelante, y prefiero que el mismo dataframe le sirva a los dos modelos.
""")

code("""\
X = train[cols].copy()
y = train[TARGET].values
X_test = test[cols].copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
""")

md("### Regresión logística (referencia)")

code("""\
lr_oof = np.zeros(len(X))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X.iloc[tr_idx], y[tr_idx])
    lr_oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]

lr_auc = roc_auc_score(y, lr_oof)
print(f"Regresión Logística — AUC out-of-fold: {lr_auc:.5f}")
""")

md("### LightGBM")

code("""\
lgbm_oof = np.zeros(len(X))
lgbm_test_preds = np.zeros(len(X_test))
fold_aucs = []
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        X.iloc[tr_idx], y[tr_idx],
        eval_X=X.iloc[va_idx], eval_y=y[va_idx],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    val_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    lgbm_oof[va_idx] = val_pred
    fold_auc = roc_auc_score(y[va_idx], val_pred)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold}/5 — AUC: {fold_auc:.5f} — mejores iteraciones: {model.best_iteration_}")

    lgbm_test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    models.append(model)

lgbm_auc = roc_auc_score(y, lgbm_oof)
print(f"\\nLightGBM — AUC promedio por fold: {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})")
print(f"LightGBM — AUC out-of-fold: {lgbm_auc:.5f}")
""")

md("### ¿Cuál gana?")

code("""\
results = pd.DataFrame({
    "modelo": ["Regresión Logística (baseline)", "LightGBM (5-fold CV)"],
    "AUC_oof": [lr_auc, lgbm_auc],
}).sort_values("AUC_oof", ascending=False).reset_index(drop=True)
results
""")

code("""\
fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y, lr_oof, name="Regresión Logística", ax=ax)
RocCurveDisplay.from_predictions(y, lgbm_oof, name="LightGBM", ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
ax.set_title("Curvas ROC (predicciones out-of-fold)")
ax.legend()
plt.show()
""")

md("""\
LightGBM le saca bastante ventaja a la regresión logística (0.9567 vs 0.8987 de AUC), lo
cual tampoco me sorprende tanto: hay interacciones entre `loan_grade`, `loan_int_rate` y
`loan_percent_income` que un modelo lineal simplemente no puede capturar. Me quedo con
LightGBM como modelo final.
""")

md("## ¿Qué variables pesan más?")

code("""\
importances = pd.DataFrame({
    "feature": cols,
    "importance": np.mean([m.feature_importances_ for m in models], axis=0),
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=importances, x="importance", y="feature", color="steelblue")
plt.title("Importancia de variables (promedio de los 5 folds, LightGBM)")
plt.tight_layout()
plt.show()

importances
""")

md("## Generando el submission")

md("""\
Uso el promedio de los 5 modelos de la validación cruzada para predecir sobre test, en
vez de reentrenar uno solo con el 100% del train — en la práctica ese promedio (un
mini-ensamble gratis) suele generalizar un poco mejor.
""")

code("""\
submission = sample_submission.copy()
submission[TARGET] = lgbm_test_preds

submission.to_csv("../submissions/submission.csv", index=False)
submission.head()
""")

code("""\
print("Filas en submission:", len(submission))
print("Rango de probabilidades predichas:", submission[TARGET].min(), "-", submission[TARGET].max())
""")

md("## Cierre")

md("""\
En resumen: los datos venían bastante limpios salvo por un par de outliers claros en edad
y años de empleo, que corregí acotándolos. `loan_grade`, `loan_percent_income` y
`loan_int_rate` terminaron siendo, como esperaba por la intuición del dominio, las
variables que más pesan a la hora de predecir si una solicitud queda marcada como riesgosa.

LightGBM (AUC 0.9567 out-of-fold)
""")

md("## MLOps: tracking con MLflow y despliegue con Docker")

md("""\
Hasta acá todo quedó en el notebook, pero para la rama de MLOps necesito dos cosas más:
que quede registrado en MLflow (params, métricas y el modelo en sí) y que el modelo se
pueda llamar como una API real, corriendo dentro de un contenedor Docker.

Toda la lógica de entrenamiento + logging a MLflow la tengo en `src/train.py` (así no
la vuelvo a escribir acá encima); lo que hago en esta sección es correrla y mostrar qué
queda registrado.
""")

md("### Entrenando y registrando en MLflow")

md("""\
`src/train.py` hace tres cosas y cada una queda como un run separado en el experimento
`loan-approval-prediction`:

1. `baseline-logreg` — la regresión logística de referencia.
2. `lightgbm-cv` — la validación cruzada de LightGBM, con la métrica por fold.
3. `lightgbm-final` — LightGBM reentrenado sobre el 100% del train, que es el que se
   registra en el Model Registry (`loan-approval-lgbm`) y el que se guarda en `model/`
   listo para servir.

El tracking store es un sqlite (`mlflow.db`) en la raíz del proyecto, así da igual si
corro esto desde el notebook o desde consola — todo cae en el mismo sitio y se ve junto
en `mlflow ui`.
""")

code("""\
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "train.py"],
    cwd="../src",
    capture_output=True,
    text=True,
)
print(result.stdout[-1500:])
if result.returncode != 0:
    print(result.stderr[-2000:])
""")

md("""\
Y así se ven los runs que acabo de generar, consultándolos directamente con el cliente
de MLflow (lo mismo que se ve, con más detalle visual, en `mlflow ui`):
""")

code("""\
import mlflow

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name("loan-approval-prediction")
runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])

runs_df = pd.DataFrame([
    {"run_name": r.data.tags.get("mlflow.runName"), **r.data.metrics}
    for r in runs
])
runs_df
""")

md("""\
Para ver esto mismo con gráficos y comparando runs entre sí, se levanta la UI de MLflow
con:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

y se abre en `http://127.0.0.1:5000`.
""")

md("### Despliegue con Docker")

md("""\
El `Dockerfile` (en la raíz del proyecto) no entrena nada — solo empaqueta el modelo que
ya quedó guardado en `model/` junto con lo mínimo para servirlo, y arranca
`mlflow models serve` al levantar el contenedor. Los pasos son:

```bash
# 1. Construir la imagen
docker build -t loan-approval-api .

# 2. Levantar el contenedor, publicando el puerto 5000
docker run -p 5000:5000 loan-approval-api
```

Con eso, el contenedor queda escuchando en `http://localhost:5000/invocations`, que es
el endpoint estándar que expone MLflow para pedir predicciones.

*Nota: en el entorno donde escribí este notebook no tengo Docker instalado, así que no
pude hacer el `docker build` real. Lo que sí hice fue levantar el mismo servidor
(`mlflow models serve`, el comando exacto que corre el `CMD` del Dockerfile) directamente
en esta máquina para probar que el endpoint funciona de verdad — es el mismo proceso que
correría dentro del contenedor, solo que sin la capa de Docker encima.*
""")

md("### Probando el modelo desplegado")

md("""\
Levanto el servidor (el mismo comando que usa el Dockerfile) como subproceso, espero a
que responda, y le mando una solicitud real con tres registros de test para que devuelva
la probabilidad de aprobación de cada uno.
""")

code("""\
import os
import site
import subprocess
import sys
import time

# El instalador de pip dejó los ejecutables (mlflow, uvicorn) en la carpeta Scripts del
# usuario, que no está en el PATH por defecto en esta máquina — la agrego para este
# subproceso.
scripts_dir = os.path.join(os.path.dirname(site.getusersitepackages()), "Scripts")
env = os.environ.copy()
env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
env["MLFLOW_TRACKING_URI"] = "sqlite:///../mlflow.db"

server = subprocess.Popen(
    ["mlflow", "models", "serve", "-m", "../model", "--host", "127.0.0.1", "--port", "5001",
     "--env-manager", "local"],
    cwd=".",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
""")

code("""\
import requests

server_ready = False
for _ in range(30):
    try:
        if requests.get("http://127.0.0.1:5001/ping", timeout=2).status_code == 200:
            server_ready = True
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(2)

print("Servidor arriba:" if server_ready else "El servidor no respondió a tiempo", server_ready)
""")

code("""\
# Tres solicitudes reales de test.csv, procesadas con el mismo pipeline de features
# que usa el modelo (build_model_frame + feature_columns), tal como llegarían en
# producción.
sample = build_model_frame(test_raw.head(3))
X_sample = sample[cols]

payload = {"dataframe_split": {"columns": cols, "data": X_sample.values.tolist()}}

response = requests.post("http://127.0.0.1:5001/invocations", json=payload)
print("Status:", response.status_code)
print("Predicciones (probabilidad de aprobación):", response.json())
""")

md("""\
Ahí está: el endpoint devuelve la probabilidad de aprobación para cada solicitud
(`predict_proba`, no solo la clase 0/1), que es el mismo tipo de score que usé para
armar el submission. Cierro el servidor de prueba para no dejarlo corriendo de fondo.
""")

code("""\
server.terminate()
try:
    server.wait(timeout=10)
except subprocess.TimeoutExpired:
    server.kill()
print("Servidor de prueba detenido.")
""")

nb["cells"] = cells
nbf.write(nb, "loan_approval_prediction.ipynb")
print("Notebook generado.")
