# Loan Approval Prediction (Kaggle Playground S4E10)

Este es mi entrega para el challenge de Kaggle [Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10) (Playground Series S4E10). La idea del reto es sencilla de explicar: dado un conjunto de datos de solicitantes de préstamo (edad, ingresos, historial crediticio, etc.), predecir la probabilidad de que la solicitud termine aprobada. Se evalúa con AUC-ROC.

Jaime — [github.com/jamescc2k](https://github.com/jamescc2k)

## Descripción

El proyecto tiene dos partes:

1. **El modelo en sí**: EDA, feature engineering y comparación de un baseline de regresión
   logística contra LightGBM, todo documentado en el notebook.
2. **La parte de MLOps** (rama `mlops`): trackeo los experimentos con MLflow (params,
   métricas y el modelo registrado), y despliego el modelo final como una API REST
   dentro de un contenedor Docker, usando `mlflow models serve`.

## Motivación

Esto empezó como la entrega de un challenge de Kaggle, pero lo aproveché para no dejar el
modelo tirado en un notebook: quería que quedara con el mismo tratamiento que le daría a
algo que va a producción — experimentos trackeados (para poder comparar corridas y no
perder de vista qué configuración dio qué resultado) y el modelo servible como API, no
solo como un `.csv` de submission.

## Objetivos

- Conseguir el mejor AUC posible en el challenge de Kaggle.
- Que el proceso de entrenamiento quede reproducible y trackeado con MLflow (no solo
  "correlo y anota el número a mano").
- Empaquetar el modelo final en un contenedor Docker que se pueda levantar con un único
  comando y responda a peticiones HTTP reales.

## Resultado

| Modelo | AUC (out-of-fold, 5-fold CV) |
|---|---|
| Regresión Logística (baseline) | 0.8987 |
| **LightGBM (modelo final)** | **0.9568** |

Predicciones finales generadas con LightGBM, promediando los 5 modelos de la validación cruzada, en [`submissions/submission.csv`](submissions/submission.csv).

## Qué hice (EDA y modelo)

Antes de tocar ningún modelo me puse a mirar los datos con calma, porque siempre me pasa que si me salto la parte de EDA termino con features que no aportan nada o, peor, con outliers metidos en el modelo sin darme cuenta. Y aquí pasó justo eso: mirando `person_age` me encontré con gente de más de 100 años pidiendo un préstamo, y en `person_emp_length` también había valores absurdos (más de 60 años de experiencia laboral). Los recorté en vez de borrar filas, porque en test también aparecen y necesito predecir igual sobre ellos.

De ahí salieron algunas cosas que me parecieron interesantes:

- `loan_grade` (la letra de riesgo, A a G) está clarísimamente correlacionada con el target, así que en vez de meterla como categórica normal la codifiqué como ordinal (A=0, ..., G=6), para que el modelo aproveche ese orden.
- Las otras tres categóricas (`person_home_ownership`, `loan_intent`, `cb_person_default_on_file`) las paso a one-hot con categorías fijas, en vez de usar el soporte nativo de LightGBM para categóricas. Lo hice a propósito pensando en el despliegue: el tipo `category` de pandas no sobrevive bien un viaje por JSON hasta el modelo servido, así que prefiero que el modelo sea 100% numérico de punta a punta.
- Añadí un par de ratios que tienen sentido de negocio: cuánto pesa el préstamo sobre el ingreso, cuánto historial crediticio tiene la persona en relación a su edad, etc. Todo esto está en `src/features.py`.
- Probé primero una regresión logística como baseline (para tener un número de referencia rápido) y después LightGBM, que es lo que terminé usando porque el salto de rendimiento fue grande.

La validación la hice con Stratified K-Fold a 5 folds, porque el target está bastante desbalanceado (solo ~14% de los préstamos quedan aprobados) y no quería que un fold se quedara con muy pocos positivos por mala suerte.

## MLOps: MLflow y Docker

`src/train.py` no solo entrena — deja todo registrado en MLflow:

- Un run **`baseline-logreg`** con el AUC de la regresión logística.
- Un run **`lightgbm-cv`** con la métrica de cada fold y el AUC out-of-fold.
- Un run **`lightgbm-final`**, que reentrena LightGBM sobre el 100% del train (con el
  número de iteraciones que salió mejor en la CV) y lo deja registrado en el Model
  Registry como `loan-approval-lgbm`, además de guardarlo en `model/` como modelo
  standalone listo para servir.

El tracking store es un sqlite (`mlflow.db`) en la raíz del proyecto — así da igual si
entrenás desde consola o desde el notebook, todo cae en el mismo sitio.

Para servir el modelo uso `mlflow models serve`, envuelto en un `Dockerfile` que solo
empaqueta el modelo ya entrenado (no entrena nada dentro del contenedor). El endpoint
devuelve la **probabilidad** de aprobación (no solo la clase 0/1) — para eso el modelo
final se guarda con un wrapper de MLflow (`mlflow.pyfunc.PythonModel`) que llama a
`predict_proba` por dentro, en vez del flavor nativo de LightGBM (que por defecto solo
expone `predict`).

Los pasos completos de cómo llegué del modelo entrenado a la API corriendo en Docker,
junto con una llamada real al endpoint desplegado, están documentados paso a paso en la
sección final del notebook.

## Tecnologías

- **Python 3.13**, pandas, scikit-learn, LightGBM — modelo y feature engineering.
- **MLflow** (tracking + Model Registry + `mlflow models serve`) — experimentos y despliegue.
- **Docker** — empaquetar el modelo servido como contenedor.
- **Jupyter** — para el notebook de EDA/modelado (generado desde `notebooks/build_notebook.py`).

## Estructura

- `data/` — train.csv, test.csv y sample_submission.csv tal cual los bajé de Kaggle.
- `notebooks/loan_approval_prediction.ipynb` — el análisis completo (EDA, features, modelos, comparación, submission, MLOps).
- `notebooks/build_notebook.py` — el script que genera ese notebook. Lo dejo porque un `.ipynb` en texto plano es horrible de revisar en un diff de git, así que prefiero mantener el contenido acá y regenerarlo cuando cambio algo.
- `src/features.py` — la limpieza y el feature engineering, para no repetir el código entre el notebook y el script de entrenamiento.
- `src/train.py` — entrenamiento + logging a MLflow + guardado del modelo final, todo en un script que corre por consola.
- `submissions/submission.csv` — el archivo final que subí a Kaggle.
- `model/` — el modelo final ya entrenado y guardado (lo genera `src/train.py`), lo que copia el Dockerfile para servir.
- `mlflow.db` / `mlruns/` — el tracking store de MLflow con los runs ya generados, para poder abrir `mlflow ui` sin tener que reentrenar antes.
- `Dockerfile` — empaqueta `model/` y lo sirve con `mlflow models serve`.

## Instalación

```bash
pip install -r requirements.txt
```

Con los CSV de Kaggle ya puestos en `data/`, hay dos caminos para correr el análisis y entrenar:

```bash
# ver todo el análisis en el notebook (EDA, modelo y MLOps)
jupyter notebook notebooks/loan_approval_prediction.ipynb

# o directo por consola, si solo quieres el submission + MLflow + el modelo guardado
python src/train.py --data-dir data --out submissions/submission.csv --folds 5
```

## Cómo levantar MLflow

Con `mlflow.db` ya generado (viene en el repo, o se genera corriendo `src/train.py`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Y se abre en `http://127.0.0.1:5000`. Ahí se ven los tres runs (`baseline-logreg`,
`lightgbm-cv`, `lightgbm-final`) con sus parámetros y métricas, y el modelo registrado en
la pestaña de Model Registry.

## Cómo levantar el modelo con Docker

```bash
# construir la imagen (empaqueta model/, no reentrena nada)
docker build -t loan-approval-api .

# levantarla, publicando el puerto 5000
docker run -p 5000:5000 loan-approval-api
```

Con el contenedor corriendo, se le puede pedir una predicción así:

```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...columnas de feature_columns()...], "data": [[...]]}}'
```

y devuelve la probabilidad de aprobación para cada fila. Un ejemplo real de esta misma
llamada (columnas, payload y respuesta) queda armado y ejecutado en la última sección del
notebook.

## Qué le faltaría

Si tuviera más tiempo probaría un ensamble de LightGBM con XGBoost/CatBoost (suele dar un empujón extra en este tipo de competencias) y tal vez un target encoding para las categóricas en vez de dejarlas nativas en LightGBM. También me quedé con ganas de tunear mejor los hiperparámetros con algo tipo Optuna en vez de los valores que usé a mano, pero para el alcance de este challenge el resultado ya me pareció sólido. Del lado de MLOps, lo próximo sería un endpoint `/health` más completo y versionar el modelo en el Registry con stages (Staging/Production) en vez de solo dejarlo en la última versión.
