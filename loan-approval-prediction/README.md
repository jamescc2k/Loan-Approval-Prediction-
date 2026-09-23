# Loan Approval Prediction (Kaggle Playground S4E10)

Este es mi entrega para el challenge de Kaggle [Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10) (Playground Series S4E10). La idea del reto es sencilla de explicar: dado un conjunto de datos de solicitantes de préstamo (edad, ingresos, historial crediticio, etc.), predecir la probabilidad de que la solicitud termine aprobada. Se evalúa con AUC-ROC.

Jaime — [github.com/jamescc2k](https://github.com/jamescc2k)

## Qué hice

Antes de tocar ningún modelo me puse a mirar los datos con calma, porque siempre me pasa que si me salto la parte de EDA termino con features que no aportan nada o, peor, con outliers metidos en el modelo sin darme cuenta. Y aquí pasó justo eso: mirando `person_age` me encontré con gente de más de 100 años pidiendo un préstamo, y en `person_emp_length` también había valores absurdos (más de 60 años de experiencia laboral). Los recorté en vez de borrar filas, porque en test también aparecen y necesito predecir igual sobre ellos.

De ahí salieron algunas cosas que me parecieron interesantes:

- `loan_grade` (la letra de riesgo, A a G) está clarísimamente correlacionada con el target, así que en vez de meterla como categórica normal la codifiqué como ordinal (A=0, ..., G=6), para que el modelo aproveche ese orden.
- Añadí un par de ratios que tienen sentido de negocio: cuánto pesa el préstamo sobre el ingreso, cuánto historial crediticio tiene la persona en relación a su edad, etc. Todo esto está en `src/features.py`.
- Probé primero una regresión logística como baseline (para tener un número de referencia rápido) y después LightGBM, que es lo que terminé usando porque el salto de rendimiento fue grande: pasé de 0.8987 a 0.9567 de AUC.

La validación la hice con Stratified K-Fold a 5 folds, porque el target está bastante desbalanceado (solo ~14% de los préstamos quedan aprobados) y no quería que un fold se quedara con muy pocos positivos por mala suerte.

**Resultado final: AUC 0.9567 (out-of-fold) con LightGBM.**

El notebook ([`notebooks/loan_approval_prediction.ipynb`](notebooks/loan_approval_prediction.ipynb)) tiene todo el proceso paso a paso con gráficos, ya ejecutado — ahí se ve el razonamiento completo, no solo el resultado final.

## Estructura

- `data/` — train.csv, test.csv y sample_submission.csv tal cual los bajé de Kaggle.
- `notebooks/loan_approval_prediction.ipynb` — el análisis completo (EDA, features, modelos, comparación, submission).
- `notebooks/build_notebook.py` — el script que genera ese notebook. Lo dejo porque un `.ipynb` en texto plano es horrible de revisar en un diff de git, así que prefiero mantener el contenido acá y regenerarlo cuando cambio algo.
- `src/features.py` — la limpieza y el feature engineering, para no repetir el código entre el notebook y el script de entrenamiento.
- `src/train.py` — versión "de línea de comandos" del entrenamiento, por si alguien (o yo mismo en unos meses) quiere reproducir el submission sin abrir Jupyter.
- `submissions/submission.csv` — el archivo final que subí a Kaggle.

## Cómo correrlo

```bash
pip install -r requirements.txt
```

Con los CSV de Kaggle ya puestos en `data/`, hay dos caminos:

```bash
# ver todo el análisis en el notebook
jupyter notebook notebooks/loan_approval_prediction.ipynb

# o directo por consola, si solo quieres el submission
python src/train.py --data-dir data --out submissions/submission.csv --folds 5
```
