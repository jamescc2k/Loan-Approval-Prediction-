# Imagen para servir el modelo final (ya entrenado y guardado en model/) como una API
# REST con MLflow. No entrena nada: solo empaqueta el modelo + el runtime necesario
# para levantarlo con `mlflow models serve`.
FROM python:3.13-slim

WORKDIR /app

# Solo lo que hace falta para servir (no todo requirements.txt, que trae jupyter,
# matplotlib, etc. que no pintan nada en el contenedor).
RUN pip install --no-cache-dir mlflow>=3.16 lightgbm>=4.7 scikit-learn>=1.7 pandas>=2.3 joblib>=1.5

# El modelo ya entrenado (lo genera src/train.py, no se entrena dentro del contenedor).
COPY model/ ./model/

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "model", "--host", "0.0.0.0", "--port", "5000", "--env-manager", "local"]
