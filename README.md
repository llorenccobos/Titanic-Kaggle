# Titanic: predicción de supervivencia

Proyecto de la competición de Kaggle [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
El objetivo es predecir qué pasajeros sobrevivieron al hundimiento a partir de sus datos (clase, sexo, edad, familia, tarifa...).

## Enfoque

- **Feature engineering:** título extraído del nombre (`Title`), tamaño de familia, `IsAlone`, `HasCabin` y `LogFare` (log1p de la tarifa por su asimetría).
- **Preprocesado en `Pipeline` + `ColumnTransformer`:** imputación, escalado y one-hot encoding dentro de la validación cruzada, para evitar fuga de datos.
- **Modelos comparados:** regresión logística, KNN, árbol de decisión, random forest y gradient boosting, más un `VotingClassifier`.
- **Tuning:** `RandomizedSearchCV` (25 iteraciones, CV estratificada de 5 folds, métrica F1).
- **Evaluación final:** hold-out del 20% usado una sola vez con el modelo elegido.

## Resultados (CV 5-fold, F1, tras el tuning)

| Modelo | F1 |
|---|---|
| Baseline (clase mayoritaria) | 0.00 en la clase "sobrevive" (accuracy 0.61) |
| Regresión logística | 0.771 |
| KNN | 0.760 |
| Árbol de decisión | 0.750 |
| Random forest | 0.776 |
| **Gradient boosting** | **0.780** |
| Voting (hard) | 0.778 |
| Voting (soft) | 0.771 |

Las diferencias entre los mejores modelos son menores que la desviación entre folds (±0.02–0.04), así que están prácticamente empatados.

- **Accuracy en hold-out:** _pendiente de rellenar tras ejecutar el notebook_
- **Score en Kaggle:** _pendiente_

## Cómo ejecutarlo

```bash
git clone https://github.com/llorenccobos/titanic-kaggle.git
cd titanic-kaggle
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Descarga los datos siguiendo [`data/README.md`](data/README.md) y abre `notebooks/01_titanic.ipynb`.
El notebook genera `submissions/submission.csv`.

## Estructura

```
├── data/            # train.csv y test.csv (no incluidos, ver data/README.md)
├── notebooks/       # análisis y modelado
├── submissions/     # predicciones para Kaggle
├── requirements.txt
└── LICENSE
```

## Licencia

MIT
