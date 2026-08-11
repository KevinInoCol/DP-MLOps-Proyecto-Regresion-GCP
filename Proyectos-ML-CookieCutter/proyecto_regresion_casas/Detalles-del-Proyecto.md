# Proyecto MLOps: Prediccion del Precio de Casas (Regresion)

## Descripcion General

Este proyecto implementa un pipeline completo de Machine Learning bajo principios de MLOps para predecir el precio de venta de casas residenciales. Se utiliza el dataset "House Prices: Advanced Regression Techniques" de Kaggle, que contiene 1460 registros de entrenamiento con 79 variables predictoras (numericas y categoricas) y la variable objetivo `SalePrice`.

El proyecto esta organizado siguiendo la estructura **Cookiecutter Data Science**, un estandar de la industria que promueve la reproducibilidad, mantenibilidad y colaboracion en proyectos de ciencia de datos.

---

## Arquitectura del Proyecto (Cookiecutter Data Science)

```
proyecto_regresion_casas/
|
|-- config/                  # Configuracion con Hydra (YAML)
|   |-- main.yaml            # Configuracion principal
|   |-- model/               # Configs de modelos (model1.yaml, model2.yaml)
|   |-- process/             # Configs de procesamiento (process1.yaml, process2.yaml)
|
|-- data/
|   |-- raw/                 # Datos crudos originales (train.csv, test.csv de Kaggle)
|   |-- processed/           # Datos transformados (xtrain.csv, xtest.csv, ytrain.csv, ytest.csv)
|   |-- final/               # Datos finales listos para produccion
|
|-- models/                  # Modelos serializados (.joblib)
|   |-- linear_regression.joblib
|   |-- precio_casas_pipeline.joblib
|
|-- notebooks/               # Jupyter Notebooks ordenados secuencialmente
|   |-- 01 a 08              # Pipeline completo de experimentacion
|   |-- preprocessors.py     # Transformadores custom de sklearn
|
|-- src/                     # Codigo fuente modularizado
|   |-- process.py           # Logica de procesamiento
|   |-- train_model.py       # Logica de entrenamiento
|   |-- utils.py             # Utilidades comunes
|
|-- tests/                   # Tests unitarios
|   |-- test_process.py
|   |-- test_train_model.py
|
|-- requirements.txt         # Dependencias del proyecto
|-- pyproject.toml           # Configuracion del proyecto Python
```

### Principios MLOps aplicados en esta estructura

- **Separacion de datos por etapa**: raw -> processed -> final. Los datos crudos nunca se modifican.
- **Configuracion externalizada**: Hydra permite cambiar hiperparametros sin tocar codigo.
- **Reproducibilidad**: Seeds fijos en cada paso aleatorio, pipelines serializados con joblib.
- **Modularizacion**: Transformadores custom en `preprocessors.py`, codigo productivo en `src/`.
- **Testing**: Directorio `tests/` con pruebas unitarias.

---

## Pipeline de Notebooks: Paso a Paso

### Notebook 01: Data Analysis (Analisis Exploratorio de Datos - EDA)

**Objetivo**: Comprender el dataset antes de cualquier transformacion.

**Que se hace**:
1. Se carga `data/raw/train.csv` (1460 filas, 81 columnas).
2. Se elimina la columna `Id` (solo es un identificador).
3. **Analisis del Target (`SalePrice`)**:
   - Se grafica el histograma: distribucion sesgada a la derecha.
   - Se aplica transformacion logaritmica: `np.log(SalePrice)` produce una distribucion mas gaussiana, ideal para modelos lineales.
4. **Identificacion de variables**:
   - 43 variables categoricas (incluyendo `MSSubClass` que es numerica por valores pero categorica por definicion).
   - 36 variables numericas.
5. **Analisis de Missing Values**:
   - Se identifica el porcentaje de datos faltantes por variable.
   - Variables como `PoolQC`, `MiscFeature`, `Alley`, `Fence` tienen >80% missing (la ausencia tiene significado: "no tiene pool", "no tiene fence").
   - Se separan los missing en categoricos vs numericos para definir estrategias de imputacion.
6. **Analisis de variables numericas**:
   - Distribucion temporal (YearBuilt, YearRemodAdd, GarageYrBlt).
   - Variables discretas vs continuas.
   - Deteccion de variables con distribucion sesgada (skewed).
   - Analisis de outliers.
7. **Analisis de variables categoricas**:
   - Cardinalidad (numero de categorias unicas).
   - Relacion entre categorias y el precio de venta.

**Herramientas**: pandas, numpy, matplotlib, seaborn, scipy.

---

### Notebook 02: Feature Engineering (Python Puro)

**Objetivo**: Aplicar todas las transformaciones de variables manualmente, paso a paso, para entender cada tecnica antes de automatizarla en un pipeline.

**Que se hace**:

1. **Separacion Train/Test** (90/10) con `train_test_split` y seed fijo (`random_state=0`).

2. **Transformacion del Target**: `y_train = np.log(y_train)`, `y_test = np.log(y_test)`.

3. **Identificacion de variables categoricas**:
   - Se usa `pd.api.types.is_string_dtype()` para detectar columnas de texto (compatible con pandas moderno que usa StringDtype).
   - `MSSubClass` se agrega manualmente a la lista de categoricas.

4. **Imputacion de Missing Values - Variables Categoricas**:
   - Variables con >10% missing: se imputan con el string `"Missing"` (la ausencia tiene significado).
   - Variables con <10% missing: se imputan con la **moda** (categoria mas frecuente).

5. **Imputacion de Missing Values - Variables Numericas**:
   - Se crea un **indicador binario** (`variable_na`) que indica si el valor era faltante (1) o no (0).
   - Se imputa con la **media** calculada sobre el conjunto de entrenamiento.

6. **Variables Temporales**:
   - Se calcula el **tiempo transcurrido**: `YrSold - YearBuilt`, `YrSold - YearRemodAdd`, `YrSold - GarageYrBlt`.
   - Se elimina `YrSold` despues de usarla como referencia.

7. **Transformaciones de variables numericas**:
   - **Log transform**: `LotFrontage`, `1stFlrSF`, `GrLivArea` para reducir sesgo.
   - **Yeo-Johnson transform**: `LotArea` (generaliza Box-Cox, permite valores negativos).

8. **Binarizacion de variables sesgadas**:
   - Variables como `BsmtFinSF2`, `LowQualFinSF`, `ScreenPorch` tienen >90% de ceros.
   - Se convierten a binarias (0/1): tiene o no tiene.

9. **Mapeo ordinal de variables categoricas con orden inherente**:
   - Variables de calidad (ExterQual, BsmtQual, etc.): Po=1, Fa=2, TA=3, Gd=4, Ex=5.
   - BsmtExposure: No=1, Mn=2, Av=3, Gd=4.
   - BsmtFinType, GarageFinish, Fence: mapeos especificos segun nivel.

10. **Encoding de variables categoricas nominales**:
    - **Rare Label Encoding**: categorias con frecuencia <1% se agrupan como "Rare".
    - **Ordinal Encoding por target**: se reemplaza cada categoria por su posicion segun el precio medio de venta (la categoria con precio mas bajo = 0, la siguiente = 1, etc.).

11. **Escalado**: `MinMaxScaler` para normalizar todas las variables al rango [0, 1].

12. **Persistencia**: Se guardan los datasets transformados en `data/processed/`:
    - `xtrain.csv`, `xtest.csv`, `ytrain.csv`, `ytest.csv`
    - `minmax_scaler.joblib` (el scaler entrenado)

**Herramientas**: pandas, numpy, matplotlib, scipy, scikit-learn.

---

### Notebook 03: Feature Engineering Pipeline (SKLearn + Feature-Engine)

**Objetivo**: Encapsular TODOS los pasos del Notebook 02 en un unico **Pipeline de scikit-learn**, usando transformadores de la libreria `feature-engine` y transformadores custom.

**Que se hace**:

1. Se define la **configuracion** como constantes (listas de variables, mappings).

2. Se construye un `Pipeline` de sklearn con los siguientes pasos encadenados:
   - `CategoricalImputer` (missing) -> imputa con "Missing"
   - `CategoricalImputer` (frequent) -> imputa con la moda
   - `AddMissingIndicator` -> crea columnas `_na` binarias
   - `MeanMedianImputer` -> imputa numericas con la media
   - `TemporalVariableTransformer` (custom) -> calcula tiempo transcurrido
   - `DropFeatures` -> elimina `YrSold`
   - `LogTransformer` -> transforma variables sesgadas
   - `YeoJohnsonTransformer` -> transforma `LotArea`
   - `SklearnTransformerWrapper(Binarizer)` -> binariza variables
   - `Mapper` (custom) -> mapeos ordinales para calidad, exposicion, etc.
   - `RareLabelEncoder` -> agrupa categorias infrecuentes como "Rare"
   - `OrdinalEncoder` -> encoding por target

3. Se entrena el pipeline con `price_pipe.fit(X_train, y_train)`.
4. Se transforma ambos conjuntos: `price_pipe.transform(X_train)`, `price_pipe.transform(X_test)`.

**Ventaja clave**: Un solo objeto `price_pipe` encapsula TODO el feature engineering. Se puede serializar y reutilizar en produccion.

**Transformadores custom** (`preprocessors.py`):
- `TemporalVariableTransformer`: calcula `reference_variable - variable` para variables temporales.
- `Mapper`: aplica diccionarios de mapeo a variables categoricas ordinales.

**Herramientas**: scikit-learn, feature-engine, pandas, numpy, joblib.

---

### Notebook 04: Feature Selection (Seleccion de Variables)

**Objetivo**: Reducir el numero de variables predictoras seleccionando solo las mas relevantes.

**Que se hace**:

1. Se cargan los datasets procesados desde `data/processed/`.

2. Se usa **Lasso Regression** (`alpha=0.001`) como metodo de seleccion:
   - Lasso aplica regularizacion L1 que fuerza coeficientes irrelevantes a exactamente cero.
   - `SelectFromModel` de sklearn selecciona automaticamente las variables con coeficiente != 0.

3. Se obtienen **36 variables seleccionadas** de las 81 originales (reduccion >55%).

4. Se guardan las variables seleccionadas en `data/processed/selected_features.csv`.

**Por que Lasso para Feature Selection**:
- La penalizacion L1 produce **sparsity** (coeficientes exactamente cero).
- Es un metodo **embedded**: la seleccion ocurre durante el entrenamiento.
- Con `alpha=0.001` se obtiene un balance entre seleccionar suficientes variables y eliminar ruido.

**Herramientas**: scikit-learn (Lasso, SelectFromModel), pandas, numpy, matplotlib.

---

### Notebook 05: Best Model with PyCaret (Seleccion Automatizada de Modelos)

**Objetivo**: Usar AutoML con PyCaret para comparar multiples algoritmos de regresion y encontrar el mejor modelo automaticamente.

**Que se hace**:

1. Se carga un dataset de ejemplo con PyCaret (`get_data`).

2. Se configura el experimento de regresion con `RegressionExperiment.setup()`:
   - PyCaret automaticamente: infiere tipos de variables, imputa missing values, preprocesa datos.
   - Se fija `session_id=123` para reproducibilidad.
   - Usa **Validacion Cruzada K-Fold** (10 folds por defecto).

3. **Comparacion de modelos** con `compare_models()`:
   - PyCaret entrena y evalua 24+ algoritmos automaticamente.
   - Incluye: Linear Regression, Lasso, Ridge, Elastic Net, KNN, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, LightGBM, SVR, MLP, y mas.
   - Compara por multiples metricas: MAE, MSE, RMSE, R2, RMSLE, MAPE.

4. **Seleccion TOP 3 modelos** ordenados por MSE.

5. **Tuning de hiperparametros** con `tune_model()`:
   - Realiza busqueda de hiperparametros (10 iteraciones, 10 folds = 100 fits).
   - Compara modelo tuneado vs original.

6. **Finalizacion del modelo** con `finalize_model()`:
   - Reentrena el mejor modelo con TODOS los datos (train + test).

7. **Persistencia**: Se guarda el pipeline completo de PyCaret como `.pkl`.

**Herramientas**: PyCaret, scikit-learn, joblib.

---

### Notebook 06: Model Training (Entrenamiento del Modelo Final)

**Objetivo**: Entrenar el modelo Lasso definitivo con las variables seleccionadas y evaluar su rendimiento.

**Que se hace**:

1. Se cargan los datasets procesados y las variables seleccionadas.

2. Se reduce `X_train` y `X_test` a las 36 variables seleccionadas.

3. Se entrena un modelo **Lasso** con `alpha=0.001` y `random_state=0`.

4. **Evaluacion del modelo**:
   - **Train**: MSE, RMSE, R2
   - **Test**: MSE, RMSE, R2
   - Precio promedio de casas como referencia.

5. **Visualizaciones**:
   - Scatter plot: precio real vs precio predicho (evaluacion visual de calidad).
   - Histograma de errores: deben seguir una distribucion gaussiana.

6. **Feature Importance**: Grafico de barras con los coeficientes absolutos del Lasso, mostrando que variables tienen mas impacto en la prediccion.

7. **Persistencia**: `joblib.dump(lin_model, '../models/linear_regression.joblib')`.

**Herramientas**: scikit-learn (Lasso), pandas, numpy, matplotlib, joblib.

---

### Notebook 07: Scoring New Data (Prediccion con Datos Nuevos)

**Objetivo**: Demostrar como usar los artefactos guardados (scaler, modelo) para hacer predicciones sobre datos nunca antes vistos.

**Que se hace**:

1. Se carga `data/raw/test.csv` (datos de test de Kaggle, nunca usados en entrenamiento).

2. Se aplican las mismas transformaciones que al conjunto de entrenamiento:
   - Se usa el `minmax_scaler.joblib` guardado para escalar.
   - Se usa el `linear_regression.joblib` guardado para predecir.

3. Se obtienen predicciones en escala logaritmica y se transforman de vuelta con `np.exp()`.

4. Se visualiza la distribucion de precios predichos.

**Importancia para MLOps**: Este notebook demuestra el flujo de **inferencia en produccion**: cargar artefactos pre-entrenados y aplicarlos a datos nuevos sin reentrenar.

**Herramientas**: pandas, numpy, matplotlib, scipy, joblib.

---

### Notebook 08: Machine Learning Pipeline (Pipeline End-to-End)

**Objetivo**: Construir un **pipeline unico** que integra feature engineering + escalado + modelo, listo para produccion.

**Que se hace**:

1. Se construye un `Pipeline` de sklearn que incluye TODO:
   - Imputacion (categorica y numerica)
   - Indicadores de missing
   - Variables temporales
   - Transformaciones (log, binarizacion)
   - Mapeos ordinales
   - Rare Label Encoding + Ordinal Encoding
   - `MinMaxScaler`
   - **Modelo Lasso** como paso final

2. Se entrena todo el pipeline con `price_pipe.fit(X_train, y_train)`.

3. Se predicen con `price_pipe.predict(X_test)` (una sola llamada hace todo).

4. Se evaluan metricas (identicas al Notebook 06, validando que el pipeline replica los resultados).

5. **Scoring con datos nuevos**: Se carga `test.csv` de Kaggle y se predice directamente con el pipeline.

6. **Persistencia**: `joblib.dump(price_pipe, '../models/precio_casas_pipeline.joblib')`.

**Ventaja clave**: Un unico archivo `.joblib` contiene TODA la logica de preprocesamiento + modelo. En produccion, basta con:
```python
pipe = joblib.load('precio_casas_pipeline.joblib')
predicciones = pipe.predict(datos_nuevos)
```

**Herramientas**: scikit-learn, feature-engine, pandas, numpy, matplotlib, joblib.

---

## Flujo de Datos del Proyecto

```
data/raw/train.csv (1460 filas, 81 columnas)
       |
       v
[NB 01] Analisis Exploratorio (EDA)
       |
       v
[NB 02] Feature Engineering Manual
       |  Salida: data/processed/xtrain.csv, xtest.csv, ytrain.csv, ytest.csv
       |          data/processed/minmax_scaler.joblib
       v
[NB 03] Feature Engineering con Pipeline (sklearn + feature-engine)
       |
       v
[NB 04] Feature Selection con Lasso
       |  Salida: data/processed/selected_features.csv (36 variables)
       v
[NB 05] AutoML con PyCaret (comparacion de 24+ modelos)
       |
       v
[NB 06] Entrenamiento del Modelo Lasso Final
       |  Salida: models/linear_regression.joblib
       v
[NB 07] Scoring con datos nuevos (data/raw/test.csv)
       |
       v
[NB 08] Pipeline End-to-End (feature engineering + modelo en un solo objeto)
       |  Salida: models/precio_casas_pipeline.joblib
       v
   PRODUCCION
```

---

## Artefactos Generados

| Artefacto | Generado por | Ubicacion | Descripcion |
|-----------|-------------|-----------|-------------|
| `xtrain.csv` | NB 02 | `data/processed/` | Features de entrenamiento transformados y escalados |
| `xtest.csv` | NB 02 | `data/processed/` | Features de test transformados y escalados |
| `ytrain.csv` | NB 02 | `data/processed/` | Target de entrenamiento (log-transformado) |
| `ytest.csv` | NB 02 | `data/processed/` | Target de test (log-transformado) |
| `minmax_scaler.joblib` | NB 02 | `data/processed/` | Scaler entrenado para normalizar variables |
| `selected_features.csv` | NB 04 | `data/processed/` | Lista de 36 variables seleccionadas por Lasso |
| `linear_regression.joblib` | NB 06 | `models/` | Modelo Lasso entrenado |
| `precio_casas_pipeline.joblib` | NB 08 | `models/` | Pipeline completo (preprocessing + modelo) |

---

## Tecnologias y Dependencias

| Tecnologia | Uso en el Proyecto |
|------------|-------------------|
| **pandas** | Manipulacion y analisis de datos tabulares |
| **numpy** | Operaciones numericas, transformaciones logaritmicas |
| **matplotlib** | Visualizacion de distribuciones, scatter plots, feature importance |
| **seaborn** | Visualizaciones estadisticas avanzadas (EDA) |
| **scikit-learn** | Pipelines, modelos (Lasso), escalado (MinMaxScaler), feature selection (SelectFromModel), metricas |
| **feature-engine** | Transformadores especializados para imputacion, encoding, transformacion de variables |
| **scipy** | Transformacion Yeo-Johnson |
| **joblib** | Serializacion de modelos y pipelines |
| **PyCaret** | AutoML para comparacion automatizada de modelos |
| **Hydra** | Gestion de configuracion externalizada (YAML) |

---

## Tecnicas de Feature Engineering Aplicadas

| Tecnica | Variables | Justificacion |
|---------|-----------|---------------|
| Imputacion con "Missing" | Alley, FireplaceQu, PoolQC, Fence, MiscFeature | La ausencia tiene significado (no tiene pool, etc.) |
| Imputacion con moda | BsmtQual, BsmtCond, Electrical, GarageType, etc. | Missing aleatorio, se reemplaza con valor mas comun |
| Imputacion con media + indicador binario | LotFrontage, MasVnrArea, GarageYrBlt | Preserva la informacion de que el dato era faltante |
| Log transform | LotFrontage, 1stFlrSF, GrLivArea | Reduce sesgo positivo, mejora linealidad |
| Yeo-Johnson transform | LotArea | Generalizacion de Box-Cox para cualquier distribucion |
| Binarizacion | BsmtFinSF2, ScreenPorch, etc. | Variables con >90% ceros, se simplifican a tiene/no tiene |
| Mapeo ordinal | ExterQual, BsmtQual, HeatingQC, etc. | Variables con orden inherente (Po < Fa < TA < Gd < Ex) |
| Rare Label Encoding | Todas las categoricas nominales | Categorias con <1% frecuencia se agrupan como "Rare" |
| Ordinal Encoding por target | Todas las categoricas nominales | Encoding monotono basado en el precio medio por categoria |
| MinMaxScaler | Todas las variables | Normaliza al rango [0, 1] para el modelo Lasso |

---

## Conceptos Clave de MLOps Aplicados en el Proyecto

### 1. Reproducibilidad

La reproducibilidad es un pilar fundamental de MLOps. Garantiza que cualquier miembro del equipo pueda obtener los mismos resultados ejecutando el mismo codigo con los mismos datos.

- **Seeds fijos** (`random_state=0`) en cada operacion aleatoria: `train_test_split`, modelo Lasso, PyCaret (`session_id=123`).
- **Versionado de dependencias**: `requirements.txt` con versiones minimas pinneadas.
- **Datos inmutables**: `data/raw/` nunca se modifica; las transformaciones generan archivos nuevos en `data/processed/`.
- **Serializacion de artefactos**: Modelos, scalers y pipelines se guardan con `joblib` para garantizar que la inferencia use exactamente los mismos parametros aprendidos en entrenamiento.

### 2. Separation of Concerns (Separacion de Responsabilidades)

Cada componente del proyecto tiene una responsabilidad unica y bien definida:

- **`data/raw/`**: Datos tal como llegan de la fuente (Kaggle). Nunca se modifican.
- **`data/processed/`**: Datos transformados, listos para modelado. Generados por los notebooks de feature engineering.
- **`notebooks/`**: Experimentacion y prototipado interactivo, ordenados secuencialmente.
- **`src/`**: Codigo productivo modularizado (`process.py`, `train_model.py`, `utils.py`).
- **`models/`**: Artefactos serializados listos para despliegue.
- **`config/`**: Configuracion externalizada con Hydra. Permite cambiar hiperparametros, rutas y comportamientos sin modificar codigo fuente.
- **`tests/`**: Pruebas unitarias que validan la logica de procesamiento y entrenamiento.

### 3. Pipeline como Artefacto de Produccion

El concepto mas importante del proyecto es la evolucion desde codigo manual (NB 02) hasta un **pipeline serializable** (NB 08):

- **NB 02**: Cada paso de feature engineering se ejecuta como codigo suelto. Funciona para experimentacion pero es fragil en produccion (hay que replicar exactamente cada paso).
- **NB 03**: Se encapsulan los mismos pasos en un `Pipeline` de sklearn. El pipeline "aprende" parametros (medias, modas, encodings) del train set y los aplica consistentemente al test set.
- **NB 08**: Se agrega el modelo al pipeline. Un unico objeto `price_pipe` hace TODO: recibe datos crudos y devuelve predicciones. Se serializa en un solo archivo `.joblib`.

**Esto es la base del despliegue en produccion**: el artefacto `precio_casas_pipeline.joblib` se puede cargar en una API (FastAPI, Flask), un servicio cloud, o un batch job, y hacer predicciones con una sola linea de codigo.

### 4. Data Lineage (Trazabilidad de Datos)

El proyecto implementa trazabilidad de datos a traves de la estructura de carpetas y el flujo secuencial de notebooks:

```
Kaggle (fuente externa)
  |
  v
data/raw/train.csv -----> [NB 01: EDA] -----> Entendimiento del dominio
  |
  v
[NB 02: Feature Engineering] -----> data/processed/xtrain.csv, xtest.csv
                                     data/processed/ytrain.csv, ytest.csv
                                     data/processed/minmax_scaler.joblib
  |
  v
[NB 04: Feature Selection] -------> data/processed/selected_features.csv
  |
  v
[NB 06: Model Training] ----------> models/linear_regression.joblib
  |
  v
[NB 08: Full Pipeline] -----------> models/precio_casas_pipeline.joblib
  |
  v
data/raw/test.csv ---> [NB 07: Scoring] ---> Predicciones sobre datos nuevos
```

En cada etapa se puede rastrear: que datos de entrada se usaron, que transformaciones se aplicaron, que artefactos se generaron, y con que parametros.

### 5. Experimentacion vs Produccion

El proyecto demuestra la transicion clasica de MLOps desde experimentacion hacia produccion:

| Fase | Notebooks | Caracteristica |
|------|-----------|----------------|
| **Exploracion** | NB 01 | EDA interactivo, visualizaciones, hipotesis |
| **Prototipado manual** | NB 02 | Feature engineering paso a paso, validacion visual de cada transformacion |
| **Automatizacion** | NB 03 | Mismas transformaciones encapsuladas en un Pipeline reutilizable |
| **Seleccion de modelo** | NB 04, 05 | Feature selection con Lasso + AutoML con PyCaret para comparar 24+ modelos |
| **Entrenamiento final** | NB 06 | Modelo entrenado con features seleccionados, evaluacion formal de metricas |
| **Inferencia** | NB 07 | Scoring con datos nunca vistos usando artefactos serializados |
| **Pipeline productivo** | NB 08 | Pipeline end-to-end listo para despliegue |

### 6. Validacion y Prevencion de Data Leakage

- **Train/Test split ANTES de cualquier transformacion**: Se separan los datos en el NB 02 antes de calcular medias, modas o cualquier estadistico. Esto previene data leakage.
- **Parametros aprendidos solo del train set**: La media para imputacion, la moda, los mappings de categorias, los parametros del scaler y los coeficientes del modelo se aprenden exclusivamente del conjunto de entrenamiento.
- **Transformacion consistente del test set**: El test set se transforma usando los mismos parametros aprendidos del train. Nunca se llama `.fit()` sobre el test set.
- **Validacion cruzada K-Fold**: En PyCaret (NB 05) se usa K-Fold CV para evaluacion robusta, evitando overfitting a un solo split.

### 7. Gestion de Configuracion con Hydra

El proyecto usa **Hydra** para externalizar la configuracion:

- `config/main.yaml`: Rutas de datos (raw, processed, final).
- `config/model/`: Configuraciones de diferentes modelos (model1.yaml, model2.yaml).
- `config/process/`: Configuraciones de diferentes pipelines de procesamiento.

**Beneficio para MLOps**: Permite ejecutar experimentos con diferentes hiperparametros sin cambiar codigo. Facilita la gestion de multiples ambientes (desarrollo, staging, produccion).

### 8. Cookiecutter Data Science como Estandar

La estructura del proyecto sigue **Cookiecutter Data Science**, un template ampliamente adoptado en la industria que establece:

- **Convencion sobre configuracion**: Cualquier data scientist que conozca el template sabe inmediatamente donde encontrar datos, modelos, notebooks y codigo fuente.
- **Separacion clara de datos por etapa**: raw (inmutable) -> processed (intermedio) -> final (listo para consumo).
- **Codigo fuente independiente de notebooks**: `src/` contiene la logica productiva; los notebooks son para experimentacion.
- **Onboarding rapido**: Un nuevo miembro del equipo puede entender la estructura del proyecto en minutos.

### 9. Serializacion de Modelos y Artefactos (Model Registry)

Todos los artefactos entrenados se persisten en disco para su reutilizacion:

| Artefacto | Formato | Contenido |
|-----------|---------|-----------|
| `minmax_scaler.joblib` | joblib | Scaler con min/max aprendidos del train set |
| `selected_features.csv` | CSV | Lista de 36 variables seleccionadas por Lasso |
| `linear_regression.joblib` | joblib | Modelo Lasso con coeficientes entrenados |
| `precio_casas_pipeline.joblib` | joblib | Pipeline completo: preprocessing + modelo |

En un entorno MLOps maduro, estos artefactos se almacenarian en un **Model Registry** (MLflow, Weights & Biases, SageMaker) con versionado, metadata y linaje.

### 10. Metricas de Evaluacion del Modelo

El proyecto evalua el modelo con multiples metricas para tener una vision completa del rendimiento:

- **MSE (Mean Squared Error)**: Penaliza errores grandes cuadraticamente. Util para detectar predicciones muy alejadas.
- **RMSE (Root Mean Squared Error)**: Misma escala que la variable objetivo. Mas interpretable que MSE.
- **R2 (Coeficiente de Determinacion)**: Proporcion de varianza explicada por el modelo (0 a 1). El modelo logra R2 ~0.84 en test.
- **Analisis de residuos**: El histograma de errores debe seguir una distribucion gaussiana, validando los supuestos del modelo lineal.
