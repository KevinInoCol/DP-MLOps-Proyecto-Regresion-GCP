# Estilo de Pipeline de Referencia del Usuario

Este documento describe la estructura exacta que el usuario prefiere para sus pipelines de ML.
Los agentes deben seguir este patrón al generar notebooks de pipeline.

## Estructura del Pipeline

El pipeline se construye como un `sklearn.pipeline.Pipeline` con pasos nombrados.
Las variables se definen como CONSTANTES en mayúsculas al inicio del notebook.
Se usan custom transformers (`TemporalVariableTransformer`, `Mapper`) definidos en un archivo `preprocessors.py` aparte.

## Orden de los pasos del Pipeline

```python
price_pipe = Pipeline([
    # ========================== IMPUTATION ==========================
    ('missing_imputation', CategoricalImputer(
        imputation_method='missing', variables=CATEGORICAL_VARS_WITH_NA_MISSING)),
    ('frequent_imputation', CategoricalImputer(
        imputation_method='frequent', variables=CATEGORICAL_VARS_WITH_NA_FREQUENT)),
    ('missing_indicator', AddMissingIndicator(variables=NUMERICAL_VARS_WITH_NA)),
    ('mean_imputation', MeanMedianImputer(
        imputation_method='mean', variables=NUMERICAL_VARS_WITH_NA)),

    # == TEMPORAL VARIABLES ====
    ('elapsed_time', pp.TemporalVariableTransformer(
        variables=TEMPORAL_VARS, reference_variable=REF_VAR)),
    ('drop_features', DropFeatures(features_to_drop=[REF_VAR])),

    # ===================== VARIABLE TRANSFORMATION ======================
    ('log', LogTransformer(variables=NUMERICALS_LOG_VARS)),
    ('yeojohnson', YeoJohnsonTransformer(variables=NUMERICALS_YEO_VARS)),
    ('binarizer', SklearnTransformerWrapper(
        transformer=Binarizer(threshold=0), variables=BINARIZE_VARS)),

    # =========================== MAPPERS ===============================
    ('mapper_qual', pp.Mapper(variables=QUAL_VARS, mappings=QUAL_MAPPINGS)),
    ('mapper_exposure', pp.Mapper(variables=EXPOSURE_VARS, mappings=EXPOSURE_MAPPINGS)),
    # ... más mappers según se necesiten ...

    # == CATEGORICAL ENCODING ==
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.01, n_categories=1, variables=CATEGORICAL_VARS)),
    ('categorical_encoder', OrdinalEncoder(
        encoding_method='ordered', variables=CATEGORICAL_VARS)),
])
```

## Custom Transformers (preprocessors.py)

```python
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X


class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)
        return X
```

## Convenciones importantes

1. Las variables se definen como CONSTANTES en mayúsculas (CATEGORICAL_VARS_WITH_NA_MISSING, etc.)
2. Se separa train/test con test_size=0.1, random_state=0
3. El target se transforma con np.log() antes del fit
4. MSSubClass se castea a string: data['MSSubClass'] = data['MSSubClass'].astype(str)
5. Se verifica ausencia de NaN después del transform
6. El pipeline se serializa con joblib.dump()
7. Los mappings (QUAL_MAPPINGS, etc.) se definen como diccionarios Python
8. Se usan comentarios con === como separadores de secciones en el Pipeline
