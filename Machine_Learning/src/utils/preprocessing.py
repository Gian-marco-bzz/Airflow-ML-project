"""
Módulo de preprocesamiento con salvaguardas anti-NaN.

Estrategia fit/transform (sklearn-compatible):
- fit(): Aprende parámetros SOLO del train set (medianas, modas, encoding, escalado)
- transform(): Aplica exactamente esos parámetros (no aprende, solo transforma)

Garantías:
- NO hay data leakage (test/inference usan parámetros de train)
- NO salen NaNs ni Infs del transformador
- Todas las transformaciones son validadas post-aplicación
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import joblib

logger = logging.getLogger(__name__)


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline de preprocesamiento robusto con garantía de limpieza.

    Procesa features numéricas y categóricas en dos caminos separados:

    **Numéricas:**
    1. Normaliza inf → NaN
    2. Imputa medianas (aprendidas en fit)
    3. Escalado StandardScaler (aprendido en fit)

    **Categóricas:**
    1. Normaliza strings raros (nan, None, NULL, etc.) → NaN
    2. Imputa moda (aprendida en fit)
    3. Reduce cardinalidad: categorías < min_freq → 'OTHER'
    4. Target encoding: media del target por categoría

    **Salvaguardas críticas:**
    - Valida que NO hay NaNs al salir de fit() y transform()
    - Reemplaza Infs por NaN, luego imputa con 0.0
    - Reporta por logger si hay anomalías

    Attributes:
        min_category_freq (int): Umbral de frecuencia para cardinality reduction
        num_cols (list): Features numéricas (detectadas en fit)
        cat_cols (list): Features categóricas (detectadas en fit)
        category_maps (dict): Mapeo {col: set(valid_categories)}
        num_imputer, cat_imputer, scaler, encoder: Transformadores sklearn

    Example:
        >>> prep = Preprocessor(min_category_freq=20)
        >>> prep.fit(X_train, y_train)
        >>> X_clean = prep.transform(X_test)
        >>> # Garantizado: X_clean sin NaNs ni Infs
    """

    def __init__(self, min_category_freq: int):
        # min_category_freq viene del config (no usamos default invisible)
        self.min_category_freq = min_category_freq

        # Se rellenan en fit()
        self.num_cols = []
        self.cat_cols = []

        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = None

        # Para reducir cardinalidad: guardamos categorías permitidas por columna
        self.category_maps = {}

    def _normalize_nulls(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out = X_out.replace(
            r"^\s*(nan|NaN|NAN|none|None|NONE|null|NULL)?\s*$",
            np.nan,
            regex=True
        )
        # Normaliza infinitos para que sean tratados por los imputers.
        return X_out.replace([np.inf, -np.inf], np.nan)

    def _reduce_cardinality_fit(self, X: pd.DataFrame) -> pd.DataFrame:

        """
        Aprende qué categorías mantener (frecuencia >= min_category_freq).
        Las demás pasan a 'OTHER'.
        """
        X_out = X.copy()

        for col in self.cat_cols:
            # Convertimos una vez a string para consistencia
            s = X_out[col].astype(str)

            freq = s.value_counts(dropna=False)
            valid = set(freq[freq >= self.min_category_freq].index)

            self.category_maps[col] = valid

            X_out[col] = s.where(s.isin(valid), "OTHER")

        return X_out

    def _reduce_cardinality_transform(self, X: pd.DataFrame) -> pd.DataFrame:

        """
        Aplica el mismo mapping aprendido en fit().
        """
        X_out = X.copy()

        for col in self.cat_cols:
            valid = self.category_maps.get(col, set())

            s = X_out[col].astype(str)

            X_out[col] = s.where(s.isin(valid), "OTHER")

        return X_out

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Aprende parámetros de transformación usando X_train + y_train.

        **IMPORTANTE**: Aprende SÓLO del training set, nunca del test/inference.

        Aprende:
        1. Valores de imputación (medianas numéricas, modas categóricas)
        2. Categorías válidas (frecuencia >= min_category_freq)
        3. Target encoding (media del target por categoría)
        4. Parámetros de escalado (media, std de cada feature)

        Args:
            X: Features de entrenamiento (DataFrame)
            y: Target para target encoding (Series, requerido para cat_cols)

        Returns:
            self (fitted preprocessor, listo para transform)

        Raises:
            ValueError: Si X está vacío después de normalizar NaNs
            ValueError: Si salen NaNs después de fit (bug en transformaciones)
        """
        # Pre-fit validation
        if X.empty:
            raise ValueError("X_train no puede estar vacío")

        logger.info(f"[fit] Iniciando con X_train={X.shape}")

        # PASO 0: Normalizar Infs, NaNs raros, etc.
        X = self._normalize_nulls(X)

        if X.empty:
            raise ValueError("X_train vacío después de normalizar NaNs")

        # PASO 1: Detectar tipos de columnas (una sola vez)
        self.num_cols = X.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()
        self.cat_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        logger.info(
            f"[fit] Detected {len(self.num_cols)} numeric, "
            f"{len(self.cat_cols)} categorical cols"
        )

        X_fit = X.copy()

        # PASO 2: Imputación numérica
        if self.num_cols:
            X_fit[self.num_cols] = self.num_imputer.fit_transform(
                X_fit[self.num_cols]
            )
            nan_after_num = X_fit[self.num_cols].isna().sum().sum()
            if nan_after_num > 0:
                logger.warning(
                    f"[fit] {nan_after_num} NaNs after numeric imputation "
                    f"(median strategy), forcing fillna(0)"
                )
                X_fit[self.num_cols] = X_fit[self.num_cols].fillna(0.0)

        # PASO 3: Imputación categórica + TargetEncoding
        if self.cat_cols:
            X_fit[self.cat_cols] = self.cat_imputer.fit_transform(
                X_fit[self.cat_cols]
            )

            nan_after_cat = X_fit[self.cat_cols].isna().sum().sum()
            if nan_after_cat > 0:
                logger.warning(
                    f"[fit] {nan_after_cat} NaNs after categorical imputation "
                    f"(most_frequent strategy), forcing fillna('OTHER')"
                )
                X_fit[self.cat_cols] = X_fit[self.cat_cols].fillna("OTHER")

            # Cardinality reduction
            X_fit = self._reduce_cardinality_fit(X_fit)

            # Target encoding
            self.encoder = ce.TargetEncoder(cols=self.cat_cols)
            X_fit = self.encoder.fit_transform(X_fit, y)

            nan_after_encoding = X_fit[self.cat_cols].isna().sum().sum()
            if nan_after_encoding > 0:
                logger.warning(
                    f"[fit] {nan_after_encoding} NaNs after target encoding, "
                    f"forcing fillna(0)"
                )
                X_fit[self.cat_cols] = X_fit[self.cat_cols].fillna(0.0)

        # PASO 4: Escalado numérico
        if self.num_cols:
            X_fit[self.num_cols] = self.scaler.fit_transform(X_fit[self.num_cols])

        # PASO 5: Salvaguarda final anti-NaN/Inf
        X_fit = X_fit.replace([np.inf, -np.inf], np.nan)
        nan_final = X_fit.isna().sum().sum()

        if nan_final > 0:
            logger.warning(
                f"[fit] {nan_final} NaNs detected after ALL transformations, "
                f"forcing fillna(0)"
            )
            X_fit = X_fit.fillna(0.0)

        # Post-fit validation
        nan_count_final = X_fit.isna().sum().sum()
        inf_count_final = int(
            np.isinf(X_fit.select_dtypes(include=[np.number]).to_numpy()).sum()
        )

        if nan_count_final > 0 or inf_count_final > 0:
            raise ValueError(
                f"[fit] CRITICAL: Salieron {nan_count_final} NaNs y "
                f"{inf_count_final} Infs del preprocessor"
            )

        logger.info(
            f"[fit] ✓ Fit completed successfully. "
            f"Output shape: {X_fit.shape}, NaNs: 0, Infs: 0"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones aprendidas en fit().

        **IMPORTANTE**: NO aprende nada nuevo aquí. Aplica EXACTAMENTE los
        parámetros aprendidos del train set (no hay data leakage).

        Garantiza:
        - Aplicación consistente con fit()
        - 0 NaNs en el output
        - 0 Infs en el output

        Args:
            X: Features a transformar (test/inference)

        Returns:
            X_transformed: DataFrame limpio sin NaNs ni Infs

        Raises:
            ValueError: Si salen NaNs/Infs del transformador (bug)
        """
        if not self.num_cols and not self.cat_cols:
            raise ValueError(
                "Preprocessor no ha sido fitted. Llama a fit() primero"
            )

        logger.info(f"[transform] Transformando X={X.shape}")

        # PASO 0: Normalizar Infs, NaNs raros, etc.
        X = self._normalize_nulls(X)

        X_tr = X.copy()

        # PASO 1: Imputación numérica (usa valores aprendidos en fit)
        if self.num_cols:
            X_tr[self.num_cols] = self.num_imputer.transform(X_tr[self.num_cols])

            nan_after_num = X_tr[self.num_cols].isna().sum().sum()
            if nan_after_num > 0:
                logger.warning(
                    f"[transform] {nan_after_num} NaNs after numeric imputation, "
                    f"forcing fillna(0)"
                )
                X_tr[self.num_cols] = X_tr[self.num_cols].fillna(0.0)

        # PASO 2: Imputación categórica + TargetEncoding
        if self.cat_cols:
            X_tr[self.cat_cols] = self.cat_imputer.transform(X_tr[self.cat_cols])

            nan_after_cat = X_tr[self.cat_cols].isna().sum().sum()
            if nan_after_cat > 0:
                logger.warning(
                    f"[transform] {nan_after_cat} NaNs after categorical imputation, "
                    f"forcing fillna('OTHER')"
                )
                X_tr[self.cat_cols] = X_tr[self.cat_cols].fillna("OTHER")

            # Cardinality reduction (usa mapas aprendidos)
            X_tr = self._reduce_cardinality_transform(X_tr)

            # Target encoding (usa encoder ya entrenado)
            X_tr = self.encoder.transform(X_tr)

            nan_after_encoding = X_tr[self.cat_cols].isna().sum().sum()
            if nan_after_encoding > 0:
                logger.warning(
                    f"[transform] {nan_after_encoding} NaNs after target encoding, "
                    f"forcing fillna(0)"
                )
                X_tr[self.cat_cols] = X_tr[self.cat_cols].fillna(0.0)

        # PASO 3: Escalado (usa scaler ya entrenado)
        if self.num_cols:
            X_tr[self.num_cols] = self.scaler.transform(X_tr[self.num_cols])

        # PASO 4: Salvaguarda final anti-NaN/Inf
        X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
        nan_final = X_tr.isna().sum().sum()

        if nan_final > 0:
            logger.warning(
                f"[transform] {nan_final} NaNs after ALL transformations, "
                f"forcing fillna(0)"
            )
            X_tr = X_tr.fillna(0.0)

        # Post-transform validation
        nan_count_final = X_tr.isna().sum().sum()
        inf_count_final = int(
            np.isinf(X_tr.select_dtypes(include=[np.number]).to_numpy()).sum()
        )

        if nan_count_final > 0 or inf_count_final > 0:
            raise ValueError(
                f"[transform] CRITICAL: Salieron {nan_count_final} NaNs y "
                f"{inf_count_final} Infs del preprocessor"
            )

        logger.info(
            f"[transform] ✓ Transform completed successfully. "
            f"Output shape: {X_tr.shape}, NaNs: 0, Infs: 0"
        )

        return X_tr

    def save(self, path: str) -> None:

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocessor":

        return joblib.load(path)