from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import pandera as pa
from pandera.typing import DataFrame
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TrafficInputSchema(pa.DataFrameModel):
    Destination_Port: int = pa.Field(coerce=True, ge=0, le=65535)
    Flow_Duration: float = pa.Field(coerce=True)
    
    Flow_Bytes_s: float = pa.Field(coerce=True, nullable=True)
    
    class Config:
        strict=False
        

@dataclass(frozen=True)
class ModelArtifacts:
    stage1_model: xgb.Booster
    stage2_model: xgb.Booster
    feature_columns: list[str]
    median_cols: dict[str, float]
    le1_stage: LabelEncoder
    le2_stage: LabelEncoder


class ArtifactsLoader:
    @staticmethod
    def load_from_local_dir(models_dir: Path) -> ModelArtifacts:
        """Загрузка артефактов c папки."""
        
        logger.info(f"Загрузка артефактов из директории: {models_dir}")
        
        stage1_model = xgb.Booster()
        stage1_model.load_model(models_dir / "stage1_xgb_model.json")
        
        stage2_model = xgb.Booster()
        stage2_model.load_model(models_dir / "stage2_xgb_model.json")
        
        return ModelArtifacts(stage1_model=stage1_model,
                              stage2_model=stage2_model,
                              feature_columns=joblib.load(models_dir / "feature_columns.pkl"),
                              median_cols = joblib.load(models_dir / "median_columns.pkl"),
                              le1_stage=joblib.load(models_dir / "stage1_label_encoder.pkl"),
                              le2_stage=joblib.load(models_dir / "stage2_label_encoder.pkl"))


class TrafficPredictor:
    def __init__(self, artifacts: ModelArtifacts):
        self.artifacts = artifacts

    def _preprocess_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Шаг 1: Очистка, клиппинг, reindex и fillna."""
        df = raw_df.copy()
        df.columns = df.columns.str.strip()

        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = np.where(np.isinf(df[numeric_cols]), np.nan, df[numeric_cols])

        for col in ("Flow Bytes/s", "Flow Duration"):
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        df = df.reindex(columns=self.artifacts.feature_columns)
        df = df.fillna(self.artifacts.median_cols)
        return df.astype("float32")

    def _run_inference(self, X: pd.DataFrame) -> np.ndarray:
        """Шаг 2: Каскадный инференс (Stage 1/Stage 2)."""
        dmatrix_all = xgb.DMatrix(X)
        preds_stage1_probs = self.artifacts.stage1_model.predict(dmatrix_all)
        preds_stage1_idx = np.argmax(preds_stage1_probs, axis=1)
        labels = self.artifacts.le1_stage.inverse_transform(preds_stage1_idx)

        # Stage 2 маска
        stage2_mask = (labels != "BENIGN") & (labels != "Web_Bot")
        
        if stage2_mask.any():
            X_stage2 = X[stage2_mask]
            dmatrix_stage2 = xgb.DMatrix(X_stage2)
            
            preds_stage2_probs = self.artifacts.stage2_model.predict(dmatrix_stage2)
            preds_stage2_idx = np.argmax(preds_stage2_probs, axis=1)
            labels[stage2_mask] = self.artifacts.le2_stage.inverse_transform(preds_stage2_idx)

        return labels

    @pa.check_types # Декоратор Pandera стоит ТОЛЬКО на входе в класс
    def __call__(self, raw_df: DataFrame[TrafficInputSchema]) -> pd.DataFrame:
        """Управление шагами обработки."""
        if raw_df.empty:
            logger.warning("Получен пустой DataFrame.")
            return pd.DataFrame(columns=["Predicted_Label"])

        X = self._preprocess_data(raw_df)
        final_labels = self._run_inference(X)

        return pd.DataFrame({"Predicted_Label": final_labels}, index=raw_df.index)        