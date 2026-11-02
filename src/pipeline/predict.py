import logging
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)


class TrafficPredictor:
    def __init__(self, models_dir: str | Path, config: dict):
        self.models_dir = Path(models_dir)
        self.config = config

        logger.info("Загрузка моделей и артефактов...")
        self.stage1_model = xgb.Booster()
        self.stage1_model.load_model(self.models_dir / "stage1_xgb_model.json")

        self.stage2_model = xgb.Booster()
        self.stage2_model.load_model(self.models_dir / "stage2_xgb_model.json")

        self.feature_columns = joblib.load(self.models_dir / "feature_columns.pkl")
        self.le_stage1 = joblib.load(self.models_dir / "stage1_label_encoder.pkl")
        self.le_stage2 = joblib.load(self.models_dir / "stage2_label_encoder.pkl")
        self.median = joblib.load(self.models_dir / "median_columns.pkl")

    def _preprocess_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных под инференс.
        """
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Замена inf на NaN
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = np.where(np.isinf(df[numeric_cols]), np.nan, df[numeric_cols])

        # Клипаем отрицательные сетевые метрики
        for col in ("Flow Bytes/s", "Flow Duration"):
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Выравниваем колонки: оставляем нужные, создаём недостающие (NaN) и задаём порядок
        df = df.reindex(columns=self.feature_columns)

        # Заполняем NaN медианами
        df = df.fillna(self.median)

        return df.astype("float32")

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Основной пайплайн двухстадийного предсказания."""
        if raw_df.empty:
            raise ValueError("Входной DataFrame пуст.")

        # Предобработка
        X = self._preprocess_inference_data(raw_df)

        # Массив под финальные ответы
        final_predictions = np.full(len(raw_df), "UNKNOWN", dtype=object)

        # Stage 1
        dmatrix_all = xgb.DMatrix(X)
        logger.info("Запуск Stage 1 (Определение макро-классов)...")
        stage1_probs = self.stage1_model.predict(dmatrix_all)
        stage1_preds = np.argmax(stage1_probs, axis=1)

        
        stage1_labels = self.le_stage1.inverse_transform(stage1_preds)

        # NumPy-маски
        benign_mask = (stage1_labels == "BENIGN")
        web_bot_mask = (stage1_labels == "Web_Bot")
        stage2_mask = (stage1_labels == "Other_Attack")

        final_predictions[benign_mask] = "BENIGN"
        final_predictions[web_bot_mask] = "Web_Bot"

        # Stage 2: только для "Other_Attack"
        count_stage2 = np.count_nonzero(stage2_mask)

        if count_stage2 > 0:
            logger.info(
                f"Stage 1 выявил подозрительный трафик. "
                f"Отправка {count_stage2} строк на Stage 2..."
            )

            X_stage2 = xgb.DMatrix(X[stage2_mask])

            stage2_probs = self.stage2_model.predict(X_stage2)
            stage2_preds_encoded = np.argmax(stage2_probs, axis=1)

            # Декодируем обратно в строки (DDoS, PortScan и т.д.)
            stage2_labels = self.le_stage2.inverse_transform(stage2_preds_encoded)

            final_predictions[stage2_mask] = stage2_labels
        else:
            logger.info("Подозрительный трафик не обнаружен. Stage 2 пропущен.")

        # Итоговый датафрейм
        result_df = pd.DataFrame({
            "Stage1_Decision": stage1_labels,
            "Final_Prediction": final_predictions,
        })

        return result_df
    
    def __call__(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return self.predict(raw_df)


def get_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if any((parent / m).exists() for m in [".git", "pyproject.toml", "README.md"]):
            return parent

    raise FileNotFoundError(
        "Корень проекта не найден. "
        "Задайте переменную окружения PROJECT_ROOT или запустите из директории репозитория."
    )


def main():
    base_dir = get_project_root()

    # Загрузка конфига
    with open(base_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Запуск скрипта предикта (Inference Stage)...")

    models_dir = base_dir / config["outputs"]["rel_path"]

    # TODO: вынести путь к данным в аргументы CLI или переменные окружения
    new_data_path = base_dir / "data" / "raw" / "test_traffic.csv"

    if not new_data_path.exists():
        logger.warning(
            f"Файл {new_data_path} не найден. "
            f"Для теста predict.py подложите туда .csv файл."
        )
        return

    # Загрузка данных
    raw_data = pd.read_csv(new_data_path, low_memory=False)

    # Инициализируем и запуск
    predictor = TrafficPredictor(models_dir=models_dir, config=config)
    predictions = predictor(raw_data)

    # Сохраняем результаты отдельно от артефактов моделей
    predictions_dir = base_dir / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_predictions_path = predictions_dir / "inference_results.csv"

    predictions.to_csv(output_predictions_path, index=False)
    logger.info(f"=== Предсказания успешно сохранены в {output_predictions_path} ===")


if __name__ == "__main__":
    main()