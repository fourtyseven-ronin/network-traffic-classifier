import logging
import yaml
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


class TrafficPredictor:
    def __init__(self, models_dir: str | Path, config: dict):
        self.models_dir = Path(models_dir)
        self.config = config
        
        # Загружаем все артефакты обучения
        logger.info("Загрузка моделей и артефактов...")
        self.stage1_model = xgb.Booster()
        self.stage1_model.load_model(self.models_dir / "stage1_xgb_model.json")
        
        self.stage2_model = xgb.Booster()
        self.stage2_model.load_model(self.models_dir / "stage2_xgb_model.json")
        
        # Список колонок, на которых училась модель (для выравнивания)
        self.feature_columns = joblib.load(self.models_dir / "feature_columns.pkl")
        # LabelEncoder для расшифровки классов Stage 2
        self.le_stage2 = joblib.load(self.models_dir / "stage2_label_encoder.pkl")

    def preprocess_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка сырых данных для инференса.
        Повторяет логику DataCleaner, но адаптирована под отсутствие колонки Label.
        """
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        # 1. Заменяем бесконечности
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if df[col].dtype == object:  # Если inf превратил колонку в строку
                df[col] = df[col].astype(str).str.replace(r"^\s*[+-]?inf(inity)?\s*$", "NaN", case=False, regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
        # 2. Клипаем отрицательные сетевые метрики (наш инсайт из EDA)
        for col in ["Flow Bytes/s", "Flow Duration"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
                
        # 3. Заполняем NaN (в проде лучше использовать сохраненные медианы обучения, 
        # но для MVP берем текущую медиану, чтобы не усложнять)
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
            
        # 4. Строгое выравнивание колонок!
        # Если каких-то колонок нет — создаем их с нулями. Лишние — дропаем.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[self.feature_columns].astype("float32")

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Основной пайплайн двухстадийного предсказания."""
        # Шаг 1: Предобработка
        X = self.preprocess_inference_data(raw_df)
        
        # Создаем массив под финальные ответы (изначально заполняем пустой строкой)
        final_predictions = np. those = np.empty(len(raw_df), dtype=object)
        
        # Переводим данные в формат DMatrix для XGBoost
        dmatrix_all = xgb.DMatrix(X)
        
        # Шаг 2: Предсказание Stage 1
        logger.info("Запуск Stage 1 (Определение макро-классов)...")
        stage1_probs = self.stage1_model.predict(dmatrix_all)
        stage1_preds = np.argmax(stage1_probs, axis=1) # Получаем ID классов (0, 1, 2)
        
        # Мапинг Stage 1 (0: BENIGN, 1: Other_Attack, 2: Web_Bot)
        # ВАЖНО: Проверь порядок классов в своем train.py, он должен совпадать с маппингом!
        stage1_mapping = {0: "BENIGN", 1: "Other_Attack", 2: "Web_Bot"}
        stage1_labels = np.vectorize(stage1_mapping.get)(stage1_preds)
        
        # Записываем в финальные ответы BENIGN и Web_Bot
        final_predictions[stage1_labels == "BENIGN"] = "BENIGN"
        final_predictions[stage1_labels == "Web_Bot"] = "Web_Bot"
        
        # Шаг 3: Маршрутизация на Stage 2 (Только для "Other_Attack")
        stage2_mask = (stage1_labels == "Other_Attack")
        count_stage2 = np.sum(stage2_mask)
        
        if count_stage2 > 0:
            logger.info(f"Stage 1 выявил подозрительный трафик. Отправка {count_stage2} строк на Stage 2...")
            
            # Берем только те строки, которые отфильтровал Stage 1
            X_stage2 = X[stage2_mask]
            dmatrix_stage2 = xgb.DMatrix(X_stage2)
            
            # Предсказание Stage 2 (Мультиклассификация конкретных атак)
            stage2_probs = self.stage2_model.predict(dmatrix_stage2)
            stage2_preds_encoded = np.argmax(stage2_probs, axis=1)
            
            # Декодируем обратно в строки (DDoS, PortScan и т.д.)
            stage2_labels = self.le_stage2.inverse_transform(stage2_preds_encoded)
            
            # Записываем результаты в общую таблицу
            final_predictions[stage2_mask] = stage2_labels
        else:
            logger.info("Подозрительный трафик не обнаружен. Stage 2 пропущен.")
            
        # Собираем красивый итоговый DataFrame
        result_df = pd.DataFrame({
            "Stage1_Decision": stage1_labels,
            "Final_Prediction": final_predictions
        })
        
        return result_df


def get_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if any((parent / m).exists() for m in [".git", "pyproject.toml", "README.md"]):
            return parent
    raise FileNotFoundError("Корень проекта не найден.")


def main():
    base_dir = get_project_root()
    
    # Загрузка конфига
    with open(base_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    logger.info("=== Запуск скрипта предикта (Inference Stage) ===")
    
    # Пути к моделям и к новым данным
    models_dir = base_dir / config["outputs"]["rel_path"]
    
    # Имулируем появление новых данных (для теста можно подсунуть сырой кусок из data/raw)
    # Замени "test_traffic.csv" на имя реального файла, который хочешь протестировать
    new_data_path = base_dir / "data" / "raw" / "test_traffic.csv" 
    
    if not new_data_path.exists():
        logger.warning(f"Файл {new_data_path} не найден. Для теста predict.py подложите туда .csv файл.")
        return

    # Загружаем сырые данные
    raw_data = pd.read_csv(new_data_path, low_memory=False)
    
    # Инициализируем предиктор и запускаем
    predictor = TrafficPredictor(models_dir=models_dir, config=config)
    predictions = predictor.predict(raw_data)
    
    # Сохраняем результаты предсказаний
    output_predictions_path = models_dir / "inference_results.csv"
    predictions.to_csv(output_predictions_path, index=False)
    logger.info(f"=== Предсказания успешно сохранены в {output_predictions_path} ===")


if __name__ == "__main__":
    main()