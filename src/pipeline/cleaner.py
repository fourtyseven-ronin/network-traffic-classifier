import logging
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class DataCleaner:
    ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
    NON_NEGATIVE_METRICS = ["Flow Bytes/s", "Flow Duration"]

    def __init__(self, raw_data_dir: str | Path, drop_cols: list = None):
        self.raw_dir = Path(raw_data_dir)
        self.files = sorted(self.raw_dir.glob("*.csv"))
        self.drop_cols = drop_cols or ["Fwd Header Length.1"]

    def check_columns(self) -> bool:
        """Проверяет идентичность колонок во всех сырых файлах."""
        if not self.files:
            return False
        base = pd.read_csv(self.files[0], nrows=0).columns
        return all(
            pd.read_csv(f, nrows=0).columns.equals(base) for f in self.files[1:]
        )

    def process_single_file(self, file_path: Path) -> pd.DataFrame:
        """Читает и очищает один файл (колонки, бесконечности, отрицательные значения)."""
        df = pd.read_csv(file_path, low_memory=False)
        df = self.clean_columns(df)
        df = self.clean_rows(df)
        return df

    def run_stage_one(self, output_path: str | Path) -> pd.DataFrame:
        """Главный пайплайн Этапа 1: пофайловая очистка, склейка и заполнение медианой."""
        if not self.files:
            raise FileNotFoundError(f"В {self.raw_dir} нет .csv файлов")

        if not self.check_columns():
            raise ValueError(
                "Колонки файлов различаются — объединение отменено."
            )

        cleaned_dfs = []
        for file_path in self.files:
            logger.info(f"Обработка файла: {file_path.name}...")
            cleaned_df = self.process_single_file(file_path)
            cleaned_dfs.append(cleaned_df)

        logger.info("Объединение файлов и удаление дубликатов...")
        final_df = pd.concat(cleaned_dfs, ignore_index=True)
        final_df = final_df.drop_duplicates()

        logger.info("Заполнение пропусков общей медианой...")
        final_df = self.fill_na_median(final_df)

        # Очистка мусорных символов в целевой переменной
        final_df["Label"] = (
            final_df["Label"]
            .astype(str)
            .str.replace("\ufffd", "-", regex=False)
        )

        logger.info(f"Сохранение очищенного датасета в Parquet: {output_path}")
        self.save_cleaned(final_df, output_path)

        return final_df

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаляет пробелы из названий колонок и дропает ID-колонки."""
        df.columns = df.columns.str.strip()
        cols_to_drop = [
            c for c in self.ID_COLS + self.drop_cols if c in df.columns
        ]
        df = df.drop(columns=cols_to_drop)
        return df

    def clean_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает inf/Infinity значения и клипает нефизичные минусы."""
        df = self._fix_string_infinity(df)

        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        df = self._clip_negative_metrics(df)
        return df

    def _clip_negative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Векторизованно заменяет отрицательные значения в ключевых метриках на 0."""
        for col in self.NON_NEGATIVE_METRICS:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        return df

    def fill_na_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Заполняет NaN в числовых колонках медианой по всему датасету."""
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        return df

    def save_cleaned(self, df: pd.DataFrame, path: str | Path) -> None:
        """Сохраняет DataFrame в формате parquet, создавая папки при необходимости."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _fix_string_infinity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Конвертирует строковые артефакты 'Infinity' в float-совместимый формат."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            mask = df[col].astype(str).str.match(
                r"^\s*[+-]?inf(inity)?\s*$", case=False, na=False
            )
            if not mask.any():
                continue

            pos_mask = mask & ~df[col].astype(str).str.match(r"^\s*-", na=False)
            neg_mask = mask & df[col].astype(str).str.match(r"^\s*-", na=False)

            if pos_mask.any():
                df.loc[pos_mask, col] = np.inf
            if neg_mask.any():
                df.loc[neg_mask, col] = -np.inf

            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


def get_project_root(
    marker_files=[".git", "pyproject.toml", "README.md"],
) -> Path:
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise FileNotFoundError("Корень проекта не найден.")


def main():
    base_dir = get_project_root()
    
    with open(base_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_file = (
        base_dir / config["outputs"]["rel_path"] / config["outputs"]["log_file"]
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )

    logger.info("Запуск этапа очистки данных (Data Cleaning Stage)")

    raw_dir = base_dir / "data" / "raw"
    output_file = base_dir / config["data"]["rel_path"]

    cleaner = DataCleaner(raw_data_dir=raw_dir)
    cleaner.run_stage_one(output_path=output_file)

    logger.info("Очистка данных успешно завершена!")


if __name__ == "__main__":
    main()