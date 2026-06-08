import logging
import yaml
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==================== CONFIGURATION ====================
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config("config.yaml")
log_file_path = Path(CONFIG["outputs"]["rel_path"]) / CONFIG["outputs"]["log_file"]
# ==================== LOGGER SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== HELPERS ====================
def get_project_root(marker_files) -> Path:
    """Ищет корень проекта, поднимаясь вверх от текущего файла."""
    current_file = Path(__file__).resolve()
    for parent in [current_file.parent, *current_file.parents]:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise FileNotFoundError(f"Корень проекта не найден. Маркеры: {marker_files}")


def compute_class_weights(y: np.ndarray, clip: float = None) -> np.ndarray:
    """Sqrt-balanced class weights with optional clipping."""
    counts = np.bincount(y)
    w = np.sqrt(len(y) / (len(counts) * counts))
    return np.clip(w[y], None, clip)


def plot_feature_importance(model, feature_cols, title, save_path, top_n=20):
    """Save top-N feature importance plot."""
    plt.figure(figsize=(10, 8))
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(top_n)
    sns.barplot(x=importance.values, y=importance.index, palette='viridis', hue=importance.index, legend=False)
    plt.title(f'{title} (Top-{top_n})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path, cmap='Blues'):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(max(6, len(labels)), max(5, len(labels) * 0.6)))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if len(labels) > 3:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_and_save(model, X_test, y_test, le, feature_cols, stage_name, output_dir, cmap='Blues', digits=4):
    """Единая функция: предсказание → метрики → графики → CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    class_names = le.classes_

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, average=None)
    metrics_df = pd.DataFrame({
        'Class': class_names, 'Precision': p, 'Recall': r, 'F1-Score': f1, 'Support': s.astype(int)
    })

    logger.info(f"=== {stage_name.upper()} RESULTS ===")
    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"Macro F1:    {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=class_names, digits=digits)}")
    
    csv_path = output_dir / f'{stage_name}_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Metrics saved: {csv_path}")

    plot_confusion_matrix(
        y_test, y_pred, class_names,
        f'{stage_name.upper()} Confusion Matrix',
        output_dir / f'{stage_name}_confusion_matrix.png', cmap=cmap
    )
    plot_feature_importance(
        model, feature_cols,
        f'{stage_name.upper()} Feature Importance',
        output_dir / f'{stage_name}_feature_importance.png'
    )
    return metrics_df



def main():
    logger.info("Starting Pipeline...")
    
    # Определение путей
    base_dir = get_project_root(CONFIG['project']["marker_files"])
    data_path = base_dir / CONFIG["data"]["rel_path"]
    output_dir = base_dir / CONFIG["outputs"]["rel_path"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── [1/5] LOAD DATA ──────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    df = pd.read_parquet(data_path)
    df['Stage1_Label'] = df['Label'].map(CONFIG["mappings"]["label1_map"]).fillna('Other_Attack')
    feature_cols = [c for c in df.columns if c not in {"Label", "Stage1_Label"}]
    # ТОЛЬКО ДЛЯ ИНФЕРЕНСА В predict.py.
    numeric_cols = df.select_dtypes(include="number").columns
    medians = {col: df[col].median() for col in numeric_cols}
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # ─── [2/5] TRAIN / VAL / TEST SPLIT ──────────────────────
    logger.info("Splitting data into Train, Validation, and Test sets...")
    
    df_train, df_temp = train_test_split(
        df, test_size=CONFIG["data"]["test_size"], 
        random_state=CONFIG["project"]["random_state"], stratify=df["Label"], shuffle=True
    )
    
    df_test, df_val = train_test_split(
        df_temp, test_size=CONFIG["data"]["ratio_val_test"], 
        random_state=CONFIG["project"]["random_state"], stratify=df_temp["Label"], shuffle=True
    )

    # Подготовка данных для Stage 1
    le1 = LabelEncoder()
    X1_train = df_train[feature_cols].astype(np.float32)
    y1_train = le1.fit_transform(df_train["Stage1_Label"])
    
    X1_test = df_test[feature_cols].astype(np.float32)
    y1_test = le1.transform(df_test["Stage1_Label"])
    
    X1_val = df_val[feature_cols].astype(np.float32)
    y1_val = le1.transform(df_val["Stage1_Label"])

    # Подготовка данных для Stage 2 (строго фильтруем из уже разделенных сетов!)
    label_stage2 = CONFIG["mappings"]["label_stage2"]
    df_train_s2 = df_train[df_train["Label"].isin(label_stage2)]
    df_val_s2 = df_val[df_val["Label"].isin(label_stage2)]
    df_test_s2 = df_test[df_test["Label"].isin(label_stage2)]

    le2 = LabelEncoder()
    X2_train = df_train_s2[feature_cols].astype(np.float32)
    y2_train = le2.fit_transform(df_train_s2["Label"])
    
    X2_test = df_test_s2[feature_cols].astype(np.float32)
    y2_test = le2.transform(df_test_s2["Label"])
    
    X2_val = df_val_s2[feature_cols].astype(np.float32)
    y2_val = le2.transform(df_val_s2["Label"])


    # ─── [3/5] TRAINING STAGE 1 ───────────────────────────────────────────────
    logger.info("Training Stage 1 Model...")
    p1 = CONFIG["models"]["xgb_stage1"]
    weights1 = compute_class_weights(y1_train, clip=CONFIG["preprocessing"]["stage1_clip_weight"])

    model1 = xgb.XGBClassifier(
        objective=p1["objective"], num_class=p1["num_class"], eval_metric=p1["eval_metric"],
        n_estimators=p1["n_estimators"], max_depth=p1["max_depth"], learning_rate=p1["learning_rate"],
        subsample=p1["subsample"], colsample_bytree=p1["colsample_bytree"], 
        min_child_weight=p1["min_child_weight"], gamma=p1["gamma"],
        random_state=CONFIG["project"]["random_state"], n_jobs=-1, 
        early_stopping_rounds=p1["early_stopping_rounds"]
    )
    
    model1.fit(X1_train, y1_train, sample_weight=weights1, eval_set=[(X1_val, y1_val)], verbose=False)

    evaluate_and_save(
        model=model1, X_test=X1_test, y_test=y1_test, le=le1, feature_cols=feature_cols,
        stage_name='stage1', output_dir=output_dir, cmap='Blues'
    )

    # ─── [4/5] TRAINING STAGE 2 ───────────────────────────────────────────────
    logger.info("Training Stage 2 Model...")
    p2 = CONFIG["models"]["xgb_stage2"]
    weights2 = compute_class_weights(y2_train, clip=CONFIG["preprocessing"]["stage2_clip_weight"])

    model2 = xgb.XGBClassifier(
        objective=p2["objective"], num_class=len(le2.classes_), eval_metric=p2["eval_metric"],
        n_estimators=p2["n_estimators"], max_depth=p2["max_depth"], learning_rate=p2["learning_rate"],
        subsample=p2["subsample"], colsample_bytree=p2["colsample_bytree"], colsample_bylevel=p2["colsample_bylevel"],
        min_child_weight=p2["min_child_weight"], gamma=p2["gamma"], reg_alpha=p2["reg_alpha"], reg_lambda=p2["reg_lambda"],
        random_state=CONFIG["project"]["random_state"], n_jobs=-1, 
        early_stopping_rounds=p2["early_stopping_rounds"]
    )
    
    # Обучаем, валидируемся на X2_val
    model2.fit(X2_train, y2_train, sample_weight=weights2, eval_set=[(X2_val, y2_val)], verbose=False)

    # Финальная оценка на чистом X2_test
    evaluate_and_save(
        model=model2, X_test=X2_test, y_test=y2_test, le=le2, feature_cols=feature_cols,
        stage_name='stage2', output_dir=output_dir, cmap='Oranges'
    )

    # ─── [5/5] ARTIFACT SAVING (XGB native format) ───────────────────────────
    logger.info("Saving models and encoders...")
    
    model1.save_model(output_dir / 'stage1_xgb_model.json')
    model2.save_model(output_dir / 'stage2_xgb_model.json')
    
    joblib.dump(le1, output_dir / 'stage1_label_encoder.pkl')
    joblib.dump(le2, output_dir / 'stage2_label_encoder.pkl')
    joblib.dump(feature_cols, output_dir / 'feature_columns.pkl')
    joblib.dump(medians, output_dir / 'median_columns.pkl')

    logger.info("Pipeline complete successfully!")


if __name__ == "__main__":
    main()