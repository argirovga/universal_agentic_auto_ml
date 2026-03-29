"""
Инструменты машинного обучения — обучение моделей, предсказания, метрики.
Используются агентами через LangGraph tool calling.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from config.settings import settings
from src.tools.validation import log_tool_call, validate_predictions

logger = logging.getLogger(__name__)

# Реестр поддерживаемых моделей
MODEL_REGISTRY = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "lightgbm": None,  # Импортируется лениво
    "xgboost": None,    # Импортируется лениво
}


def _get_model_class(model_type: str):
    """Возвращает класс модели по имени."""
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor
    elif model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type]
    else:
        raise ValueError(f"Неизвестная модель: {model_type}. Доступные: {list(MODEL_REGISTRY.keys())}")


# Кэш энкодеров, заполняется при обработке train, применяется к test
_fitted_encoders: dict[str, LabelEncoder] = {}
_train_medians: dict[str, float] = {}


def _prepare_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Подготовка признаков — кодирование категорий, обработка дат, удаление текстовых.

    При is_train=True — фитит энкодеры и запоминает медианы.
    При is_train=False — применяет ранее зафитенные энкодеры и медианы.

    Args:
        df: Исходный DataFrame.
        is_train: Флаг, является ли это обучающей выборкой.

    Returns:
        DataFrame с подготовленными признаками.
    """
    global _fitted_encoders, _train_medians
    df = df.copy()

    # Удаляем текстовые колонки, которые не несут ML-информации
    drop_cols = ["name", "host_name", "_id"]
    drop_cols_existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols_existing, errors="ignore")

    # Обработка даты last_dt
    if "last_dt" in df.columns:
        df["last_dt"] = pd.to_datetime(df["last_dt"], errors="coerce")
        reference_date = pd.Timestamp("2019-07-08")  # Последняя дата в датасете + 1
        df["days_since_last_review"] = (reference_date - df["last_dt"]).dt.days
        df["last_review_month"] = df["last_dt"].dt.month
        df = df.drop(columns=["last_dt"])

    # Кодирование категориальных переменных
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if is_train:
        _fitted_encoders.clear()
        for col in cat_cols:
            df[col] = df[col].fillna("unknown").astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            _fitted_encoders[col] = le
    else:
        for col in cat_cols:
            df[col] = df[col].fillna("unknown").astype(str)
            if col in _fitted_encoders:
                le = _fitted_encoders[col]
                # Неизвестные категории заменяем на -1
                known = set(le.classes_)
                df[col] = df[col].map(
                    lambda x, _known=known, _le=le: (
                        _le.transform([x])[0] if x in _known else -1
                    )
                )
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

    # Заполнение пропусков в числовых колонках медианой
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if is_train:
        _train_medians.clear()
        for col in num_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                _train_medians[col] = median_val
                df[col] = df[col].fillna(median_val)
    else:
        for col in num_cols:
            if df[col].isna().any():
                median_val = _train_medians.get(col, df[col].median())
                df[col] = df[col].fillna(median_val)

    return df


@tool
def train_model(model_type: str = "lightgbm", hyperparams: str = "{}") -> str:
    """
    Обучает модель регрессии и возвращает метрики на валидации.

    Args:
        model_type: Тип модели — ridge, random_forest, gradient_boosting, lightgbm, xgboost.
        hyperparams: JSON-строка с гиперпараметрами модели.

    Returns:
        Строка с результатами: RMSE, MAE, R² на валидации.
    """
    # Парсим гиперпараметры
    try:
        params = json.loads(hyperparams) if hyperparams else {}
    except json.JSONDecodeError:
        params = {}
        logger.warning("Не удалось распарсить гиперпараметры, используются значения по умолчанию.")

    # Загрузка и подготовка данных
    df = pd.read_csv(settings.train_file)
    target = df[settings.target_column]
    df_features = _prepare_features(df, is_train=True)

    # Удаляем целевую переменную из признаков
    if settings.target_column in df_features.columns:
        df_features = df_features.drop(columns=[settings.target_column])

    # Разделение на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        df_features, target,
        test_size=settings.validation_split,
        random_state=settings.random_seed,
    )

    # Создание и обучение модели
    model_class = _get_model_class(model_type)

    # Дефолтные параметры для стабильности
    default_params = {"random_state": settings.random_seed}
    if model_type in ("lightgbm", "xgboost"):
        default_params["verbosity"] = -1 if model_type == "lightgbm" else 0
        default_params["n_estimators"] = params.pop("n_estimators", 500)
        default_params["learning_rate"] = params.pop("learning_rate", 0.05)
        # Один поток — предотвращает segfault при fork() на macOS
        default_params["n_jobs"] = 1
    elif model_type == "random_forest":
        default_params["n_estimators"] = params.pop("n_estimators", 200)
        default_params["n_jobs"] = 1
    elif model_type == "gradient_boosting":
        default_params["n_estimators"] = params.pop("n_estimators", 300)
        default_params["learning_rate"] = params.pop("learning_rate", 0.05)

    default_params.update(params)
    model = model_class(**default_params)

    logger.info("Обучение модели %s с параметрами: %s", model_type, default_params)
    model.fit(X_train, y_train)

    # Метрики на валидации
    y_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))

    # Feature importances (если доступно)
    feature_importance_str = ""
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        top_features = importances.sort_values(ascending=False).head(10)
        feature_importance_str = f"\nТоп-10 важных признаков:\n{top_features.to_string()}"

    result = (
        f"=== Результаты обучения: {model_type} ===\n"
        f"Параметры: {json.dumps(default_params, default=str)}\n"
        f"Train размер: {len(X_train)}, Validation размер: {len(X_val)}\n"
        f"Признаков: {len(X_train.columns)}\n"
        f"\nМетрики на валидации:\n"
        f"  RMSE: {rmse:.4f}\n"
        f"  MAE:  {mae:.4f}\n"
        f"  R²:   {r2:.4f}"
        f"{feature_importance_str}"
    )

    logger.info("Модель %s — RMSE: %.4f, MAE: %.4f, R²: %.4f", model_type, rmse, mae, r2)
    return result


@tool
def predict_and_submit(model_type: str = "lightgbm", hyperparams: str = "{}") -> str:
    """
    Обучает модель на всех данных, предсказывает на тесте, сохраняет submission.

    Args:
        model_type: Тип модели.
        hyperparams: JSON-строка с гиперпараметрами.

    Returns:
        Путь к файлу submission и статистика предсказаний.
    """
    try:
        params = json.loads(hyperparams) if hyperparams else {}
    except json.JSONDecodeError:
        params = {}

    # Загрузка данных
    train_df = pd.read_csv(settings.train_file)
    test_df = pd.read_csv(settings.test_file)

    target = train_df[settings.target_column]

    # Подготовка признаков
    train_features = _prepare_features(train_df, is_train=True)
    if settings.target_column in train_features.columns:
        train_features = train_features.drop(columns=[settings.target_column])

    test_features = _prepare_features(test_df, is_train=False)

    # Выравниваем колонки — test должен иметь те же фичи, что и train
    missing_in_test = set(train_features.columns) - set(test_features.columns)
    for col in missing_in_test:
        test_features[col] = 0
    extra_in_test = set(test_features.columns) - set(train_features.columns)
    test_features = test_features.drop(columns=list(extra_in_test), errors="ignore")
    test_features = test_features[train_features.columns]

    # Создание и обучение модели на всех данных
    model_class = _get_model_class(model_type)
    default_params = {"random_state": settings.random_seed}
    if model_type in ("lightgbm", "xgboost"):
        default_params["verbosity"] = -1 if model_type == "lightgbm" else 0
        default_params["n_estimators"] = params.pop("n_estimators", 500)
        default_params["learning_rate"] = params.pop("learning_rate", 0.05)
        default_params["n_jobs"] = 1
    elif model_type == "random_forest":
        default_params["n_estimators"] = params.pop("n_estimators", 200)
        default_params["n_jobs"] = 1
    elif model_type == "gradient_boosting":
        default_params["n_estimators"] = params.pop("n_estimators", 300)
        default_params["learning_rate"] = params.pop("learning_rate", 0.05)

    default_params.update(params)
    model = model_class(**default_params)
    model.fit(train_features, target)

    # Предсказание
    predictions = model.predict(test_features)
    predictions = validate_predictions(predictions, len(test_df))

    # Сохранение submission
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = settings.output_dir / "submission.csv"
    submission = pd.DataFrame({
        "index": range(len(predictions)),
        "prediction": predictions,
    })
    submission.to_csv(submission_path, index=False)

    result = (
        f"=== Submission создан ===\n"
        f"Файл: {submission_path}\n"
        f"Количество предсказаний: {len(predictions)}\n"
        f"Среднее предсказание: {predictions.mean():.2f}\n"
        f"Мин: {predictions.min():.2f}, Макс: {predictions.max():.2f}\n"
        f"Стд. откл.: {predictions.std():.2f}"
    )

    logger.info("Submission сохранён: %s (%d строк)", submission_path, len(predictions))
    return result
