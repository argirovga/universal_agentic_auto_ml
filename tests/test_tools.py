"""
Тесты для инструментов агентов.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.tools.validation import ValidationError, validate_dataframe, validate_predictions


class TestValidation:
    """Тесты модуля валидации."""

    def test_validate_dataframe_ok(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validate_dataframe(df) is True

    def test_validate_dataframe_empty(self):
        with pytest.raises(ValidationError):
            validate_dataframe(pd.DataFrame())

    def test_validate_dataframe_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValidationError):
            validate_dataframe(df, expected_columns=["a", "b"])

    def test_validate_predictions_ok(self):
        preds = np.array([100.0, 200.0, 50.0])
        result = validate_predictions(preds, expected_length=3)
        assert len(result) == 3

    def test_validate_predictions_wrong_length(self):
        preds = np.array([100.0, 200.0])
        with pytest.raises(ValidationError):
            validate_predictions(preds, expected_length=3)

    def test_validate_predictions_clips_range(self):
        preds = np.array([-10.0, 500.0, 100.0])
        result = validate_predictions(preds, expected_length=3)
        assert result[0] == 0.0
        assert result[1] == 365.0
        assert result[2] == 100.0

    def test_validate_predictions_handles_nan(self):
        preds = np.array([100.0, np.nan, 200.0])
        result = validate_predictions(preds, expected_length=3)
        assert not np.any(np.isnan(result))


class TestDataTools:
    """Тесты для инструментов работы с данными."""

    def test_load_data(self):
        from src.tools.data_tools import load_data
        result = load_data.invoke({"file_type": "train"})
        assert "train" in result.lower() or "Загружен" in result
        assert "target" in result

    def test_get_data_profile(self):
        from src.tools.data_tools import get_data_profile
        result = get_data_profile.invoke({"file_type": "train"})
        assert "Профиль" in result or "профиль" in result

    def test_get_correlations(self):
        from src.tools.data_tools import get_correlations
        result = get_correlations.invoke({})
        assert "Корреляции" in result or "корреляц" in result


class TestMLTools:
    """Тесты для инструментов ML."""

    def test_train_ridge(self):
        from src.tools.ml_tools import train_model
        result = train_model.invoke({"model_type": "ridge", "hyperparams": "{}"})
        assert "RMSE" in result
        assert "MAE" in result
        assert "R²" in result

    def test_train_invalid_model(self):
        from src.tools.ml_tools import train_model
        with pytest.raises(Exception):
            train_model.invoke({"model_type": "nonexistent", "hyperparams": "{}"})


class TestExperimentStore:
    """Тесты для хранилища экспериментов."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        from src.memory import experiment_store
        monkeypatch.setattr(experiment_store, "EXPERIMENTS_FILE", tmp_path / "test_exp.json")

        experiment_store.save_experiment(
            iteration=1,
            model_type="ridge",
            hyperparams={"alpha": 1.0},
            metrics={"rmse": 100.0, "mae": 80.0, "r2": 0.3},
        )

        history = experiment_store.load_history()
        assert len(history) == 1
        assert history[0]["model_type"] == "ridge"
        assert history[0]["metrics"]["rmse"] == 100.0

    def test_get_best(self, tmp_path, monkeypatch):
        from src.memory import experiment_store
        monkeypatch.setattr(experiment_store, "EXPERIMENTS_FILE", tmp_path / "test_exp.json")

        experiment_store.save_experiment(1, "ridge", {}, {"rmse": 100.0, "mae": 80.0, "r2": 0.3})
        experiment_store.save_experiment(2, "lgbm", {}, {"rmse": 80.0, "mae": 60.0, "r2": 0.5})

        best = experiment_store.get_best_experiment()
        assert best["model_type"] == "lgbm"
        assert best["metrics"]["rmse"] == 80.0
