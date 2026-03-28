# Тюнинг гиперпараметров

## Общие рекомендации
1. Сначала установите n_estimators достаточно большим (500-1000) и подберите learning_rate.
2. Затем настройте параметры дерева (max_depth, min_samples_leaf).
3. Добавьте регуляризацию если нужно.

## LightGBM — типичные диапазоны
- `n_estimators`: 300-2000
- `learning_rate`: 0.01-0.1
- `max_depth`: 4-10 (или -1 для авто)
- `num_leaves`: 20-100
- `min_child_samples`: 10-50
- `subsample`: 0.6-0.9
- `colsample_bytree`: 0.6-0.9
- `reg_alpha`: 0.0-1.0 (L1)
- `reg_lambda`: 0.0-1.0 (L2)

## XGBoost — типичные диапазоны
- `n_estimators`: 300-2000
- `learning_rate`: 0.01-0.1
- `max_depth`: 3-8
- `min_child_weight`: 1-10
- `subsample`: 0.6-0.9
- `colsample_bytree`: 0.6-0.9
- `gamma`: 0.0-1.0
- `reg_alpha`: 0.0-1.0
- `reg_lambda`: 0.0-1.0

## RandomForest — типичные диапазоны
- `n_estimators`: 100-500
- `max_depth`: 10-30 (или None)
- `min_samples_leaf`: 1-10
- `min_samples_split`: 2-10
- `max_features`: "sqrt", "log2", 0.3-0.8

## Стратегии поиска
- **Grid Search** — полный перебор, медленный.
- **Random Search** — эффективнее, рекомендуется 50-100 итераций.
- **Bayesian** (Optuna) — самый эффективный, но сложнее в настройке.
