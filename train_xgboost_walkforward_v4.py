from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

import re
from augment import jitter, window_slice
from inference import NativeCatBoostClassifier

AUG_RNG = np.random.default_rng(42)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / 'data_0603/processed/flight_model_dataset_final_v3.csv'

data_folder_name = DATA_FILE.parts[-3]  # "data_0421"
match = re.search(r'data_(\d+)', data_folder_name)
date_suffix = match.group(1) if match else 'unknown'

OUTPUT_DIR = PROJECT_ROOT / f'xgb_walkforward_outputs_v2_{date_suffix}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = 'searched_at'
TRAJECTORY_COL = 'trajectory_id'
CLASS_TARGET = 'target_wait'
REG_TARGET = 'target_log_ratio'

MIN_OBS_PER_TRAJECTORY = 2
N_SPLITS = 4
TEST_FRACTION = 0.2
MIN_TRAIN_ROWS = 12
CLASS_BASELINE_PROB = 0.3
RANDOM_STATE = 42
SKIP_ZERO_ONLY_REG_FOLDS = True
SKIP_SINGLE_CLASS_CLS_FOLDS = True

# 미래 관측값이 충분히 쌓인 샘플만 학습에 사용
# future_obs_count < MIN_FUTURE_OBS 인 샘플은 타겟이 right-censored일 수 있음
MIN_FUTURE_OBS = 30   # 0으로 설정하면 필터 비활성화

EXCLUDE_COLS = {
    TIME_COL,
    TRAJECTORY_COL,
    'flight_search_id',
    'future_obs_count',   # 메타 컬럼, 피처에서 제외
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if MIN_FUTURE_OBS > 0 and 'future_obs_count' in df.columns:
        before = len(df)
        df = df[df['future_obs_count'] >= MIN_FUTURE_OBS].reset_index(drop=True)
        print(f'[filter] future_obs_count >= {MIN_FUTURE_OBS}: {before} → {len(df)} rows '
              f'({before - len(df)} 제거)')

    return df


def filter_min_obs_per_trajectory(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    tmp = df.sort_values([TRAJECTORY_COL, TIME_COL]).copy()
    tmp['obs_rank_in_trajectory'] = tmp.groupby(TRAJECTORY_COL).cumcount() + 1
    out = tmp[tmp['obs_rank_in_trajectory'] >= min_obs].copy()
    return out.drop(columns=['obs_rank_in_trajectory'])


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if c.startswith('target_'):
            continue
        cols.append(c)
    return cols


def split_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    X = df[feature_cols].copy()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
    return numeric_cols, categorical_cols


CAT_COLS_NATIVE = ['route_id', 'searched_day_of_week', 'outbound_day_of_week']


def make_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_feature_types(df, feature_cols)
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop',
    )


def make_classification_pipeline(df: pd.DataFrame, feature_cols: List[str], scale_pos_weight: float):
    """CatBoost 네이티브 범주형 분류기 반환 (sklearn Pipeline 아님)."""
    return NativeCatBoostClassifier(
        cat_cols=[c for c in CAT_COLS_NATIVE if c in feature_cols],
        iterations=120,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )


def make_regression_pipeline(df: pd.DataFrame, feature_cols: List[str]) -> Pipeline:
    preprocessor = make_preprocessor(df, feature_cols)
    reg = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=2,
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([('preprocessor', preprocessor), ('model', reg)])


def get_feature_names(fitted_pipeline: Pipeline, feature_cols: List[str], df: pd.DataFrame) -> List[str]:
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    numeric_cols, categorical_cols = split_feature_types(df, feature_cols)
    names: List[str] = []
    names.extend(numeric_cols)
    if categorical_cols:
        onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
        names.extend(onehot.get_feature_names_out(categorical_cols).tolist())
    return names


def make_walkforward_splits(df: pd.DataFrame, n_splits: int, test_fraction: float, min_train_rows: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    test_size = max(5, int(np.floor(n * test_fraction)))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    max_possible = max(1, (n - min_train_rows) // test_size)
    n_actual = min(n_splits, max_possible)

    for i in range(n_actual):
        train_end = min_train_rows + i * test_size
        test_end = min(train_end + test_size, n)
        if test_end <= train_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return None


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def main() -> None:
    df = load_dataset(DATA_FILE)
    df = filter_min_obs_per_trajectory(df, MIN_OBS_PER_TRAJECTORY)
    feature_cols = infer_feature_columns(df)
    splits = make_walkforward_splits(df, N_SPLITS, TEST_FRACTION, MIN_TRAIN_ROWS)

    fold_metrics_cls: List[dict[str, Any]] = []
    fold_metrics_reg: List[dict[str, Any]] = []
    fold_rows_cls: List[pd.DataFrame] = []
    fold_rows_reg: List[pd.DataFrame] = []
    clf_importances: List[pd.Series] = []
    reg_importances: List[pd.Series] = []

    last_clf_pipe = None
    last_reg_pipe = None

    # best classification / regression model 추적
    best_clf_pipe = None
    best_clf_f1 = -1.0
    best_clf_fold = None
    best_reg_pipe = None
    best_reg_mae = float('inf')
    best_reg_fold = None

    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Classification
        cls_train = train_df[train_df[CLASS_TARGET].notna()].copy()
        cls_test = test_df[test_df[CLASS_TARGET].notna()].copy()
        if len(cls_train) > 0 and len(cls_test) > 0:
            X_train_cls = cls_train[feature_cols]
            y_train_cls = cls_train[CLASS_TARGET].astype(int).to_numpy()
            X_test_cls = cls_test[feature_cols]
            y_test_cls = cls_test[CLASS_TARGET].astype(int).to_numpy()

            unique_train = np.unique(y_train_cls)
            if SKIP_SINGLE_CLASS_CLS_FOLDS and len(unique_train) < 2:
                print(f'[INFO] Skipping classification fold {fold_no}: train has single class {unique_train.tolist()}')
            else:
                # 분류: jitter σ=0.02 증강 (AUC +5.9% 실험 결과)
                cls_train_aug = jitter(cls_train, sigma=0.02, n_copies=3, rng=AUG_RNG)
                X_train_cls   = cls_train_aug[feature_cols]
                y_train_cls   = cls_train_aug[CLASS_TARGET].astype(int).to_numpy()

                pos = int((y_train_cls == 1).sum())
                neg = int((y_train_cls == 0).sum())
                scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
                clf_pipe = make_classification_pipeline(cls_train, feature_cols, scale_pos_weight)
                clf_pipe.fit(cls_train_aug[feature_cols], y_train_cls)
                last_clf_pipe = clf_pipe

                y_prob = clf_pipe.predict_proba(cls_test[feature_cols])[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                base_prob = np.full(len(y_test_cls), CLASS_BASELINE_PROB)
                base_pred = (base_prob >= 0.5).astype(int)

                fold_f1 = float(f1_score(y_test_cls, y_pred, zero_division=0))
                if fold_f1 > best_clf_f1:
                    best_clf_f1 = fold_f1
                    best_clf_pipe = clf_pipe
                    best_clf_fold = fold_no

                fold_metrics_cls.append({
                    'fold': fold_no,
                    'train_rows': int(len(cls_train)),
                    'test_rows': int(len(cls_test)),
                    'train_positive_ratio': float(np.mean(y_train_cls)),
                    'test_positive_ratio': float(np.mean(y_test_cls)),
                    'model_accuracy': float(accuracy_score(y_test_cls, y_pred)),
                    'model_precision': float(precision_score(y_test_cls, y_pred, zero_division=0)),
                    'model_recall': float(recall_score(y_test_cls, y_pred, zero_division=0)),
                    'model_f1': float(f1_score(y_test_cls, y_pred, zero_division=0)),
                    'model_roc_auc': safe_roc_auc(y_test_cls, y_prob),
                    'model_log_loss': safe_log_loss(y_test_cls, y_prob),
                    'model_brier': brier_score(y_test_cls, y_prob),
                    'model_confusion_matrix': confusion_matrix(y_test_cls, y_pred, labels=[0, 1]).tolist(),
                    'baseline_accuracy': float(accuracy_score(y_test_cls, base_pred)),
                    'baseline_precision': float(precision_score(y_test_cls, base_pred, zero_division=0)),
                    'baseline_recall': float(recall_score(y_test_cls, base_pred, zero_division=0)),
                    'baseline_f1': float(f1_score(y_test_cls, base_pred, zero_division=0)),
                    'baseline_roc_auc': safe_roc_auc(y_test_cls, base_prob),
                    'baseline_log_loss': safe_log_loss(y_test_cls, base_prob),
                    'baseline_brier': brier_score(y_test_cls, base_prob),
                    'baseline_confusion_matrix': confusion_matrix(y_test_cls, base_pred, labels=[0, 1]).tolist(),
                })

                pred_rows = cls_test[[TIME_COL]].copy()
                pred_rows['fold'] = fold_no
                pred_rows['y_true'] = y_test_cls
                pred_rows['model_prob'] = y_prob
                pred_rows['model_pred'] = y_pred
                pred_rows['baseline_prob'] = base_prob
                pred_rows['baseline_pred'] = base_pred
                fold_rows_cls.append(pred_rows)

                clf_importances.append(
                    pd.Series(
                        clf_pipe.feature_importances_,
                        index=clf_pipe.feat_cols_,
                        name=f'fold_{fold_no}',
                    )
                )

        # Regression
        reg_train = train_df[train_df[REG_TARGET].notna()].copy()
        reg_test = test_df[test_df[REG_TARGET].notna()].copy()
        if len(reg_train) > 0 and len(reg_test) > 0:
            train_zero_ratio = float((reg_train[REG_TARGET] == 0).mean())
            test_zero_ratio = float((reg_test[REG_TARGET] == 0).mean())
            print(f'[Fold {fold_no}] reg train zero ratio: {train_zero_ratio:.4f}, reg test zero ratio: {test_zero_ratio:.4f}')

            if SKIP_ZERO_ONLY_REG_FOLDS and train_zero_ratio == 1.0:
                print(f'[INFO] Skipping regression fold {fold_no}: all train targets are zero')
            else:
                # 회귀: window_slice 증강 (MAE -12.3% 실험 결과)
                reg_train_aug = window_slice(reg_train, n_slices=4, rng=AUG_RNG)
                X_train_reg   = reg_train_aug[feature_cols]
                y_train_reg   = reg_train_aug[REG_TARGET].astype(float).to_numpy()
                X_test_reg    = reg_test[feature_cols]
                y_test_reg    = reg_test[REG_TARGET].astype(float).to_numpy()

                reg_pipe = make_regression_pipeline(reg_train, feature_cols)
                reg_pipe.fit(X_train_reg, y_train_reg)
                last_reg_pipe = reg_pipe

                y_pred_reg = reg_pipe.predict(X_test_reg)
                base_pred_reg = np.zeros(len(y_test_reg), dtype=float)

                model_mae = float(mean_absolute_error(y_test_reg, y_pred_reg))
                model_rmse = float(root_mean_squared_error(y_test_reg, y_pred_reg))
                baseline_mae = float(mean_absolute_error(y_test_reg, base_pred_reg))
                baseline_rmse = float(root_mean_squared_error(y_test_reg, base_pred_reg))

                # 추가: best regression model 갱신
                if model_mae < best_reg_mae:
                    best_reg_mae = model_mae
                    best_reg_pipe = reg_pipe
                    best_reg_fold = fold_no

                fold_metrics_reg.append({
                    'fold': fold_no,
                    'train_rows': int(len(reg_train)),
                    'test_rows': int(len(reg_test)),
                    'train_zero_ratio': train_zero_ratio,
                    'test_zero_ratio': test_zero_ratio,
                    'model_mae': model_mae,
                    'model_rmse': model_rmse,
                    'baseline_predict_value': 0.0,
                    'baseline_mae': baseline_mae,
                    'baseline_rmse': baseline_rmse,
                })

                pred_rows = reg_test[[TIME_COL]].copy()
                pred_rows['fold'] = fold_no
                pred_rows['y_true'] = y_test_reg
                pred_rows['model_pred'] = y_pred_reg
                pred_rows['baseline_pred'] = base_pred_reg
                fold_rows_reg.append(pred_rows)

                feat_names = get_feature_names(reg_pipe, feature_cols, reg_train)
                reg_importances.append(
                    pd.Series(
                        reg_pipe.named_steps['model'].feature_importances_,
                        index=feat_names,
                        name=f'fold_{fold_no}',
                    )
                )

    cls_metrics_df = pd.DataFrame(fold_metrics_cls)
    reg_metrics_df = pd.DataFrame(fold_metrics_reg)
    cls_predictions_df = pd.concat(fold_rows_cls, ignore_index=True) if fold_rows_cls else pd.DataFrame()
    reg_predictions_df = pd.concat(fold_rows_reg, ignore_index=True) if fold_rows_reg else pd.DataFrame()

    if clf_importances:
        clf_imp_df = pd.concat(clf_importances, axis=1).fillna(0.0)
        clf_imp_df['mean_importance'] = clf_imp_df.mean(axis=1)
        clf_imp_df = clf_imp_df.sort_values('mean_importance', ascending=False).reset_index().rename(columns={'index': 'feature'})
    else:
        clf_imp_df = pd.DataFrame(columns=['feature', 'mean_importance'])

    if reg_importances:
        reg_imp_df = pd.concat(reg_importances, axis=1).fillna(0.0)
        reg_imp_df['mean_importance'] = reg_imp_df.mean(axis=1)
        reg_imp_df = reg_imp_df.sort_values('mean_importance', ascending=False).reset_index().rename(columns={'index': 'feature'})
    else:
        reg_imp_df = pd.DataFrame(columns=['feature', 'mean_importance'])

    def mean_or_none(series: pd.Series) -> float | None:
        vals = series.dropna()
        return None if len(vals) == 0 else float(vals.mean())

    classification_summary = {
        'n_folds_used': int(len(cls_metrics_df)),
        'mean_model_accuracy': mean_or_none(cls_metrics_df.get('model_accuracy', pd.Series(dtype=float))),
        'mean_model_precision': mean_or_none(cls_metrics_df.get('model_precision', pd.Series(dtype=float))),
        'mean_model_recall': mean_or_none(cls_metrics_df.get('model_recall', pd.Series(dtype=float))),
        'mean_model_f1': mean_or_none(cls_metrics_df.get('model_f1', pd.Series(dtype=float))),
        'mean_model_roc_auc': mean_or_none(cls_metrics_df.get('model_roc_auc', pd.Series(dtype=float))),
        'mean_model_log_loss': mean_or_none(cls_metrics_df.get('model_log_loss', pd.Series(dtype=float))),
        'mean_model_brier': mean_or_none(cls_metrics_df.get('model_brier', pd.Series(dtype=float))),
        'mean_baseline_accuracy': mean_or_none(cls_metrics_df.get('baseline_accuracy', pd.Series(dtype=float))),
        'mean_baseline_log_loss': mean_or_none(cls_metrics_df.get('baseline_log_loss', pd.Series(dtype=float))),
        'mean_baseline_brier': mean_or_none(cls_metrics_df.get('baseline_brier', pd.Series(dtype=float))),
        'best_clf_fold': best_clf_fold,
        'best_clf_f1': None if best_clf_f1 < 0 else best_clf_f1,
    }

    regression_summary = {
        'n_folds_used': int(len(reg_metrics_df)),
        'mean_model_mae': mean_or_none(reg_metrics_df.get('model_mae', pd.Series(dtype=float))),
        'mean_model_rmse': mean_or_none(reg_metrics_df.get('model_rmse', pd.Series(dtype=float))),
        'mean_baseline_mae': mean_or_none(reg_metrics_df.get('baseline_mae', pd.Series(dtype=float))),
        'mean_baseline_rmse': mean_or_none(reg_metrics_df.get('baseline_rmse', pd.Series(dtype=float))),
        'best_reg_fold': best_reg_fold,
        'best_reg_mae': None if best_reg_mae == float('inf') else best_reg_mae,
    }

    cls_metrics_df.to_csv(OUTPUT_DIR / 'classification_fold_metrics.csv', index=False, encoding='utf-8-sig')
    reg_metrics_df.to_csv(OUTPUT_DIR / 'regression_fold_metrics.csv', index=False, encoding='utf-8-sig')
    cls_predictions_df.to_csv(OUTPUT_DIR / 'classification_predictions.csv', index=False, encoding='utf-8-sig')
    reg_predictions_df.to_csv(OUTPUT_DIR / 'regression_predictions.csv', index=False, encoding='utf-8-sig')
    clf_imp_df.to_csv(OUTPUT_DIR / 'classification_feature_importance.csv', index=False, encoding='utf-8-sig')
    reg_imp_df.to_csv(OUTPUT_DIR / 'regression_feature_importance.csv', index=False, encoding='utf-8-sig')

    with open(OUTPUT_DIR / 'classification_summary.json', 'w', encoding='utf-8') as f:
        json.dump(classification_summary, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'regression_summary.json', 'w', encoding='utf-8') as f:
        json.dump(regression_summary, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump({
            'data_file': str(DATA_FILE),
            'n_total_rows_after_filter': int(len(df)),
            'n_splits_requested': N_SPLITS,
            'n_splits_created': int(len(splits)),
            'feature_columns': feature_cols,
            'skip_zero_only_reg_folds': SKIP_ZERO_ONLY_REG_FOLDS,
            'skip_single_class_cls_folds': SKIP_SINGLE_CLASS_CLS_FOLDS,
            'best_reg_fold': best_reg_fold,
            'best_reg_mae': None if best_reg_mae == float('inf') else best_reg_mae,
        }, f, ensure_ascii=False, indent=2)

    if last_clf_pipe is not None:
        joblib.dump(last_clf_pipe, OUTPUT_DIR / 'catboost_classifier_last_fold.joblib')
    if best_clf_pipe is not None:
        joblib.dump(best_clf_pipe, OUTPUT_DIR / f'catboost_classifier_best_{date_suffix}.joblib')
    if last_reg_pipe is not None:
        joblib.dump(last_reg_pipe, OUTPUT_DIR / 'xgb_regressor_last_fold.joblib')
    if best_reg_pipe is not None:
        joblib.dump(best_reg_pipe, OUTPUT_DIR / f'xgb_regressor_best_{date_suffix}.joblib')

    print(f'Saved outputs to: {OUTPUT_DIR}')
    print('\n=== Classification Walk-Forward Summary ===')
    print(json.dumps(classification_summary, ensure_ascii=False, indent=2))
    if not cls_metrics_df.empty:
        print('\n=== Classification Fold Metrics ===')
        print(cls_metrics_df.to_string(index=False))
        print('\n=== Top 15 Classification Feature Importances (mean) ===')
        print(clf_imp_df[['feature', 'mean_importance']].head(15).to_string(index=False))

    print('\n=== Regression Walk-Forward Summary ===')
    print(json.dumps(regression_summary, ensure_ascii=False, indent=2))
    if not reg_metrics_df.empty:
        print('\n=== Regression Fold Metrics ===')
        print(reg_metrics_df.to_string(index=False))
        print('\n=== Top 15 Regression Feature Importances (mean) ===')
        print(reg_imp_df[['feature', 'mean_importance']].head(15).to_string(index=False))

if __name__ == '__main__':
    main()
