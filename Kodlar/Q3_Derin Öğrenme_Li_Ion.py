# ============================================================
# NASA dataset (Li- Ion)
# B0005, B0006, B0007, B0018
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold

from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare

import shap

try:
    import scikit_posthocs as sp
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False
    print("scikit-posthocs bulunamadı.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow bulunamadı.")

DL_LOOKBACK = 5
DL_EPOCHS = 80
DL_BATCH_SIZE = 32
DL_PATIENCE = 10
DL_SEED = 42
EPSILON = 1e-8

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. FEATURE EXTRACTION
# ============================================================

def extract_cycle_features_no_leakage_multi(df):
    df = df.copy()
    df['voltage_range'] = df.groupby('cycle')['terminal_voltage'].transform(
        lambda x: x.max() - x.min())
    df['current_range'] = df.groupby('cycle')['terminal_current'].transform(
        lambda x: x.max() - x.min())
    df['temp_range'] = df.groupby('cycle')['temperature'].transform(
        lambda x: x.max() - x.min())

    grouped = df.groupby('cycle').agg({
        'terminal_voltage': ['mean', 'std', 'min', 'max'],
        'terminal_current': ['mean', 'std', 'min', 'max'],
        'temperature': ['mean', 'std', 'min', 'max'],
        'voltage_range': 'first',
        'current_range': 'first',
        'temp_range': 'first',
        'SOH': 'first'
    })
    grouped.columns = ['_'.join(col).rstrip('_') for col in grouped.columns]
    grouped = grouped.reset_index(drop=True)
    grouped['cycle_id'] = grouped.index
    if 'SOH_first' in grouped.columns:
        grouped.rename(columns={'SOH_first': 'SOH'}, inplace=True)
    return grouped


def create_rolling_features_no_leakage_multi(df, window=3):
    df = df.copy()
    for col in ['terminal_voltage_mean', 'terminal_current_mean', 'temperature_mean']:
        if col in df.columns:
            shifted = df.groupby('battery_id')[col].shift(1)
            df[f'{col}_rolling_{window}'] = shifted.groupby(df['battery_id']).transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_rolling_std_{window}'] = shifted.groupby(df['battery_id']).transform(
                lambda x: x.rolling(window, min_periods=1).std())
    return df


def add_degradation_features_adaptive_multi(group_df):
    group_df = group_df.copy()
    window = 5
    rolling_baseline_voltage = group_df['terminal_voltage_mean'].rolling(
        window, min_periods=1).mean()
    rolling_baseline_temp = group_df['temperature_mean'].rolling(
        window, min_periods=1).mean()

    group_df['voltage_degradation_adaptive'] = (
        rolling_baseline_voltage - group_df['terminal_voltage_mean']
    ) / (rolling_baseline_voltage + EPSILON)

    group_df['temp_increase_adaptive'] = (
        group_df['temperature_mean'] - rolling_baseline_temp
    ) / (rolling_baseline_temp + EPSILON)

    group_df['voltage_drop_rate'] = group_df['terminal_voltage_mean'].diff() / (
        group_df['terminal_voltage_mean'].shift(1) + EPSILON)
    group_df['temp_rise_rate'] = group_df['temperature_mean'].diff() / (
        group_df['temperature_mean'].shift(1) + EPSILON)
    return group_df


def add_lag_features_no_initial_gt_multi(df):
    df = df.copy()
    df["SOH_lag1"] = df.groupby("battery_id")["SOH"].shift(1)
    df["SOH_lag2"] = df.groupby("battery_id")["SOH"].shift(2)
    return df


# ============================================================
# 2. SHAP
# ============================================================

def get_feature_categories(feature_cols):
    categories = {
        'voltage_features': [],
        'current_features': [],
        'temperature_features': [],
        'degradation_features': [],
        'lag_features': []
    }

    for f in feature_cols:
        f_lower = f.lower()

        if 'lag' in f_lower:
            categories['lag_features'].append(f)
            continue

        if any(k in f_lower for k in ['degradation', 'drop_rate', 'rise_rate']):
            categories['degradation_features'].append(f)
            continue
        if f_lower.endswith('_range'):
            categories['degradation_features'].append(f)
            continue

        if 'voltage' in f_lower:
            categories['voltage_features'].append(f)
        elif 'current' in f_lower:
            categories['current_features'].append(f)
        elif 'temperature' in f_lower or 'temp' in f_lower:
            categories['temperature_features'].append(f)
        else:
            categories['degradation_features'].append(f)

    return categories


def physical_interpretation_analysis(model, features, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    feature_categories = get_feature_categories(features)

    physical_importance = {}
    for category, feat_list in feature_categories.items():
        if feat_list:
            idxs = [i for i, f in enumerate(features) if f in feat_list]
            if idxs:
                physical_importance[category] = float(
                    np.mean(np.abs(shap_values[:, idxs])))
    return physical_importance


# ============================================================
# 3. RECURSIVE
# ============================================================

def recursive_forecast_deployment_realistic_multi(
        model, test_df, features, scaler,
        initial_soh_known=False, initial_soh_estimate=None):
    df = test_df.copy().reset_index(drop=True)
    preds = []

    if initial_soh_known:
        current_soh = df['SOH'].iloc[0]
    else:
        current_soh = (initial_soh_estimate
                       if initial_soh_estimate is not None else 1.0)

    for i in range(len(df)):
        if i == 0:
            pred = current_soh
        else:
            row = df.iloc[i].copy()
            if 'SOH_lag1' in features:
                row['SOH_lag1'] = preds[i - 1] if i >= 1 else current_soh
            if 'SOH_lag2' in features:
                row['SOH_lag2'] = preds[i - 2] if i >= 2 else current_soh

            X = row[features].values.reshape(1, -1)
            X_scaled = scaler.transform(X) if scaler is not None else X
            pred = model.predict(X_scaled)[0]
        preds.append(pred)
    return np.array(preds)


# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def evaluate_metrics_only(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = ~(np.isnan(y_true) | np.isnan(y_pred) |
             np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf, 'n': 0}

    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) /
                          (y_true_clean + EPSILON))) * 100
    return {'RMSE': float(rmse), 'MAE': float(mae),
            'MAPE': float(mape), 'n': int(len(y_true_clean))}


def evaluate_horizon_performance_metrics_only(
        y_true, y_pred, horizons=[1, 5, 10, 20, 50]):
    horizon_results = {}
    n = len(y_true)

    for horizon in horizons:
        if n > horizon:
            y_true_h = y_true[horizon:]
            y_pred_h = y_pred[:-horizon]
            if len(y_true_h) == len(y_pred_h) and len(y_true_h) > 0:
                horizon_results[f'horizon_{horizon}_RMSE'] = float(np.sqrt(
                    mean_squared_error(y_true_h, y_pred_h)))
                horizon_results[f'horizon_{horizon}_MAE'] = float(mean_absolute_error(
                    y_true_h, y_pred_h))
                horizon_results[f'horizon_{horizon}_MAPE'] = float(np.mean(
                    np.abs((y_true_h - y_pred_h) /
                           (y_true_h + EPSILON))) * 100)

    errors = np.abs(y_true - y_pred)
    if len(errors) > 1:
        horizon_results['absolute_error_temporal_slope_beta1'] = float(
            np.polyfit(range(len(errors)), errors, 1)[0])
    else:
        horizon_results['absolute_error_temporal_slope_beta1'] = 0.0

    horizon_results['note'] = (
        'Horizon RMSE: y(t) ile y_hat(t-h) arasındaki hata. '
        'beta1: mutlak hatanın zamansal eğilimi.')
    return horizon_results


# ============================================================
# 5. LAG-FREE ABLATION
# ============================================================

def identify_lag_free_features(feature_cols):
    return [f for f in feature_cols if 'lag' not in f.lower()]


def train_xgboost_lag_free(X_train, y_train, groups_train, use_tuning=True):
    if use_tuning:
        return tune_xgboost_battery_aware(X_train, y_train, groups_train)
    else:
        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
            random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model


def run_lag_free_ablation_xgboost(train_df, test_df, feature_cols,
                                  use_tuning=True, common_test_mask=None):
    lag_free_features = identify_lag_free_features(feature_cols)
    train_lf = train_df.dropna(subset=lag_free_features + ['SOH']).copy()

    if common_test_mask is not None:
        test_lf = test_df[common_test_mask].dropna(
            subset=lag_free_features + ['SOH']).copy()
    else:
        test_lf = test_df.dropna(subset=lag_free_features + ['SOH']).copy()

    if len(train_lf) == 0 or len(test_lf) == 0:
        return None

    X_train = train_lf[lag_free_features]
    y_train = train_lf['SOH']
    groups_train = train_lf['battery_id']

    model_lf = train_xgboost_lag_free(
        X_train, y_train, groups_train, use_tuning=use_tuning)

    X_test = test_lf[lag_free_features]
    y_test = test_lf['SOH']
    pred_lf = model_lf.predict(X_test)
    metrics_lf = evaluate_metrics_only(y_test, pred_lf)

    return {
        'model': model_lf,
        'metrics': metrics_lf,
        'features': lag_free_features,
        'n_features': len(lag_free_features),
        'predictions': pred_lf,
        'y_true': y_test.values,
        'n_test_cycles': len(test_lf)
    }


def shap_lag_free_analysis(model, feature_cols, X_sample):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        lag_free_features = identify_lag_free_features(feature_cols)
        feature_categories = get_feature_categories(lag_free_features)

        importance = {}
        for category, feat_list in feature_categories.items():
            if feat_list:
                idxs = [i for i, f in enumerate(lag_free_features) if f in feat_list]
                if idxs:
                    importance[category] = float(
                        np.mean(np.abs(shap_values[:, idxs])))
        return importance
    except Exception as e:
        print(f"   SHAP (lag-free) hatası: {e}")
        return {}


# ============================================================
# 6. HYPERPARAMETER TUNING
# ============================================================

def tune_xgboost_battery_aware(X_train, y_train, groups_train):
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [3, 4],
        'learning_rate': [0.03, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0.1, 1.0],
        'reg_lambda': [1.0, 2.0],
        'min_child_weight': [3, 5]
    }
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    unique_groups = np.unique(groups_train)
    n_splits = min(3, len(unique_groups))

    if n_splits < 2:
        print("  Not enough groups for CV, using default parameters")
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
                            random_state=42)

    group_kfold = GroupKFold(n_splits=n_splits)
    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, cv=group_kfold,
        scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train, groups=groups_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV score: {-grid_search.best_score_:.4f} RMSE")
    return grid_search.best_estimator_


# ============================================================
# 7. DEEP LEARNING
# ============================================================

def _set_tf_seed(seed=42):
    if TF_AVAILABLE:
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.10),
        LSTM(16),
        Dropout(0.10),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


def build_gru_model(input_shape):
    model = Sequential([
        GRU(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.10),
        GRU(16),
        Dropout(0.10),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


def create_sequence_data(df, feature_cols, target_col='SOH', lookback=5):
    X_seq, y_seq = [], []
    for _, group in df.groupby('battery_id', sort=False):
        group = group.sort_values('cycle_id').reset_index(drop=True)
        group = group.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
        if len(group) < lookback:
            continue
        X_values = group[feature_cols].values.astype(np.float32)
        y_values = group[target_col].values.astype(np.float32)
        for i in range(lookback - 1, len(group)):
            X_seq.append(X_values[i - lookback + 1:i + 1])
            y_seq.append(y_values[i])
    if not X_seq:
        return (np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    return (np.asarray(X_seq, dtype=np.float32),
            np.asarray(y_seq, dtype=np.float32))


def train_deep_model(model_name, X_train_seq, y_train_seq):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed.")
    _set_tf_seed(DL_SEED)
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = (build_lstm_model(input_shape) if model_name == 'LSTM'
             else build_gru_model(input_shape))
    callbacks = [EarlyStopping(monitor='val_loss', patience=DL_PATIENCE,
                               restore_best_weights=True, verbose=0)]
    model.fit(
        X_train_seq, y_train_seq, validation_split=0.10,
        epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
        shuffle=False, callbacks=callbacks, verbose=0)
    return model


def deep_learning_standard_prediction(model, test_df, feature_cols,
                                      x_scaler, y_scaler, lookback=5):
    valid = test_df.sort_values('cycle_id').copy()
    valid = valid.dropna(subset=feature_cols + ['SOH']).reset_index(drop=True)
    if len(valid) < lookback:
        return np.array([]), np.array([])
    X_raw = valid[feature_cols].values.astype(np.float32)
    X_scaled = x_scaler.transform(X_raw).astype(np.float32)
    X_seq, y_true = [], []
    for i in range(lookback - 1, len(valid)):
        X_seq.append(X_scaled[i - lookback + 1:i + 1])
        y_true.append(valid['SOH'].iloc[i])
    X_seq = np.asarray(X_seq, dtype=np.float32)
    pred_scaled = model.predict(X_seq, verbose=0).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).ravel()
    return np.asarray(y_true), pred


def deep_learning_recursive_prediction(model, test_df, feature_cols,
                                       x_scaler, y_scaler,
                                       lookback=5, initial_soh=1.0,
                                       skip_first_n=None):
    if skip_first_n is None:
        skip_first_n = lookback - 1

    df = test_df.sort_values('cycle_id').copy().reset_index(drop=True)
    if len(df) == 0:
        return np.array([]), np.array([])

    work = df.copy()
    for c in ['SOH_lag1', 'SOH_lag2']:
        if c in work.columns:
            work[c] = initial_soh

    preds = []
    raw_rows = work[feature_cols].copy()

    for i in range(len(work)):
        if i == 0:
            pred = initial_soh
        else:
            if 'SOH_lag1' in feature_cols:
                raw_rows.loc[i, 'SOH_lag1'] = preds[i - 1]
            if 'SOH_lag2' in feature_cols:
                raw_rows.loc[i, 'SOH_lag2'] = preds[i - 2] if i >= 2 else initial_soh

            start = max(0, i - lookback + 1)
            for j in range(start, i + 1):
                if 'SOH_lag1' in feature_cols:
                    raw_rows.loc[j, 'SOH_lag1'] = preds[j - 1] if j >= 1 else initial_soh
                if 'SOH_lag2' in feature_cols:
                    raw_rows.loc[j, 'SOH_lag2'] = preds[j - 2] if j >= 2 else initial_soh

            seq_raw = raw_rows.iloc[start:i + 1].values.astype(np.float32)
            if len(seq_raw) < lookback:
                pad = np.repeat(seq_raw[[0]], lookback - len(seq_raw), axis=0)
                seq_raw = np.vstack([pad, seq_raw])
            seq_scaled = x_scaler.transform(seq_raw).astype(np.float32)
            pred_scaled = model.predict(seq_scaled[np.newaxis, :, :], verbose=0)
            pred = float(y_scaler.inverse_transform(
                pred_scaled.reshape(-1, 1))[0, 0])
        preds.append(pred)

    preds = np.asarray(preds)
    y_true = df['SOH'].values.astype(float)

    if skip_first_n > 0 and len(preds) > skip_first_n:
        preds = preds[skip_first_n:]
        y_true = y_true[skip_first_n:]

    return preds, y_true


def run_deep_learning_supplementary(train_df, test_df, feature_cols,
                                    model_name, lookback=5,
                                    standard_common_mask=None,
                                    recursive_skip=None):
    if not TF_AVAILABLE:
        return None

    train_work = train_df.copy()
    test_work = test_df.copy()

    train_valid = train_work.dropna(subset=feature_cols + ['SOH']).copy()
    x_scaler = StandardScaler().fit(train_valid[feature_cols].values)
    y_scaler = StandardScaler().fit(train_valid[['SOH']].values)

    train_scaled = train_work.copy()
    train_scaled[feature_cols] = x_scaler.transform(train_scaled[feature_cols].values)
    train_scaled['SOH'] = y_scaler.transform(train_scaled[['SOH']].values).ravel()

    X_train_seq, y_train_seq = create_sequence_data(
        train_scaled, feature_cols, target_col='SOH', lookback=lookback)
    if len(X_train_seq) == 0:
        return None

    model = train_deep_model(model_name, X_train_seq, y_train_seq)

    y_std, pred_std = deep_learning_standard_prediction(
        model, test_work, feature_cols, x_scaler, y_scaler, lookback)
    metrics_std = evaluate_metrics_only(y_std, pred_std) if len(y_std) else {
        'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}

    if recursive_skip is None:
        recursive_skip = lookback - 1

    pred_rec, y_rec = deep_learning_recursive_prediction(
        model, test_work, feature_cols, x_scaler, y_scaler,
        lookback=lookback, initial_soh=1.0,
        skip_first_n=recursive_skip)
    metrics_rec = evaluate_metrics_only(y_rec, pred_rec)

    return {
        'standard': metrics_std, 'recursive': metrics_rec,
        'standard_true': y_std, 'standard_pred': pred_std,
        'recursive_true': y_rec, 'recursive_pred': pred_rec,
        'n_std': len(y_std), 'n_rec': len(y_rec)
    }

def run_lag_free_ablation_deep(train_df, test_df, feature_cols,
                               model_name, common_test_mask=None,
                               lookback=DL_LOOKBACK):
    if not TF_AVAILABLE:
        return None

    lag_free_features = identify_lag_free_features(feature_cols)
    train_lf = train_df.dropna(subset=lag_free_features + ['SOH']).copy()

    if common_test_mask is not None:
        test_lf = test_df[common_test_mask].dropna(
            subset=lag_free_features + ['SOH']).copy()
    else:
        test_lf = test_df.dropna(subset=lag_free_features + ['SOH']).copy()

    if len(train_lf) < lookback or len(test_lf) < lookback:
        return None

    train_valid = train_lf.dropna(subset=lag_free_features + ['SOH']).copy()
    x_scaler = StandardScaler().fit(train_valid[lag_free_features].values)
    y_scaler = StandardScaler().fit(train_valid[['SOH']].values)

    train_scaled = train_lf.copy()
    train_scaled[lag_free_features] = x_scaler.transform(
        train_scaled[lag_free_features].values)
    train_scaled['SOH'] = y_scaler.transform(
        train_scaled[['SOH']].values).ravel()

    X_train_seq, y_train_seq = create_sequence_data(
        train_scaled, lag_free_features, target_col='SOH', lookback=lookback)
    if len(X_train_seq) == 0:
        return None

    model = train_deep_model(model_name, X_train_seq, y_train_seq)

    y_true, pred = deep_learning_standard_prediction(
        model, test_lf, lag_free_features, x_scaler, y_scaler, lookback)
    if len(y_true) == 0:
        return None

    metrics = evaluate_metrics_only(y_true, pred)

    return {
        'model': model,
        'metrics': metrics,
        'features': lag_free_features,
        'predictions': pred,
        'y_true': y_true,
        'n_test_cycles': len(test_lf)
    }
# ============================================================
# 8. STATISTIC
# ============================================================

def bootstrap_confidence_interval(data, n_bootstrap=10000, ci=95, seed=42):
    data = np.asarray(data).ravel()
    n = len(data)
    if n == 0:
        return {'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_bootstrap': n_bootstrap, 'method': 'standard',
                'boot_mean': np.nan, 'boot_std': np.nan}

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    lower = float(np.percentile(boot_means, (100 - ci) / 2))
    upper = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return {'ci_lower': lower, 'ci_upper': upper,
            'n_bootstrap': n_bootstrap, 'method': 'standard',
            'boot_mean': float(np.mean(boot_means)),
            'boot_std': float(np.std(boot_means))}


def statistical_tests_standard_vs_recursive(std_vals, rec_vals, metric_name="RMSE"):
    std_vals = np.asarray(std_vals).ravel()
    rec_vals = np.asarray(rec_vals).ravel()

    if len(std_vals) < 2 or len(rec_vals) < 2:
        return {
            'metric': metric_name, 'n_batteries': int(min(len(std_vals), len(rec_vals))),
            'unit': 'battery-level', 'p_wilcoxon': np.nan,
            'significant_wilcoxon': False, 'cohen_d_z': np.nan,
            'effect_size': 'N/A', 'ci_95_lower': np.nan, 'ci_95_upper': np.nan,
            'bootstrap_n': 10000, 'bootstrap_method': 'standard',
            'mean_std': float(np.mean(std_vals)) if len(std_vals) else np.nan,
            'mean_rec': float(np.mean(rec_vals)) if len(rec_vals) else np.nan,
            'w_stat': np.nan, 'delta': np.nan}

    try:
        w_stat, p_w = wilcoxon(std_vals, rec_vals)
    except Exception:
        w_stat, p_w = np.nan, np.nan

    diff = std_vals - rec_vals
    diff_std = np.std(diff, ddof=1)
    cohen_d = float(np.mean(diff) / diff_std) if diff_std > 0 else 0.0
    effect_size = (
        'Büyük' if abs(cohen_d) > 0.8 else
        'Orta' if abs(cohen_d) > 0.5 else
        'Küçük' if abs(cohen_d) > 0.2 else 'İhmal Edilebilir')

    boot = bootstrap_confidence_interval(diff, n_bootstrap=10000)
    return {
        'metric': metric_name, 'n_batteries': int(len(std_vals)),
        'unit': 'battery-level',
        'p_wilcoxon': float(p_w) if not np.isnan(p_w) else np.nan,
        'significant_wilcoxon': bool(p_w < 0.05) if not np.isnan(p_w) else False,
        'cohen_d_z': cohen_d, 'effect_size': effect_size,
        'ci_95_lower': boot['ci_lower'], 'ci_95_upper': boot['ci_upper'],
        'bootstrap_n': boot['n_bootstrap'], 'bootstrap_method': boot['method'],
        'mean_std': float(np.mean(std_vals)),
        'mean_rec': float(np.mean(rec_vals)),
        'w_stat': float(w_stat) if not np.isnan(w_stat) else np.nan,
        'delta': float(np.mean(rec_vals) - np.mean(std_vals))
    }


def statistical_test_lag_li_vs_lag_free(
        lag_li_rmse_per_battery, lag_free_rmse_per_battery,
        metric_name="RMSE"):
    li = np.asarray(lag_li_rmse_per_battery).ravel()
    lf = np.asarray(lag_free_rmse_per_battery).ravel()

    if len(li) < 2 or len(lf) < 2 or len(li) != len(lf):
        return {'metric': metric_name, 'error': 'Yetersiz veri'}

    try:
        w_stat, p_w = wilcoxon(li, lf)
    except Exception:
        w_stat, p_w = np.nan, np.nan

    diff = li - lf
    diff_std = np.std(diff, ddof=1)
    cohen_d = float(np.mean(diff) / diff_std) if diff_std > 0 else 0.0

    effect_size = (
        'Büyük' if abs(cohen_d) > 0.8 else
        'Orta' if abs(cohen_d) > 0.5 else
        'Küçük' if abs(cohen_d) > 0.2 else 'İhmal Edilebilir')

    boot = bootstrap_confidence_interval(diff, n_bootstrap=10000)

    return {
        'metric': metric_name, 'n_batteries': int(len(li)),
        'unit': 'battery-level',
        'p_wilcoxon': float(p_w) if not np.isnan(p_w) else np.nan,
        'significant_wilcoxon': bool(p_w < 0.05) if not np.isnan(p_w) else False,
        'cohen_d_z': cohen_d, 'effect_size': effect_size,
        'ci_95_lower': boot['ci_lower'], 'ci_95_upper': boot['ci_upper'],
        'mean_lag_li': float(np.mean(li)),
        'mean_lag_free': float(np.mean(lf)),
        'delta': float(np.mean(li) - np.mean(lf)),
        'note': 'n=4 için betimleyici olarak yorumlanmalıdır.'
    }


def friedman_test_with_posthoc(model_results, metric='RMSE',
                               unit='battery', bonferroni=True):
    model_names = list(model_results.keys())
    data = [np.asarray(model_results[m]).ravel() for m in model_names]

    if len(data) >= 3 and all(len(d) == len(data[0]) for d in data):
        stat, p_friedman = friedmanchisquare(*data)
        significant = bool(p_friedman < 0.05)
    else:
        stat, p_friedman = np.nan, np.nan
        significant = False

    posthoc = {}
    posthoc_method = "N/A"
    bonferroni_results = {}
    n_pairs = len(model_names) * (len(model_names) - 1) // 2
    alpha_corrected = 0.05 / n_pairs if n_pairs > 0 else 0.05

    if significant and len(model_names) >= 3:
        if SP_AVAILABLE:
            try:
                data_matrix = pd.DataFrame(
                    {name: data[i] for i, name in enumerate(model_names)})
                nemenyi_result = sp.posthoc_nemenyi_friedman(data_matrix)
                posthoc_method = "Nemenyi (rank-based post-hoc)"
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        p_val = nemenyi_result.iloc[i, j]
                        posthoc[f"{model_names[i]} vs {model_names[j]}"] = {
                            'p_value': float(p_val),
                            'significant': bool(p_val < 0.05) if not np.isnan(p_val) else False,
                            'method': 'Nemenyi'}
            except Exception as e:
                print(f"  Nemenyi hatası: {e}")

        if bonferroni:
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    try:
                        _, p_pair = wilcoxon(data[i], data[j])
                    except Exception:
                        p_pair = np.nan
                    bonferroni_results[f"{model_names[i]}_vs_{model_names[j]}"] = {
                        'p_raw': float(p_pair) if not np.isnan(p_pair) else np.nan,
                        'p_bonferroni': float(min(p_pair * n_pairs, 1.0))
                        if not np.isnan(p_pair) else np.nan,
                        'significant_bonferroni': bool(p_pair < alpha_corrected)
                        if not np.isnan(p_pair) else False}

    return {
        'metric': metric, 'unit': unit,
        'n_observations': int(len(data[0])) if data else 0,
        'n_models': len(model_names),
        'statistic': float(stat) if not np.isnan(stat) else np.nan,
        'p_value': float(p_friedman) if not np.isnan(p_friedman) else np.nan,
        'significant': significant,
        'posthoc': posthoc, 'posthoc_method': posthoc_method,
        'bonferroni_alpha': float(alpha_corrected),
        'bonferroni_pairs': bonferroni_results
    }


def compute_descriptive_per_battery_rank(results_df, metric='recursive_RMSE',
                                         model_names=['XGBoost', 'LSTM', 'GRU']):
    print("\n" + "=" * 70)
    print(f"DESCRIPTIVE PER-BATTERY RANK (metric={metric})")
    print("=" * 70)

    model_cols = {
        'XGBoost': f'xgb_{metric}', 'LSTM': f'lstm_{metric}',
        'GRU': f'gru_{metric}'}
    available_models = [m for m in model_names if model_cols[m] in results_df.columns]

    if len(available_models) < 2:
        return {}

    batteries = results_df['battery'].values
    rank_matrix = []
    for battery in batteries:
        idx = results_df[results_df['battery'] == battery].index[0]
        vals = [results_df.loc[idx, model_cols[m]] for m in available_models]
        rank_matrix.append(stats.rankdata(vals))

    rank_matrix = np.array(rank_matrix)
    rankings = {}
    for i, model in enumerate(available_models):
        rankings[model] = {
            'mean_rank': float(np.mean(rank_matrix[:, i])),
            'ranks_per_battery': rank_matrix[:, i].tolist(),
            'batteries': [str(b) for b in batteries]}

    sorted_models = sorted(rankings.items(), key=lambda x: x[1]['mean_rank'])
    for rank_pos, (model, info) in enumerate(sorted_models, start=1):
        print(f"  #{rank_pos} {model}: mean rank = {info['mean_rank']:.2f}")
    return rankings


def comprehensive_statistical_report_multi(
        results_df, all_predictions,
        model_names=['XGBoost', 'LSTM', 'GRU']):
    report = {}

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: NASA (BATTERY-LEVEL, n=4)")
    print("=" * 70)

    model_map = {
        'XGBoost': ('xgb_standard_RMSE', 'xgb_recursive_RMSE',
                    'xgb_standard_MAE', 'xgb_recursive_MAE',
                    'xgb_standard_MAPE', 'xgb_recursive_MAPE'),
        'LSTM': ('lstm_standard_RMSE', 'lstm_recursive_RMSE',
                 'lstm_standard_MAE', 'lstm_recursive_MAE',
                 'lstm_standard_MAPE', 'lstm_recursive_MAPE'),
        'GRU': ('gru_standard_RMSE', 'gru_recursive_RMSE',
                'gru_standard_MAE', 'gru_recursive_MAE',
                'gru_standard_MAPE', 'gru_recursive_MAPE')}

    for model in model_names:
        if model in model_map:
            std_rmse, rec_rmse, std_mae, rec_mae, std_mape, rec_mape = model_map[model]
            print(f"\n{model}:")
            for metric_name, std_col, rec_col in [
                ('RMSE', std_rmse, rec_rmse),
                ('MAE', std_mae, rec_mae),
                ('MAPE', std_mape, rec_mape)]:
                test = statistical_tests_standard_vs_recursive(
                    results_df[std_col].values, results_df[rec_col].values,
                    metric_name)
                print(f"  {metric_name}: Std={test['mean_std']:.6f}, "
                      f"Rec={test['mean_rec']:.6f}, "
                      f"Cohen's d_z={test['cohen_d_z']:.4f}")
                report[f'{model}_{metric_name}_test'] = test

    if len(results_df) >= 3:
        model_results = {
            'XGBoost': results_df['xgb_recursive_RMSE'].values,
            'LSTM': results_df['lstm_recursive_RMSE'].values,
            'GRU': results_df['gru_recursive_RMSE'].values}
        friedman_results = friedman_test_with_posthoc(
            model_results, metric='RMSE', unit='battery', bonferroni=True)
        report['friedman_test'] = friedman_results
        print(f"\nFriedman chi2={friedman_results['statistic']:.4f}, "
              f"p={friedman_results['p_value']:.6f}")

    report['descriptive_per_battery_rank'] = compute_descriptive_per_battery_rank(
        results_df, metric='recursive_RMSE', model_names=model_names)

    return report


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main_pipeline_multi(files):
    print("=" * 70)
    print("MULTI-BATTERY SOH PREDICTION - MAJOR FIX v5")
    print("=" * 70)

    base_path = os.path.dirname(list(files.values())[0])
    output_dir = os.path.join(base_path, "soh_analysis_results_multi_majorfix_v5")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    all_data = []
    for name, path in files.items():
        print(f"   Loading {name}...")
        df = pd.read_csv(path)
        df_feat = extract_cycle_features_no_leakage_multi(df)
        df_feat['battery_id'] = name
        all_data.append(df_feat)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['battery_id', 'cycle_id'])
    final_df = create_rolling_features_no_leakage_multi(final_df, window=3)
    final_df = final_df.groupby('battery_id').apply(
        add_degradation_features_adaptive_multi).reset_index(drop=True)

    exclude_cols = ['SOH', 'battery_id', 'cycle_id']
    feature_cols = [c for c in final_df.columns if c not in exclude_cols]
    batteries = final_df['battery_id'].unique()

    print(f"\n2. Features: {len(feature_cols)}, Batteries: {batteries}")
    print(f"   Total samples: {len(final_df)}")

    results = {k: [] for k in [
        'battery',
        'xgb_standard_RMSE', 'xgb_recursive_RMSE',
        'xgb_standard_MAE', 'xgb_recursive_MAE',
        'xgb_standard_MAPE', 'xgb_recursive_MAPE',
        'xgb_beta1_std', 'xgb_beta1_rec',
        'xgb_horizon_std_RMSE_h1', 'xgb_horizon_rec_RMSE_h1',
        'xgb_horizon_std_RMSE_h5', 'xgb_horizon_rec_RMSE_h5',
        'xgb_horizon_std_RMSE_h10', 'xgb_horizon_rec_RMSE_h10',
        'xgb_horizon_std_RMSE_h20', 'xgb_horizon_rec_RMSE_h20',
        'xgb_horizon_std_RMSE_h50', 'xgb_horizon_rec_RMSE_h50',
        'lstm_standard_RMSE', 'lstm_recursive_RMSE',
        'lstm_standard_MAE', 'lstm_recursive_MAE',
        'lstm_standard_MAPE', 'lstm_recursive_MAPE',
        'lstm_beta1_std', 'lstm_beta1_rec',
        'lstm_horizon_std_RMSE_h1', 'lstm_horizon_rec_RMSE_h1',
        'lstm_horizon_std_RMSE_h5', 'lstm_horizon_rec_RMSE_h5',
        'lstm_horizon_std_RMSE_h10', 'lstm_horizon_rec_RMSE_h10',
        'lstm_horizon_std_RMSE_h20', 'lstm_horizon_rec_RMSE_h20',
        'lstm_horizon_std_RMSE_h50', 'lstm_horizon_rec_RMSE_h50',
        'gru_standard_RMSE', 'gru_recursive_RMSE',
        'gru_standard_MAE', 'gru_recursive_MAE',
        'gru_standard_MAPE', 'gru_recursive_MAPE',
        'gru_beta1_std', 'gru_beta1_rec',
        'gru_horizon_std_RMSE_h1', 'gru_horizon_rec_RMSE_h1',
        'gru_horizon_std_RMSE_h5', 'gru_horizon_rec_RMSE_h5',
        'gru_horizon_std_RMSE_h10', 'gru_horizon_rec_RMSE_h10',
        'gru_horizon_std_RMSE_h20', 'gru_horizon_rec_RMSE_h20',
        'gru_horizon_std_RMSE_h50', 'gru_horizon_rec_RMSE_h50',
        'persistence_RMSE', 'persistence_MAE', 'persistence_MAPE']}

    all_predictions = {}
    all_shap_results = {}
    ablation_lag_li = {'XGBoost': {}, 'LSTM': {}, 'GRU': {}}
    ablation_lag_free = {'XGBoost': {}, 'LSTM': {}, 'GRU': {}}
    ablation_shap_results = {}

    for test_battery in batteries:
        print(f"\n{'=' * 50}\nTEST BATTERY: {test_battery}\n{'=' * 50}")

        train_df = final_df[final_df['battery_id'] != test_battery].copy()
        test_df = final_df[final_df['battery_id'] == test_battery].copy()

        train_df = add_lag_features_no_initial_gt_multi(train_df).dropna()
        test_df_full = add_lag_features_no_initial_gt_multi(test_df)

        lag_features = feature_cols + ['SOH_lag1', 'SOH_lag2']
        lag_features = [f for f in lag_features if f in train_df.columns]

        common_test_mask = test_df_full[['SOH_lag1', 'SOH_lag2']].notna().all(axis=1)
        test_df_common = test_df_full[common_test_mask].copy()

        # XGBoost
        print("   XGBoost (lag'li)...")
        X_train = train_df[lag_features]
        y_train = train_df['SOH']
        groups_train = train_df['battery_id']
        model_xgb = tune_xgboost_battery_aware(X_train, y_train, groups_train)

        X_test_common = test_df_common[lag_features]
        y_test_common = test_df_common['SOH']
        pred_xgb_std = model_xgb.predict(X_test_common)
        m_xgb_std = evaluate_metrics_only(y_test_common, pred_xgb_std)

        pred_xgb_rec = recursive_forecast_deployment_realistic_multi(
            model_xgb, test_df_common, lag_features, scaler=None,
            initial_soh_known=False)
        m_xgb_rec = evaluate_metrics_only(
            test_df_common['SOH'].values, pred_xgb_rec)

        horizon_xgb_std = evaluate_horizon_performance_metrics_only(
            test_df_common['SOH'].values, pred_xgb_std)
        horizon_xgb_rec = evaluate_horizon_performance_metrics_only(
            test_df_common['SOH'].values, pred_xgb_rec)

        print(f"   XGB Std (common test) -> RMSE: {m_xgb_std['RMSE']:.6f}, n={m_xgb_std['n']}")
        print(f"   XGB Rec (common test) -> RMSE: {m_xgb_rec['RMSE']:.6f}, n={m_xgb_rec['n']}")
        print(f"   XGB beta1 Std: {horizon_xgb_std.get('absolute_error_temporal_slope_beta1', np.nan):.6f}, "
              f"Rec: {horizon_xgb_rec.get('absolute_error_temporal_slope_beta1', np.nan):.6f}")

        all_predictions[test_battery] = {
            'true': test_df_common['SOH'].values,
            'recursive': pred_xgb_rec, 'standard': pred_xgb_std}

        ablation_lag_li['XGBoost'][test_battery] = {
            'standard_RMSE': m_xgb_std['RMSE'],
            'standard_MAE': m_xgb_std['MAE'],
            'recursive_RMSE': m_xgb_rec['RMSE'],
            'recursive_MAE': m_xgb_rec['MAE']}

        print("   XGBoost (lag'siz ablation, tuned)...")
        lag_free_out = run_lag_free_ablation_xgboost(
            train_df, test_df_full, feature_cols,
            use_tuning=True, common_test_mask=common_test_mask)
        if lag_free_out:
            m_xgb_lf_std = lag_free_out['metrics']
            print(f"   XGB Lag-siz Std -> RMSE: {m_xgb_lf_std['RMSE']:.6f}")
            ablation_lag_free['XGBoost'][test_battery] = {
                'standard_RMSE': m_xgb_lf_std['RMSE'],
                'standard_MAE': m_xgb_lf_std['MAE']}

            try:
                X_test_lf = test_df_common[lag_free_out['features']].iloc[:100].values
                ablation_shap_results[test_battery] = shap_lag_free_analysis(
                    lag_free_out['model'], lag_free_out['features'], X_test_lf)
            except Exception as e:
                print(f"   SHAP (lag-free) hatası: {e}")

        try:
            ns = min(100, len(X_test_common))
            shap_res = physical_interpretation_analysis(
                model_xgb, lag_features, X_test_common.iloc[:ns].values)
            all_shap_results[test_battery] = shap_res
        except Exception as e:
            print(f"   SHAP note: {e}")

        y_tp = test_df_common['SOH'].values[1:]
        y_pp = test_df_common['SOH'].shift(1).dropna().values
        m_pers = evaluate_metrics_only(y_tp, y_pp) if len(y_tp) > 0 else {
            'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}

        # LSTM
        if TF_AVAILABLE:
            print("   LSTM (lag'li)...")
            try:
                lstm_out = run_deep_learning_supplementary(
                    train_df, test_df_full, lag_features, 'LSTM', DL_LOOKBACK)
                if lstm_out:
                    m_lstm_std = lstm_out['standard']
                    m_lstm_rec = lstm_out['recursive']
                    all_predictions[test_battery]['lstm_recursive'] = lstm_out['recursive_pred']
                    ablation_lag_li['LSTM'][test_battery] = {
                        'standard_RMSE': m_lstm_std['RMSE'],
                        'standard_MAE': m_lstm_std['MAE'],
                        'recursive_RMSE': m_lstm_rec['RMSE'],
                        'recursive_MAE': m_lstm_rec['MAE']}

                    horizon_lstm_std = evaluate_horizon_performance_metrics_only(
                        lstm_out['standard_true'], lstm_out['standard_pred'])
                    horizon_lstm_rec = evaluate_horizon_performance_metrics_only(
                        lstm_out['recursive_true'], lstm_out['recursive_pred'])

                    print(f"   LSTM Std -> RMSE: {m_lstm_std['RMSE']:.6f}, "
                          f"Rec -> RMSE: {m_lstm_rec['RMSE']:.6f}")
                    print(f"   LSTM beta1 Std: {horizon_lstm_std.get('absolute_error_temporal_slope_beta1', np.nan):.6f}, "
                          f"Rec: {horizon_lstm_rec.get('absolute_error_temporal_slope_beta1', np.nan):.6f}")
                else:
                    raise RuntimeError('No sequence data')
            except Exception as e:
                print(f"   LSTM failed: {e}")
                m_lstm_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
                m_lstm_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
                horizon_lstm_std = {'absolute_error_temporal_slope_beta1': np.nan}
                horizon_lstm_rec = {'absolute_error_temporal_slope_beta1': np.nan}
        else:
            m_lstm_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
            m_lstm_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
            horizon_lstm_std = {'absolute_error_temporal_slope_beta1': np.nan}
            horizon_lstm_rec = {'absolute_error_temporal_slope_beta1': np.nan}

        if TF_AVAILABLE:
            try:
                lstm_lf = run_lag_free_ablation_deep(
                    train_df, test_df_full, feature_cols, 'LSTM',
                    common_test_mask=common_test_mask)
                if lstm_lf:
                    m_lstm_lf = lstm_lf['metrics']
                    ablation_lag_free['LSTM'][test_battery] = {
                        'standard_RMSE': m_lstm_lf['RMSE'],
                        'standard_MAE': m_lstm_lf['MAE']}
            except Exception as e:
                print(f"   LSTM lag-free failed: {e}")

        # GRU
        if TF_AVAILABLE:
            print("   GRU (lag'li)...")
            try:
                gru_out = run_deep_learning_supplementary(
                    train_df, test_df_full, lag_features, 'GRU', DL_LOOKBACK)
                if gru_out:
                    m_gru_std = gru_out['standard']
                    m_gru_rec = gru_out['recursive']
                    all_predictions[test_battery]['gru_recursive'] = gru_out['recursive_pred']
                    ablation_lag_li['GRU'][test_battery] = {
                        'standard_RMSE': m_gru_std['RMSE'],
                        'standard_MAE': m_gru_std['MAE'],
                        'recursive_RMSE': m_gru_rec['RMSE'],
                        'recursive_MAE': m_gru_rec['MAE']}

                    horizon_gru_std = evaluate_horizon_performance_metrics_only(
                        gru_out['standard_true'], gru_out['standard_pred'])
                    horizon_gru_rec = evaluate_horizon_performance_metrics_only(
                        gru_out['recursive_true'], gru_out['recursive_pred'])

                    print(f"   GRU Std -> RMSE: {m_gru_std['RMSE']:.6f}, "
                          f"Rec -> RMSE: {m_gru_rec['RMSE']:.6f}")
                    print(f"   GRU beta1 Std: {horizon_gru_std.get('absolute_error_temporal_slope_beta1', np.nan):.6f}, "
                          f"Rec: {horizon_gru_rec.get('absolute_error_temporal_slope_beta1', np.nan):.6f}")
                else:
                    raise RuntimeError('No sequence data')
            except Exception as e:
                print(f"   GRU failed: {e}")
                m_gru_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
                m_gru_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
                horizon_gru_std = {'absolute_error_temporal_slope_beta1': np.nan}
                horizon_gru_rec = {'absolute_error_temporal_slope_beta1': np.nan}
        else:
            m_gru_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
            m_gru_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
            horizon_gru_std = {'absolute_error_temporal_slope_beta1': np.nan}
            horizon_gru_rec = {'absolute_error_temporal_slope_beta1': np.nan}

        if TF_AVAILABLE:
            try:
                gru_lf = run_lag_free_ablation_deep(
                    train_df, test_df_full, feature_cols, 'GRU',
                    common_test_mask=common_test_mask)
                if gru_lf:
                    m_gru_lf = gru_lf['metrics']
                    ablation_lag_free['GRU'][test_battery] = {
                        'standard_RMSE': m_gru_lf['RMSE'],
                        'standard_MAE': m_gru_lf['MAE']}
            except Exception as e:
                print(f"   GRU lag-free failed: {e}")

        # Sonuçları kaydet
        results['battery'].append(test_battery)
        for prefix, m_std, m_rec in [('xgb', m_xgb_std, m_xgb_rec),
                                     ('lstm', m_lstm_std, m_lstm_rec),
                                     ('gru', m_gru_std, m_gru_rec)]:
            results[f'{prefix}_standard_RMSE'].append(m_std['RMSE'])
            results[f'{prefix}_recursive_RMSE'].append(m_rec['RMSE'])
            results[f'{prefix}_standard_MAE'].append(m_std['MAE'])
            results[f'{prefix}_recursive_MAE'].append(m_rec['MAE'])
            results[f'{prefix}_standard_MAPE'].append(m_std['MAPE'])
            results[f'{prefix}_recursive_MAPE'].append(m_rec['MAPE'])

        results['xgb_beta1_std'].append(horizon_xgb_std.get('absolute_error_temporal_slope_beta1', np.nan))
        results['xgb_beta1_rec'].append(horizon_xgb_rec.get('absolute_error_temporal_slope_beta1', np.nan))
        results['xgb_horizon_std_RMSE_h1'].append(horizon_xgb_std.get('horizon_1_RMSE', np.nan))
        results['xgb_horizon_rec_RMSE_h1'].append(horizon_xgb_rec.get('horizon_1_RMSE', np.nan))
        results['xgb_horizon_std_RMSE_h5'].append(horizon_xgb_std.get('horizon_5_RMSE', np.nan))
        results['xgb_horizon_rec_RMSE_h5'].append(horizon_xgb_rec.get('horizon_5_RMSE', np.nan))
        results['xgb_horizon_std_RMSE_h10'].append(horizon_xgb_std.get('horizon_10_RMSE', np.nan))
        results['xgb_horizon_rec_RMSE_h10'].append(horizon_xgb_rec.get('horizon_10_RMSE', np.nan))
        results['xgb_horizon_std_RMSE_h20'].append(horizon_xgb_std.get('horizon_20_RMSE', np.nan))
        results['xgb_horizon_rec_RMSE_h20'].append(horizon_xgb_rec.get('horizon_20_RMSE', np.nan))
        results['xgb_horizon_std_RMSE_h50'].append(horizon_xgb_std.get('horizon_50_RMSE', np.nan))
        results['xgb_horizon_rec_RMSE_h50'].append(horizon_xgb_rec.get('horizon_50_RMSE', np.nan))

        results['lstm_beta1_std'].append(horizon_lstm_std.get('absolute_error_temporal_slope_beta1', np.nan))
        results['lstm_beta1_rec'].append(horizon_lstm_rec.get('absolute_error_temporal_slope_beta1', np.nan))
        results['lstm_horizon_std_RMSE_h1'].append(horizon_lstm_std.get('horizon_1_RMSE', np.nan))
        results['lstm_horizon_rec_RMSE_h1'].append(horizon_lstm_rec.get('horizon_1_RMSE', np.nan))
        results['lstm_horizon_std_RMSE_h5'].append(horizon_lstm_std.get('horizon_5_RMSE', np.nan))
        results['lstm_horizon_rec_RMSE_h5'].append(horizon_lstm_rec.get('horizon_5_RMSE', np.nan))
        results['lstm_horizon_std_RMSE_h10'].append(horizon_lstm_std.get('horizon_10_RMSE', np.nan))
        results['lstm_horizon_rec_RMSE_h10'].append(horizon_lstm_rec.get('horizon_10_RMSE', np.nan))
        results['lstm_horizon_std_RMSE_h20'].append(horizon_lstm_std.get('horizon_20_RMSE', np.nan))
        results['lstm_horizon_rec_RMSE_h20'].append(horizon_lstm_rec.get('horizon_20_RMSE', np.nan))
        results['lstm_horizon_std_RMSE_h50'].append(horizon_lstm_std.get('horizon_50_RMSE', np.nan))
        results['lstm_horizon_rec_RMSE_h50'].append(horizon_lstm_rec.get('horizon_50_RMSE', np.nan))

        results['gru_beta1_std'].append(horizon_gru_std.get('absolute_error_temporal_slope_beta1', np.nan))
        results['gru_beta1_rec'].append(horizon_gru_rec.get('absolute_error_temporal_slope_beta1', np.nan))
        results['gru_horizon_std_RMSE_h1'].append(horizon_gru_std.get('horizon_1_RMSE', np.nan))
        results['gru_horizon_rec_RMSE_h1'].append(horizon_gru_rec.get('horizon_1_RMSE', np.nan))
        results['gru_horizon_std_RMSE_h5'].append(horizon_gru_std.get('horizon_5_RMSE', np.nan))
        results['gru_horizon_rec_RMSE_h5'].append(horizon_gru_rec.get('horizon_5_RMSE', np.nan))
        results['gru_horizon_std_RMSE_h10'].append(horizon_gru_std.get('horizon_10_RMSE', np.nan))
        results['gru_horizon_rec_RMSE_h10'].append(horizon_gru_rec.get('horizon_10_RMSE', np.nan))
        results['gru_horizon_std_RMSE_h20'].append(horizon_gru_std.get('horizon_20_RMSE', np.nan))
        results['gru_horizon_rec_RMSE_h20'].append(horizon_gru_rec.get('horizon_20_RMSE', np.nan))
        results['gru_horizon_std_RMSE_h50'].append(horizon_gru_std.get('horizon_50_RMSE', np.nan))
        results['gru_horizon_rec_RMSE_h50'].append(horizon_gru_rec.get('horizon_50_RMSE', np.nan))

        results['persistence_RMSE'].append(m_pers['RMSE'])
        results['persistence_MAE'].append(m_pers['MAE'])
        results['persistence_MAPE'].append(m_pers['MAPE'])

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'results_multi_v5.csv'), index=False)

    if all_shap_results:
        pd.DataFrame(all_shap_results).T.to_csv(
            os.path.join(output_dir, 'shap_lagli_v5.csv'))

    print("\n" + "=" * 70)
    print("LAG-FREE ABLATION ÖZETİ")
    print("=" * 70)

    ablation_rows = []
    ablation_stat_tests = {}
    for model_name in ['XGBoost', 'LSTM', 'GRU']:
        li = ablation_lag_li.get(model_name, {})
        lf = ablation_lag_free.get(model_name, {})
        if li and lf:
            batteries_list = [b for b in batteries if b in li and b in lf]
            li_vals = [li[b]['standard_RMSE'] for b in batteries_list]
            lf_vals = [lf[b]['standard_RMSE'] for b in batteries_list]

            stat_test = statistical_test_lag_li_vs_lag_free(
                li_vals, lf_vals, metric_name=f'{model_name}_RMSE')
            ablation_stat_tests[model_name] = stat_test

            ablation_rows.append({
                'Model': model_name,
                'Lag-li Std RMSE': np.mean(li_vals),
                'Lag-siz Std RMSE': np.mean(lf_vals),
                'DeltaRMSE': np.mean(li_vals) - np.mean(lf_vals),
                'Wilcoxon p': stat_test.get('p_wilcoxon', np.nan),
                "Cohen's dz": stat_test.get('cohen_d_z', np.nan)})

    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        print(ablation_df.to_string(index=False))
        ablation_df.to_csv(os.path.join(output_dir, 'ablation_v5.csv'), index=False)

    model_names = ['XGBoost', 'LSTM', 'GRU'] if TF_AVAILABLE else ['XGBoost']
    stat_report = comprehensive_statistical_report_multi(
        results_df, all_predictions, model_names)
    stat_report['ablation_lag_free'] = {
        'results': ablation_stat_tests,
        'summary_table': ablation_df.to_dict('records') if not ablation_df.empty else []}

    with open(os.path.join(output_dir, 'stat_report_v5.json'),
              'w', encoding='utf-8') as f:
        json.dump(stat_report, f, indent=2, default=str, ensure_ascii=False)

    print("\nAnalysis complete!")
    return results_df, all_predictions, stat_report


if __name__ == "__main__":
    base_path = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree"

    files = {
        "B0005": os.path.join(base_path, "B0005_discharge_soh_standardized.csv"),
        "B0006": os.path.join(base_path, "B0006_discharge_soh_standardized.csv"),
        "B0007": os.path.join(base_path, "B0007_discharge_soh_standardized.csv"),
        "B0018": os.path.join(base_path, "B0018_discharge_soh_standardized.csv")}

    for name, path in files.items():
        print(f"  {'OK' if os.path.exists(path) else 'EKSIK'} {name}")

    results_df, predictions, stat_report = main_pipeline_multi(files)
    print("\nNASA ANALİZİ TAMAMLANDI !")