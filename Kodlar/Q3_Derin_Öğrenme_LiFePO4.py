# ============================================================
# LiFePO4 Dataset
# CH17
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
from scipy.stats import wilcoxon
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shap

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

XGB_PARAMS = {
    'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 3,
    'random_state': 42}

EPSILON = 1e-8
BLOCK_SIZE_DEFAULT = 10
N_BOOTSTRAP_DEFAULT = 10000

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. FEATURE EXTRACTION
# ============================================================

def extract_cycle_features_no_leakage_single(df):
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
        'voltage_range': 'first', 'current_range': 'first',
        'temp_range': 'first', 'SOH': 'first'})
    grouped.columns = ['_'.join(col).rstrip('_') for col in grouped.columns]
    grouped = grouped.reset_index(drop=True)
    grouped['cycle_id'] = grouped.index
    if 'SOH_first' in grouped.columns:
        grouped.rename(columns={'SOH_first': 'SOH'}, inplace=True)
    return grouped


def create_rolling_features_no_leakage_single(df, window=3):
    df = df.copy()
    for col in ['terminal_voltage_mean', 'terminal_current_mean', 'temperature_mean']:
        if col in df.columns:
            shifted = df[col].shift(1)
            df[f'{col}_rolling_{window}'] = shifted.rolling(window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}'] = shifted.rolling(window, min_periods=1).std()
    return df


def add_degradation_features_adaptive_single(group_df):
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


def add_lag_features_no_initial_gt_single(df):
    df = df.copy()
    df["SOH_lag1"] = df["SOH"].shift(1)
    df["SOH_lag2"] = df["SOH"].shift(2)
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
    importance = {}
    for cat, feat_list in feature_categories.items():
        if feat_list:
            idxs = [i for i, f in enumerate(features) if f in feat_list]
            if idxs:
                importance[cat] = float(np.mean(np.abs(shap_values[:, idxs])))
    return importance


# ============================================================
# 3. RECURSIVE FORECAST
# ============================================================

def recursive_forecast_deployment_realistic_single(
        model, test_df, features, scaler,
        initial_soh_known=False, initial_soh_estimate=None):
    df = test_df.copy().reset_index(drop=True)
    preds = []
    if initial_soh_known:
        current_soh = df['SOH'].iloc[0]
    else:
        current_soh = (initial_soh_estimate if initial_soh_estimate is not None else 1.0)

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
    ytc = y_true[mask]
    ypc = y_pred[mask]
    if len(ytc) == 0:
        return {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf, 'n': 0}
    rmse = np.sqrt(mean_squared_error(ytc, ypc))
    mae = mean_absolute_error(ytc, ypc)
    mape = np.mean(np.abs((ytc - ypc) / (ytc + EPSILON))) * 100
    return {'RMSE': float(rmse), 'MAE': float(mae),
            'MAPE': float(mape), 'n': int(len(ytc))}


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
                    np.abs((y_true_h - y_pred_h) / (y_true_h + EPSILON))) * 100)

    errors = np.abs(y_true - y_pred)
    if len(errors) > 1:
        horizon_results['absolute_error_temporal_slope_beta1'] = float(
            np.polyfit(range(len(errors)), errors, 1)[0])
    else:
        horizon_results['absolute_error_temporal_slope_beta1'] = 0.0
    return horizon_results


# ============================================================
# 5. LAG-FREE ABLATION
# ============================================================

def identify_lag_free_features(feature_cols):
    return [f for f in feature_cols if 'lag' not in f.lower()]


def train_xgboost_lag_free_single(X_train, y_train):
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def run_lag_free_ablation_xgboost_single(train_df, test_df, feature_cols,
                                         common_test_mask=None):
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
    model_lf = train_xgboost_lag_free_single(X_train, y_train)

    X_test = test_lf[lag_free_features]
    y_test = test_lf['SOH']
    pred_lf = model_lf.predict(X_test)
    metrics_lf = evaluate_metrics_only(y_test, pred_lf)

    return {
        'model': model_lf, 'metrics': metrics_lf,
        'features': lag_free_features,
        'predictions': pred_lf, 'y_true': y_test.values}


def shap_lag_free_analysis(model, feature_cols, X_sample):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        lag_free_features = identify_lag_free_features(feature_cols)
        feature_categories = get_feature_categories(lag_free_features)
        importance = {}
        for cat, feat_list in feature_categories.items():
            if feat_list:
                idxs = [i for i, f in enumerate(lag_free_features) if f in feat_list]
                if idxs:
                    importance[cat] = float(np.mean(np.abs(shap_values[:, idxs])))
        return importance
    except Exception as e:
        print(f"   SHAP (lag-free) hatası: {e}")
        return {}


# ============================================================
# 6. DEEP LEARNING
# ============================================================

def _set_tf_seed(seed=42):
    if TF_AVAILABLE:
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.10), LSTM(16), Dropout(0.10),
        Dense(16, activation='relu'), Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


def build_gru_model(input_shape):
    model = Sequential([
        GRU(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.10), GRU(16), Dropout(0.10),
        Dense(16, activation='relu'), Dense(1)])
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
    _set_tf_seed(DL_SEED)
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = (build_lstm_model(input_shape) if model_name == 'LSTM'
             else build_gru_model(input_shape))
    callbacks = [EarlyStopping(monitor='val_loss', patience=DL_PATIENCE,
                               restore_best_weights=True, verbose=0)]
    model.fit(X_train_seq, y_train_seq, validation_split=0.10,
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
        lookback=lookback, initial_soh=1.0, skip_first_n=recursive_skip)
    metrics_rec = evaluate_metrics_only(y_rec, pred_rec)

    return {'standard': metrics_std, 'recursive': metrics_rec,
            'standard_true': y_std, 'standard_pred': pred_std,
            'recursive_true': y_rec, 'recursive_pred': pred_rec}


def run_lag_free_ablation_deep_single(train_df, test_df, feature_cols,
                                      model_name, common_test_mask=None):
    if not TF_AVAILABLE:
        return None
    lag_free_features = identify_lag_free_features(feature_cols)
    train_lf = train_df.dropna(subset=lag_free_features + ['SOH']).copy()

    if common_test_mask is not None:
        test_lf = test_df[common_test_mask].dropna(
            subset=lag_free_features + ['SOH']).copy()
    else:
        test_lf = test_df.dropna(subset=lag_free_features + ['SOH']).copy()

    if len(train_lf) < DL_LOOKBACK or len(test_lf) < DL_LOOKBACK:
        return None

    x_scaler = StandardScaler().fit(train_lf[lag_free_features].values)
    y_scaler = StandardScaler().fit(train_lf[['SOH']].values)
    train_scaled = train_lf.copy()
    train_scaled[lag_free_features] = x_scaler.transform(
        train_scaled[lag_free_features].values)
    train_scaled['SOH'] = y_scaler.transform(
        train_scaled[['SOH']].values).ravel()
    X_train_seq, y_train_seq = create_sequence_data(
        train_scaled, lag_free_features, target_col='SOH', lookback=DL_LOOKBACK)
    if len(X_train_seq) == 0:
        return None
    model = train_deep_model(model_name, X_train_seq, y_train_seq)
    y_true, pred = deep_learning_standard_prediction(
        model, test_lf, lag_free_features, x_scaler, y_scaler, DL_LOOKBACK)
    if len(y_true) == 0:
        return None
    metrics = evaluate_metrics_only(y_true, pred)
    return {'model': model, 'metrics': metrics,
            'features': lag_free_features,
            'predictions': pred, 'y_true': y_true}


# ============================================================
# 7. STATISCTIC
# ============================================================

def bootstrap_confidence_interval_block(data, n_bootstrap=N_BOOTSTRAP_DEFAULT,
                                        ci=95, block_size=BLOCK_SIZE_DEFAULT,
                                        seed=42):
    data = np.asarray(data).ravel()
    n = len(data)
    if n == 0:
        return {'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_bootstrap': n_bootstrap, 'method': 'block',
                'block_size': block_size}

    rng = np.random.default_rng(seed)
    boot_means = []
    n_blocks = max(2, n // block_size)
    block_indices = np.array_split(np.arange(n), n_blocks)
    for _ in range(n_bootstrap):
        sampled = rng.choice(len(block_indices), size=len(block_indices), replace=True)
        idx = np.concatenate([block_indices[b] for b in sampled])
        boot_means.append(np.mean(data[idx]))
    boot_means = np.array(boot_means)
    lower = float(np.percentile(boot_means, (100 - ci) / 2))
    upper = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return {'ci_lower': lower, 'ci_upper': upper,
            'n_bootstrap': n_bootstrap, 'method': 'block',
            'block_size': block_size,
            'boot_mean': float(np.mean(boot_means)),
            'boot_std': float(np.std(boot_means))}


def cycle_level_wilcoxon_lfp(y_true, std_pred, rec_pred,
                             block_size=BLOCK_SIZE_DEFAULT,
                             n_bootstrap=N_BOOTSTRAP_DEFAULT):
    print("\n" + "=" * 70)
    print("CYCLE-LEVEL WILCOXON (Standard vs Recursive)")
    print("=" * 70)

    n = min(len(y_true), len(std_pred), len(rec_pred))
    y_true = np.asarray(y_true[:n])
    std_pred = np.asarray(std_pred[:n])
    rec_pred = np.asarray(rec_pred[:n])

    std_errors = np.abs(y_true - std_pred)
    rec_errors = np.abs(y_true - rec_pred)
    diff = std_errors - rec_errors

    try:
        w_stat, p_raw = wilcoxon(std_errors, rec_errors)
        diff_std = np.std(diff, ddof=1)
        cohen_d = float(np.mean(diff) / diff_std) if diff_std > 0 else 0.0

        if n > 2:
            rho = float(np.corrcoef(diff[:-1], diff[1:])[0, 1])
            if np.isnan(rho):
                rho = 0.0
        else:
            rho = 0.0
        rho = max(min(rho, 0.99), -0.99)
        n_eff = n * (1 - rho) / (1 + rho)
        n_eff = min(n_eff, n)

        boot = bootstrap_confidence_interval_block(
            diff, n_bootstrap=n_bootstrap, ci=95, block_size=block_size)

        effect_size = (
            'Büyük' if abs(cohen_d) > 0.8 else
            'Orta' if abs(cohen_d) > 0.5 else
            'Küçük' if abs(cohen_d) > 0.2 else 'İhmal Edilebilir')

        print(f"  n_cycles: {n}, n_eff (AR(1)): {n_eff:.2f}")
        print(f"  rho (lag-1): {rho:.4f}")
        print(f"  Wilcoxon p (exploratory): {p_raw:.6e}")
        print(f"  Cohen's d_z: {cohen_d:.4f} ({effect_size})")
        print(f"  Block-bootstrap %95 GA: [{boot['ci_lower']:.6f}, {boot['ci_upper']:.6f}]")

        return {
            'metric_unit': 'cycle-level',
            'n_cycles': int(n),
            'n_effective_ar1': float(n_eff),
            'autocorr_lag1': float(rho),
            'p_wilcoxon_raw_exploratory': float(p_raw),
            'cohen_d_z': cohen_d,
            'effect_size': effect_size,
            'ci_95_lower': float(boot['ci_lower']),
            'ci_95_upper': float(boot['ci_upper']),
            'bootstrap_n': boot['n_bootstrap'],
            'bootstrap_method': boot['method'],
            'bootstrap_block_size': boot['block_size'],
            'mean_std_error': float(np.mean(std_errors)),
            'mean_rec_error': float(np.mean(rec_errors))}
    except Exception as e:
        print(f"  Wilcoxon hatası: {e}")
        return {'error': str(e)}


def ablation_cycle_level_wilcoxon_lfp(y_true_li, li_pred, lf_pred,
                                       block_size=BLOCK_SIZE_DEFAULT,
                                       n_bootstrap=N_BOOTSTRAP_DEFAULT):
    n = min(len(y_true_li), len(li_pred), len(lf_pred))
    y_true_li = np.asarray(y_true_li[:n])
    li_pred = np.asarray(li_pred[:n])
    lf_pred = np.asarray(lf_pred[:n])

    li_errors = np.abs(y_true_li - li_pred)
    lf_errors = np.abs(y_true_li - lf_pred)
    diff = li_errors - lf_errors

    try:
        w_stat, p_raw = wilcoxon(li_errors, lf_errors)
    except Exception:
        w_stat, p_raw = np.nan, np.nan

    diff_std = np.std(diff, ddof=1)
    cohen_d = float(np.mean(diff) / diff_std) if diff_std > 0 else 0.0

    effect_size = (
        'Büyük' if abs(cohen_d) > 0.8 else
        'Orta' if abs(cohen_d) > 0.5 else
        'Küçük' if abs(cohen_d) > 0.2 else 'İhmal Edilebilir')

    boot = bootstrap_confidence_interval_block(
        diff, n_bootstrap=n_bootstrap, ci=95, block_size=block_size)

    return {
        'n_cycles': int(n),
        'p_wilcoxon_exploratory': float(p_raw) if not np.isnan(p_raw) else np.nan,
        'cohen_d_z': cohen_d,
        'effect_size': effect_size,
        'ci_95_lower': float(boot['ci_lower']),
        'ci_95_upper': float(boot['ci_upper']),
        'mean_lag_li_error': float(np.mean(li_errors)),
        'mean_lag_free_error': float(np.mean(lf_errors)),
        'delta_error': float(np.mean(li_errors) - np.mean(lf_errors)),
        'note': 'n=1 batarya oldugu icin kesifsel (exploratory).'}


def linear_mixed_model_analysis_lfp(all_predictions,
                                    model_name='XGBoost',
                                    block_size=BLOCK_SIZE_DEFAULT):
    print("\n" + "=" * 70)
    print(f"LMM (blok rastgele etki) - {model_name}")
    print("=" * 70)

    pred_data = None
    for key in ['xgb', 'lstm', 'gru']:
        if key in all_predictions:
            pred_data = all_predictions[key]
            break
    if pred_data is None:
        return {'error': 'Yetersiz veri'}

    n = min(len(pred_data['true']), len(pred_data['standard']),
            len(pred_data['recursive']))
    y_true = np.asarray(pred_data['true'][:n])
    std_pred = np.asarray(pred_data['standard'][:n])
    rec_pred = np.asarray(pred_data['recursive'][:n])

    records = []
    for i in range(n):
        block_id = i // block_size
        records.append({'cycle': i, 'block': block_id,
                        'is_recursive': 0,
                        'abs_error': abs(y_true[i] - std_pred[i])})
        records.append({'cycle': i, 'block': block_id,
                        'is_recursive': 1,
                        'abs_error': abs(y_true[i] - rec_pred[i])})
    df_lmm = pd.DataFrame(records)

    try:
        md = smf.mixedlm("abs_error ~ is_recursive",
                         data=df_lmm, groups=df_lmm["block"],
                         re_formula="~1")
        mdf = md.fit(reml=True, method='lbfgs', maxiter=200)

        coef_key = [k for k in mdf.params.index if 'is_recursive' in k]
        if coef_key:
            coef = float(mdf.params[coef_key[0]])
            se = float(mdf.bse[coef_key[0]])
            pval = float(mdf.pvalues[coef_key[0]])
        else:
            coef, se, pval = np.nan, np.nan, np.nan

        print(f"  beta1: {coef:.8f} (SE: {se:.8f}), p={pval:.6f}")
        print(f"  Yakınsadı: {mdf.converged}")

        return {'coef_recursive': coef, 'se_recursive': se, 'p_value': pval,
                'converged': bool(mdf.converged),
                'n_cycles': int(n),
                'block_size': block_size,
                'significant': bool(pval < 0.05) if not np.isnan(pval) else False}
    except Exception as e:
        print(f"  LMM hatası: {e}")
        return {'error': str(e), 'converged': False}


def comprehensive_statistical_report_lfp(results_df, all_predictions,
                                         ablation_lag_li=None,
                                         ablation_lag_free=None,
                                         model_names=['XGBoost', 'LSTM', 'GRU']):
    report = {}
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: LFP (CH17)")
    print("=" * 70)

    model_map = {
        'XGBoost': ('xgb_standard_RMSE', 'xgb_recursive_RMSE',
                    'xgb_standard_MAE', 'xgb_recursive_MAE'),
        'LSTM': ('lstm_standard_RMSE', 'lstm_recursive_RMSE',
                 'lstm_standard_MAE', 'lstm_recursive_MAE'),
        'GRU': ('gru_standard_RMSE', 'gru_recursive_RMSE',
                'gru_standard_MAE', 'gru_recursive_MAE')}

    for model in model_names:
        if model in model_map:
            std_rmse, rec_rmse, std_mae, rec_mae = model_map[model]
            std_val = results_df[std_rmse].values[0]
            rec_val = results_df[rec_rmse].values[0]
            print(f"\n{model}: Std RMSE={std_val:.6f}, Rec RMSE={rec_val:.6f}, "
                  f"DeltaRMSE={rec_val - std_val:.6f}")
            report[f'{model}_RMSE_diff'] = {
                'std': float(std_val), 'rec': float(rec_val),
                'diff': float(rec_val - std_val)}

    print("\n" + "=" * 70)
    print("CYCLE-LEVEL WILCOXON")
    print("=" * 70)
    for model_key, model_name in [('xgb', 'XGBoost'), ('lstm', 'LSTM'), ('gru', 'GRU')]:
        if model_key in all_predictions:
            pred_data = all_predictions[model_key]
            if all(k in pred_data for k in ['true', 'standard', 'recursive']):
                print(f"\n{model_name}:")
                test_result = cycle_level_wilcoxon_lfp(
                    pred_data['true'], pred_data['standard'], pred_data['recursive'])
                report[f'{model_name}_cycle_level_wilcoxon'] = test_result

    lmm_results = {}
    for model_key, model_name in [('xgb', 'XGBoost'), ('lstm', 'LSTM'), ('gru', 'GRU')]:
        if model_key in all_predictions:
            pred_data = all_predictions[model_key]
            if all(k in pred_data for k in ['true', 'standard', 'recursive']):
                lmm_results[model_name] = linear_mixed_model_analysis_lfp(
                    {model_key: pred_data}, model_name=model_name)
    report['lmm_analysis'] = lmm_results

    if ablation_lag_li and ablation_lag_free:
        print("\n" + "=" * 70)
        print("LAG-LI vs LAG-SIZ ABLATION ISTATISTIKSEL TESTI (CH17)")
        print("=" * 70)
        ablation_stats = {}
        for model_name in ['XGBoost', 'LSTM', 'GRU']:
            li = ablation_lag_li.get(model_name, {})
            lf = ablation_lag_free.get(model_name, {})
            if li and lf and 'y_true' in li and 'predicted' in li and \
               'y_true' in lf and 'predicted' in lf:
                print(f"\n{model_name}:")
                stat = ablation_cycle_level_wilcoxon_lfp(
                    li['y_true'], li['predicted'], lf['predicted'])
                ablation_stats[model_name] = stat
                print(f"  n_cycles: {stat['n_cycles']}")
                print(f"  Wilcoxon p (exploratory): {stat['p_wilcoxon_exploratory']:.6e}")
                print(f"  Cohen's d_z: {stat['cohen_d_z']:.4f} ({stat['effect_size']})")
                print(f"  Block-bootstrap %95 GA: [{stat['ci_95_lower']:.6f}, {stat['ci_95_upper']:.6f}]")
                print(f"  Delta error (lag-li - lag-siz): {stat['delta_error']:.6f}")
        report['ablation_statistical_tests'] = ablation_stats

    report['friedman_nemenyi'] = {
        'applied': False,
        'reason': 'CH17 tek batarya + yuksek otokorelasyon'}

    persistence_val = results_df['persistence_RMSE'].values[0]
    xgb_std_val = results_df['xgb_standard_RMSE'].values[0]
    report['persistence_baseline'] = {
        'persistence_rmse': float(persistence_val),
        'xgb_standard_rmse': float(xgb_std_val)}

    return report


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main_pipeline_lfp(file_path):
    print("=" * 70)
    print("LFP (CH17) SOH PREDICTION - MAJOR FIX v5")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(file_path),
                              "soh_analysis_results_lfp_majorfix_v5")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    print(f"   Total rows: {len(df):,}")

    df_feat = extract_cycle_features_no_leakage_single(df)
    df_feat['battery_id'] = 'CH17'
    df_feat = df_feat.sort_values(['battery_id', 'cycle_id'])
    df_feat = create_rolling_features_no_leakage_single(df_feat, window=3)
    df_feat = add_degradation_features_adaptive_single(df_feat)

    exclude_cols = ['SOH', 'battery_id', 'cycle_id']
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols]

    n = len(df_feat)
    train_size = int(n * 0.7)
    train_df = df_feat.iloc[:train_size].copy()
    test_df = df_feat.iloc[train_size:].copy()

    train_df = add_lag_features_no_initial_gt_single(train_df).dropna()
    test_df_full = add_lag_features_no_initial_gt_single(test_df)

    lag_features = feature_cols + ['SOH_lag1', 'SOH_lag2']
    lag_features = [f for f in lag_features if f in train_df.columns]

    common_test_mask = test_df_full[['SOH_lag1', 'SOH_lag2']].notna().all(axis=1)
    test_df_common = test_df_full[common_test_mask].copy()
    print(f"   Common test cycles: {len(test_df_common)}")

    shap_results = {}
    ablation_shap_results = {}
    ablation_lag_li = {}
    ablation_lag_free = {}

    # XGBoost
    print("\n" + "=" * 50 + "\nXGBOOST (lag'li)\n" + "=" * 50)
    X_train = train_df[lag_features]
    y_train = train_df['SOH']

    model_xgb = XGBRegressor(**XGB_PARAMS)
    model_xgb.fit(X_train, y_train)

    X_test_common = test_df_common[lag_features]
    y_test_common = test_df_common['SOH']

    pred_xgb_std = model_xgb.predict(X_test_common)
    m_xgb_std = evaluate_metrics_only(y_test_common, pred_xgb_std)

    pred_xgb_rec = recursive_forecast_deployment_realistic_single(
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

    y_tp = test_df_common['SOH'].values[1:]
    y_pp = test_df_common['SOH'].shift(1).dropna().values
    m_pers = evaluate_metrics_only(y_tp, y_pp) if len(y_tp) > 0 else {
        'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}

    ablation_lag_li['XGBoost'] = {
        'standard_RMSE': m_xgb_std['RMSE'],
        'standard_MAE': m_xgb_std['MAE'],
        'recursive_RMSE': m_xgb_rec['RMSE'],
        'recursive_MAE': m_xgb_rec['MAE'],
        'y_true': y_test_common.values,
        'predicted': pred_xgb_std}

    try:
        ns = min(100, len(X_test_common))
        shap_results['CH17'] = physical_interpretation_analysis(
            model_xgb, lag_features, X_test_common.iloc[:ns].values)
    except Exception as e:
        print(f"   SHAP hatası: {e}")

    print("\n" + "=" * 50 + "\nXGBOOST (lag'siz)\n" + "=" * 50)
    lag_free_out = run_lag_free_ablation_xgboost_single(
        train_df, test_df_full, feature_cols,
        common_test_mask=common_test_mask)
    if lag_free_out:
        m_xgb_lf = lag_free_out['metrics']
        print(f"   XGB Lag-siz Std -> RMSE: {m_xgb_lf['RMSE']:.6f}")
        ablation_lag_free['XGBoost'] = {
            'standard_RMSE': m_xgb_lf['RMSE'],
            'standard_MAE': m_xgb_lf['MAE'],
            'y_true': lag_free_out['y_true'],
            'predicted': lag_free_out['predictions']}

        try:
            X_test_lf = test_df_common[lag_free_out['features']].iloc[:100].values
            ablation_shap_results['CH17'] = shap_lag_free_analysis(
                lag_free_out['model'], lag_free_out['features'], X_test_lf)
        except Exception as e:
            print(f"   SHAP (lag-free) hatası: {e}")

    # LSTM
    lstm_out = None
    horizon_lstm_std = {'absolute_error_temporal_slope_beta1': np.nan}
    horizon_lstm_rec = {'absolute_error_temporal_slope_beta1': np.nan}
    if TF_AVAILABLE:
        print("\n" + "=" * 50 + "\nLSTM (lag'li)\n" + "=" * 50)
        try:
            lstm_out = run_deep_learning_supplementary(
                train_df, test_df_full, lag_features, 'LSTM', DL_LOOKBACK)
            if lstm_out:
                m_lstm_std = lstm_out['standard']
                m_lstm_rec = lstm_out['recursive']
                print(f"   LSTM Std -> RMSE: {m_lstm_std['RMSE']:.6f}, "
                      f"Rec -> RMSE: {m_lstm_rec['RMSE']:.6f}")
                ablation_lag_li['LSTM'] = {
                    'standard_RMSE': m_lstm_std['RMSE'],
                    'standard_MAE': m_lstm_std['MAE'],
                    'recursive_RMSE': m_lstm_rec['RMSE'],
                    'recursive_MAE': m_lstm_rec['MAE'],
                    'y_true': lstm_out['standard_true'],
                    'predicted': lstm_out['standard_pred']}

                horizon_lstm_std = evaluate_horizon_performance_metrics_only(
                    lstm_out['standard_true'], lstm_out['standard_pred'])
                horizon_lstm_rec = evaluate_horizon_performance_metrics_only(
                    lstm_out['recursive_true'], lstm_out['recursive_pred'])
                print(f"   LSTM beta1 Std: {horizon_lstm_std.get('absolute_error_temporal_slope_beta1', np.nan):.6f}, "
                      f"Rec: {horizon_lstm_rec.get('absolute_error_temporal_slope_beta1', np.nan):.6f}")
        except Exception as e:
            print(f"   LSTM failed: {e}")
            m_lstm_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
            m_lstm_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
    else:
        m_lstm_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
        m_lstm_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}

    if TF_AVAILABLE:
        try:
            lstm_lf = run_lag_free_ablation_deep_single(
                train_df, test_df_full, feature_cols, 'LSTM',
                common_test_mask=common_test_mask)
            if lstm_lf:
                ablation_lag_free['LSTM'] = {
                    'standard_RMSE': lstm_lf['metrics']['RMSE'],
                    'standard_MAE': lstm_lf['metrics']['MAE'],
                    'y_true': lstm_lf['y_true'],
                    'predicted': lstm_lf['predictions']}
        except Exception as e:
            print(f"   LSTM lag-free failed: {e}")

    # GRU
    gru_out = None
    horizon_gru_std = {'absolute_error_temporal_slope_beta1': np.nan}
    horizon_gru_rec = {'absolute_error_temporal_slope_beta1': np.nan}
    if TF_AVAILABLE:
        print("\n" + "=" * 50 + "\nGRU (lag'li)\n" + "=" * 50)
        try:
            gru_out = run_deep_learning_supplementary(
                train_df, test_df_full, lag_features, 'GRU', DL_LOOKBACK)
            if gru_out:
                m_gru_std = gru_out['standard']
                m_gru_rec = gru_out['recursive']
                print(f"   GRU Std -> RMSE: {m_gru_std['RMSE']:.6f}, "
                      f"Rec -> RMSE: {m_gru_rec['RMSE']:.6f}")
                ablation_lag_li['GRU'] = {
                    'standard_RMSE': m_gru_std['RMSE'],
                    'standard_MAE': m_gru_std['MAE'],
                    'recursive_RMSE': m_gru_rec['RMSE'],
                    'recursive_MAE': m_gru_rec['MAE'],
                    'y_true': gru_out['standard_true'],
                    'predicted': gru_out['standard_pred']}

                horizon_gru_std = evaluate_horizon_performance_metrics_only(
                    gru_out['standard_true'], gru_out['standard_pred'])
                horizon_gru_rec = evaluate_horizon_performance_metrics_only(
                    gru_out['recursive_true'], gru_out['recursive_pred'])
                print(f"   GRU beta1 Std: {horizon_gru_std.get('absolute_error_temporal_slope_beta1', np.nan):.6f}, "
                      f"Rec: {horizon_gru_rec.get('absolute_error_temporal_slope_beta1', np.nan):.6f}")
        except Exception as e:
            print(f"   GRU failed: {e}")
            m_gru_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
            m_gru_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
    else:
        m_gru_std = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
        m_gru_rec = {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}

    if TF_AVAILABLE:
        try:
            gru_lf = run_lag_free_ablation_deep_single(
                train_df, test_df_full, feature_cols, 'GRU',
                common_test_mask=common_test_mask)
            if gru_lf:
                ablation_lag_free['GRU'] = {
                    'standard_RMSE': gru_lf['metrics']['RMSE'],
                    'standard_MAE': gru_lf['metrics']['MAE'],
                    'y_true': gru_lf['y_true'],
                    'predicted': gru_lf['predictions']}
        except Exception as e:
            print(f"   GRU lag-free failed: {e}")

    # Sonuçlar
    results_df = pd.DataFrame({
        'battery': ['CH17'],
        'xgb_standard_RMSE': [m_xgb_std['RMSE']],
        'xgb_recursive_RMSE': [m_xgb_rec['RMSE']],
        'xgb_standard_MAE': [m_xgb_std['MAE']],
        'xgb_recursive_MAE': [m_xgb_rec['MAE']],
        'xgb_beta1_std': [horizon_xgb_std.get('absolute_error_temporal_slope_beta1', np.nan)],
        'xgb_beta1_rec': [horizon_xgb_rec.get('absolute_error_temporal_slope_beta1', np.nan)],
        'xgb_horizon_std_RMSE_h1': [horizon_xgb_std.get('horizon_1_RMSE', np.nan)],
        'xgb_horizon_rec_RMSE_h1': [horizon_xgb_rec.get('horizon_1_RMSE', np.nan)],
        'xgb_horizon_std_RMSE_h5': [horizon_xgb_std.get('horizon_5_RMSE', np.nan)],
        'xgb_horizon_rec_RMSE_h5': [horizon_xgb_rec.get('horizon_5_RMSE', np.nan)],
        'xgb_horizon_std_RMSE_h10': [horizon_xgb_std.get('horizon_10_RMSE', np.nan)],
        'xgb_horizon_rec_RMSE_h10': [horizon_xgb_rec.get('horizon_10_RMSE', np.nan)],
        'xgb_horizon_std_RMSE_h20': [horizon_xgb_std.get('horizon_20_RMSE', np.nan)],
        'xgb_horizon_rec_RMSE_h20': [horizon_xgb_rec.get('horizon_20_RMSE', np.nan)],
        'xgb_horizon_std_RMSE_h50': [horizon_xgb_std.get('horizon_50_RMSE', np.nan)],
        'xgb_horizon_rec_RMSE_h50': [horizon_xgb_rec.get('horizon_50_RMSE', np.nan)],
        'lstm_standard_RMSE': [m_lstm_std['RMSE']],
        'lstm_recursive_RMSE': [m_lstm_rec['RMSE']],
        'lstm_standard_MAE': [m_lstm_std['MAE']],
        'lstm_recursive_MAE': [m_lstm_rec['MAE']],
        'lstm_beta1_std': [horizon_lstm_std.get('absolute_error_temporal_slope_beta1', np.nan)],
        'lstm_beta1_rec': [horizon_lstm_rec.get('absolute_error_temporal_slope_beta1', np.nan)],
        'lstm_horizon_std_RMSE_h1': [horizon_lstm_std.get('horizon_1_RMSE', np.nan)],
        'lstm_horizon_rec_RMSE_h1': [horizon_lstm_rec.get('horizon_1_RMSE', np.nan)],
        'lstm_horizon_std_RMSE_h5': [horizon_lstm_std.get('horizon_5_RMSE', np.nan)],
        'lstm_horizon_rec_RMSE_h5': [horizon_lstm_rec.get('horizon_5_RMSE', np.nan)],
        'lstm_horizon_std_RMSE_h10': [horizon_lstm_std.get('horizon_10_RMSE', np.nan)],
        'lstm_horizon_rec_RMSE_h10': [horizon_lstm_rec.get('horizon_10_RMSE', np.nan)],
        'lstm_horizon_std_RMSE_h20': [horizon_lstm_std.get('horizon_20_RMSE', np.nan)],
        'lstm_horizon_rec_RMSE_h20': [horizon_lstm_rec.get('horizon_20_RMSE', np.nan)],
        'lstm_horizon_std_RMSE_h50': [horizon_lstm_std.get('horizon_50_RMSE', np.nan)],
        'lstm_horizon_rec_RMSE_h50': [horizon_lstm_rec.get('horizon_50_RMSE', np.nan)],
        'gru_standard_RMSE': [m_gru_std['RMSE']],
        'gru_recursive_RMSE': [m_gru_rec['RMSE']],
        'gru_standard_MAE': [m_gru_std['MAE']],
        'gru_recursive_MAE': [m_gru_rec['MAE']],
        'gru_beta1_std': [horizon_gru_std.get('absolute_error_temporal_slope_beta1', np.nan)],
        'gru_beta1_rec': [horizon_gru_rec.get('absolute_error_temporal_slope_beta1', np.nan)],
        'gru_horizon_std_RMSE_h1': [horizon_gru_std.get('horizon_1_RMSE', np.nan)],
        'gru_horizon_rec_RMSE_h1': [horizon_gru_rec.get('horizon_1_RMSE', np.nan)],
        'gru_horizon_std_RMSE_h5': [horizon_gru_std.get('horizon_5_RMSE', np.nan)],
        'gru_horizon_rec_RMSE_h5': [horizon_gru_rec.get('horizon_5_RMSE', np.nan)],
        'gru_horizon_std_RMSE_h10': [horizon_gru_std.get('horizon_10_RMSE', np.nan)],
        'gru_horizon_rec_RMSE_h10': [horizon_gru_rec.get('horizon_10_RMSE', np.nan)],
        'gru_horizon_std_RMSE_h20': [horizon_gru_std.get('horizon_20_RMSE', np.nan)],
        'gru_horizon_rec_RMSE_h20': [horizon_gru_rec.get('horizon_20_RMSE', np.nan)],
        'gru_horizon_std_RMSE_h50': [horizon_gru_std.get('horizon_50_RMSE', np.nan)],
        'gru_horizon_rec_RMSE_h50': [horizon_gru_rec.get('horizon_50_RMSE', np.nan)],
        'persistence_RMSE': [m_pers['RMSE']],
        'persistence_MAE': [m_pers['MAE']]})

    all_predictions = {'xgb': {
        'true': test_df_common['SOH'].values,
        'recursive': pred_xgb_rec, 'standard': pred_xgb_std}}
    if lstm_out:
        all_predictions['lstm'] = {
            'true': lstm_out['recursive_true'],
            'recursive': lstm_out['recursive_pred'],
            'standard': lstm_out['standard_pred']}
    if gru_out:
        all_predictions['gru'] = {
            'true': gru_out['recursive_true'],
            'recursive': gru_out['recursive_pred'],
            'standard': gru_out['standard_pred']}

    results_df.to_csv(os.path.join(output_dir, 'results_lfp_v5.csv'), index=False)

    if shap_results:
        pd.DataFrame(shap_results).T.to_csv(
            os.path.join(output_dir, 'shap_lagli_lfp_v5.csv'))

    print("\n" + "=" * 70)
    print("LAG-FREE ABLATION ÖZETİ (CH17)")
    print("=" * 70)

    ablation_rows = []
    for model_name in ['XGBoost', 'LSTM', 'GRU']:
        li = ablation_lag_li.get(model_name, {})
        lf = ablation_lag_free.get(model_name, {})
        if li and lf:
            li_rmse = li.get('standard_RMSE', np.nan)
            lf_rmse = lf.get('standard_RMSE', np.nan)
            li_mae = li.get('standard_MAE', np.nan)
            lf_mae = lf.get('standard_MAE', np.nan)

            ablation_rows.append({
                'Model': model_name,
                'Lag-li Std RMSE': li_rmse,
                'Lag-li Recursive RMSE': li.get('recursive_RMSE', np.nan),
                'Lag-siz Std RMSE': lf_rmse,
                'DeltaRMSE': li_rmse - lf_rmse,
                'DeltaMAE': li_mae - lf_mae})

    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        print(ablation_df.to_string(index=False))
        ablation_df.to_csv(os.path.join(output_dir, 'ablation_lfp_v5.csv'), index=False)

    if ablation_shap_results:
        pd.DataFrame(ablation_shap_results).T.to_csv(
            os.path.join(output_dir, 'shap_lagsiz_lfp_v5.csv'))

    model_names = ['XGBoost', 'LSTM', 'GRU'] if TF_AVAILABLE else ['XGBoost']
    stat_report = comprehensive_statistical_report_lfp(
        results_df, all_predictions,
        ablation_lag_li=ablation_lag_li,
        ablation_lag_free=ablation_lag_free,
        model_names=model_names)

    with open(os.path.join(output_dir, 'stat_report_lfp_v5.json'),
              'w', encoding='utf-8') as f:
        json.dump(stat_report, f, indent=2, default=str, ensure_ascii=False)

    print("\nLFP analysis complete!")
    return results_df, all_predictions, stat_report


if __name__ == "__main__":
    lfp_file = r"C:\Users\LENOVO\Desktop\LiFePO4 Veri Setleri\LiFePO4_CH17\duzenlenmis_2018-04-12_batch8_CH17.csv"

    if os.path.exists(lfp_file):
        print("LFP veri dosyası bulundu")
        results_df, predictions, stat_report = main_pipeline_lfp(lfp_file)
        print("\nLFP ANALİZİ TAMAMLANDI!")
    else:
        print(f"LFP veri dosyası bulunamadı: {lfp_file}")
