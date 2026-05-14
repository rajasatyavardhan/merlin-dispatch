# ai/time_predictor.py
# MERLIN — Medical Emergency Routing and Live Intelligence Network
# Response Time Predictor — Random Forest regression model
#
# Predicts T_travel: time from dispatch to patient arrival.
#
# Why Random Forest and not XGBoost?
# - Regression tasks with moderate data → Random Forest
#   is more stable and less prone to overfitting
# - XGBoost excels at classification with large datasets
# - Both are valid — using different models for different
#   tasks demonstrates ML breadth in your paper

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ── Configuration ────────────────────────────────────────────────
TIME_MODEL_SAVE_PATH = 'data/synthetic/time_predictor.pkl'
RESULTS_CSV_PATH     = 'data/synthetic/simulation_results.csv'

# Features for response time prediction
# Different from severity model — focuses on
# operational variables, not clinical ones
TIME_FEATURES = [
    'distance_to_vehicle',   # primary predictor
    'terrain',               # affects road speed
    'weather',               # affects helicopter speed
    'has_landing_zone',      # affects helicopter choice
    'population_density',    # proxy for road quality
    'distance_to_hospital',  # total journey context
    'season',                # winter = slower roads
    'hour_of_day',           # traffic patterns
]

# Random Forest configuration
RF_PARAMS = {
    'n_estimators':  200,
    'max_depth':     8,
    'min_samples_split': 5,
    'min_samples_leaf':  2,
    'random_state':  42,
    'n_jobs':        -1,    # use all CPU cores
}

def load_time_training_data(csv_path=RESULTS_CSV_PATH):
    """
    Loads simulation data for response time prediction.

    Target variable (Y): response_time in minutes
    Features (X): operational variables

    Uses ALL system decisions — not just MERLIN —
    because response time is a physical calculation
    that applies regardless of dispatch strategy.
    This gives us 4× more training data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No data at {csv_path}. "
            f"Run simulation/runner.py first."
        )

    df = pd.read_csv(csv_path)

    # Filter out no-vehicle cases
    df = df[df['response_time'] > 0].copy()

    print(f"Total records loaded:     {len(df)}")
    print(f"After filtering zeros:    {len(df)}")

    # Engineer missing features if needed
    if 'season' not in df.columns:
        df['season'] = (df['tick'] // 131400) % 4
    if 'hour_of_day' not in df.columns:
        df['hour_of_day'] = (df['tick'] % 1440) // 60

    # Fill missing features
    for feat in TIME_FEATURES:
        if feat not in df.columns:
            df[feat] = 0

    X = df[TIME_FEATURES].values
    y = df['response_time'].values

    print(f"Response time stats:")
    print(f"  Min:    {y.min():.1f} min")
    print(f"  Max:    {y.max():.1f} min")
    print(f"  Mean:   {y.mean():.1f} min")
    print(f"  Median: {np.median(y):.1f} min")
    print()

    return X, y, df

def train_time_model(X, y, test_size=0.2,
                     random_state=42, verbose=True):
    """
    Trains Random Forest regressor to predict response time.

    Key metrics for regression:
    - MAE (Mean Absolute Error): average error in minutes
      Most interpretable — "model is off by X minutes on average"
    - RMSE (Root Mean Squared Error): penalises large errors more
    - R² (R-squared): how much variance the model explains
      1.0 = perfect, 0.0 = no better than predicting the mean

    For a dispatch system, MAE is the most important metric.
    A MAE of 2-3 minutes is clinically acceptable.

    Returns: (model, train_metrics, test_metrics)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state
    )

    if verbose:
        print("Training Random Forest Response Time Predictor")
        print("-" * 50)
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples:     {len(X_test)}")
        print()

    # ── Cross-validation ─────────────────────────────────────────
    cv_model = RandomForestRegressor(**RF_PARAMS)
    cv_scores = cross_val_score(
        cv_model, X, y, cv=5,
        scoring='neg_mean_absolute_error'
    )
    cv_mae = -cv_scores

    if verbose:
        print("5-Fold Cross-Validation MAE:")
        for i, score in enumerate(cv_mae):
            print(f"  Fold {i+1}: {score:.2f} min")
        print(f"  Mean:   {cv_mae.mean():.2f} min "
              f"(±{cv_mae.std():.2f})")
        print()

    # ── Train final model ─────────────────────────────────────────
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_mae  = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2   = r2_score(y_train, y_train_pred)

    test_mae  = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2   = r2_score(y_test, y_test_pred)

    if verbose:
        print("Training Results:")
        print(f"  MAE:  {train_mae:.2f} min")
        print(f"  RMSE: {train_rmse:.2f} min")
        print(f"  R²:   {train_r2:.4f}")
        print()
        print("Test Results (unseen data):")
        print(f"  MAE:  {test_mae:.2f} min")
        print(f"  RMSE: {test_rmse:.2f} min")
        print(f"  R²:   {test_r2:.4f}")
        print()

        # Overfitting check
        mae_gap = test_mae - train_mae
        if mae_gap > 2.0:
            print(f"⚠ Possible overfitting: MAE gap = {mae_gap:.2f}")
        else:
            print(f"✓ No overfitting: MAE gap = {mae_gap:.2f} min")
        print()

        # Clinical interpretation
        print("Clinical Interpretation:")
        if test_mae < 2.0:
            print(f"  ✓ Excellent — model error < 2 minutes")
        elif test_mae < 4.0:
            print(f"  ✓ Acceptable — model error < 4 minutes")
        elif test_mae < 6.0:
            print(f"  ⚠ Moderate — error may affect dispatch quality")
        else:
            print(f"  ✗ Poor — error too large for clinical use")
        print()

        # Feature importance
        importances = model.feature_importances_
        fi = sorted(
            zip(TIME_FEATURES, importances),
            key=lambda x: x[1], reverse=True
        )
        print("Feature Importance:")
        print("-" * 50)
        for feat, imp in fi:
            bar = "█" * int(imp * 40)
            print(f"  {feat:<25} {imp:.4f}  {bar}")
        print()

    train_metrics = {
        'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2
    }
    test_metrics = {
        'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2,
        'cv_mae_mean': cv_mae.mean(), 'cv_mae_std': cv_mae.std()
    }

    return model, train_metrics, test_metrics

def save_time_model(model, path=TIME_MODEL_SAVE_PATH):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Time model saved to: {path}")


def load_time_model(path=TIME_MODEL_SAVE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"Time model loaded from: {path}")
    return model


def predict_response_time(model, distance_km, terrain,
                           weather, has_landing_zone,
                           population_density,
                           distance_to_hospital,
                           season, hour_of_day):
    """
    Predicts response time for one dispatch scenario.

    Called by dispatch engine to estimate T_travel
    before making the helicopter vs ambulance decision.

    Returns: predicted response time in minutes (float)
    """
    features = np.array([[
        distance_km,
        terrain,
        weather,
        int(has_landing_zone),
        population_density,
        distance_to_hospital,
        season,
        hour_of_day
    ]])

    return float(model.predict(features)[0])


def run_time_training_pipeline(
        csv_path=RESULTS_CSV_PATH,
        save_path=TIME_MODEL_SAVE_PATH):
    """
    Full training pipeline for response time predictor.
    """
    print("=" * 50)
    print("MERLIN Response Time Predictor — Training Pipeline")
    print("=" * 50)
    print()

    # Load data
    X, y, df = load_time_training_data(csv_path)

    # Train
    model, train_m, test_m = train_time_model(
        X, y, verbose=True
    )

    # Save
    save_time_model(model, save_path)
    print()

    # Test predictions
    print("Sample Predictions:")
    print("-" * 50)

    scenarios = [
        # (dist, terrain, weather, landing, density, hosp_dist, season, hour)
        (5,   0, 0, 1, 0.8, 10,  1, 14, "Town, 5km, clear summer"),
        (80,  0, 0, 1, 0.1, 80,  3,  2, "Remote, 80km, clear winter"),
        (80,  0, 2, 1, 0.1, 80,  3,  2, "Remote, 80km, storm winter"),
        (30,  2, 1, 0, 0.0, 45,  0, 10, "Forest, 30km, cloudy spring"),
        (120, 0, 0, 1, 0.0, 120, 3, 22, "Far remote, 120km, winter night"),
    ]

    for dist, terr, wthr, lz, dens, hosp, ssn, hr, desc in scenarios:
        pred = predict_response_time(
            model, dist, terr, wthr, lz, dens, hosp, ssn, hr
        )
        print(f"  {desc}")
        print(f"  → Predicted response: {pred:.1f} minutes")
        print()

    print("=" * 50)
    print("Time predictor training complete.")
    print("=" * 50)

    return model, test_m