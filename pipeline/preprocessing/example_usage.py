"""Lower-level scalar preprocessing examples, separate from main.py's 12 targets.

Executing this file loads data with the ingestion defaults. For the supported
multi-horizon experiment, use main.py and docs/experiments.md instead.
"""

from pipeline.preprocessing import (
    NumericPreprocessor,
    make_sequences,
    prepare_model_data,
    suggested_scaling_groups,
)

# ============================================================
# 1. CLASSICAL / TREE MODEL
# ============================================================

# One call does loading, train/test feature construction, history handling,
# region filtering, target construction and feature selection.
data = prepare_model_data(
    region="NSW1",
    horizon=60,
    features="all",
    target="change",
)

print("Train:", data.X_train.shape)
print("Test:", data.X_test.shape)
print("Features:", len(data.features))

# Example XGBoost / LightGBM / Random Forest usage:
#
# model.fit(data.X_train, data.y_train)
# predicted_change = model.predict(data.X_test)
# predicted_rrp = data.test_df["rrp"].to_numpy() + predicted_change


# ============================================================
# 2. CUSTOM FEATURE EXPERIMENT
# ============================================================

small_data = prepare_model_data(
    region="NSW1",
    horizon=60,
    features=[
        "rrp",
        "rrp_delta_1",
        "rrp_delta_3",
        "rrp_roll_std_6",
        "reserve_ratio",
        "p5_h60_rrp_vs_now",
        "p5_h60_demand_vs_now",
    ],
    target="change",
)

# model.fit(small_data.X_train, small_data.y_train)


# ============================================================
# 3. LINEAR / NEURAL MODEL WITH SCALING
# ============================================================

# Reuse exactly the same prepared data. Fit transforms on TRAIN only.
groups = suggested_scaling_groups(data.features)

scaler = NumericPreprocessor(
    asinh_columns=tuple(groups["asinh_then_standardise"]),
    standardise_columns=tuple(groups["standardise_only"]),
)

X_train_scaled = scaler.fit_transform(data.X_train)
X_test_scaled = scaler.transform(data.X_test)

# linear_or_neural_model.fit(X_train_scaled, data.y_train)


# ============================================================
# 4. LSTM / GRU / TCN / TRANSFORMER
# ============================================================

# The returned full frames contain the same selected features plus a standard
# ``target`` column, so sequence models do not need another preprocessing path.
train_sequences = make_sequences(
    data.train_timeline,
    feature_columns=data.features,
    target_columns=["target"],
    sequence_length=288,
)

test_sequences = make_sequences(
    data.test_timeline,
    feature_columns=data.features,
    target_columns=["target"],
    sequence_length=288,
    history=data.train_timeline,
)

print("Sequence train X:", train_sequences.X.shape)
print("Sequence train y:", train_sequences.y.shape)
