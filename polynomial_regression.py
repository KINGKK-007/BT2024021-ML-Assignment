"""Polynomial regression assignment for BT2024021 (Kanav Kumar)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROLL_NO = "BT2024021"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
DATA_DIR = Path("Dataset")
FIGURE_DIR = Path("figures")
RESULTS_DIR = Path("results")

# Assignment limits: var1 <= 10 and var2 <= 20.
CONFIG = {
    "var1": {"max_degree": 10, "expected_features": 6},
    "var2": {"max_degree": 20, "expected_features": 3},
}


def make_model(degree: int, alpha: float) -> Pipeline:
    """Create a stable polynomial Ridge regression pipeline."""
    return Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                    solver="lsqr",
                    tol=1e-7,
                    max_iter=10_000,
                ),
            ),
        ]
    )


def inspect_dataset(name: str, data: pd.DataFrame) -> dict:
    """Print basic checks and return a compact dataset summary row."""
    print(f"\n{name}: shape={data.shape}")
    print(f"columns={list(data.columns)}")
    print(f"missing values={int(data.isna().sum().sum())}")
    print(f"duplicate rows={int(data.duplicated().sum())}")
    print(data.describe().round(4).to_string())
    return {
        "dataset": name,
        "rows": len(data),
        "columns": len(data.columns),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def validate_input(
    train: pd.DataFrame, test: pd.DataFrame, expected_features: int
) -> list[str]:
    """Check the input schema before modelling."""
    if "y" not in train.columns:
        raise ValueError("Training data must contain the target column 'y'.")
    if "y" in test.columns:
        raise ValueError("Test data unexpectedly contains 'y'; refusing possible leakage.")

    feature_columns = [column for column in train.columns if column != "y"]
    if len(feature_columns) != expected_features:
        raise ValueError(
            f"Expected {expected_features} input variables, found {len(feature_columns)}."
        )
    if list(test.columns) != feature_columns:
        raise ValueError("Training and test feature columns/order do not match.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing values found in input data.")
    if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
        raise ValueError("Non-finite values found in input data.")
    return feature_columns


def select_model(
    X: pd.DataFrame, y: pd.Series, max_degree: int
) -> tuple[pd.DataFrame, int, float]:
    """Choose degree and Ridge alpha using only a training/validation split."""
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    rows = []
    for degree in range(1, max_degree + 1):
        best_for_degree = None
        for alpha in ALPHAS:
            model = make_model(degree, alpha)
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)
            mse = mean_squared_error(y_validation, predictions)
            r2 = r2_score(y_validation, predictions)

            if best_for_degree is None or mse < best_for_degree["validation_mse"]:
                best_for_degree = {
                    "degree": degree,
                    "alpha": alpha,
                    "polynomial_features": model.named_steps[
                        "polynomial"
                    ].n_output_features_,
                    "validation_mse": mse,
                    "validation_r2": r2,
                }
        rows.append(best_for_degree)

    results = pd.DataFrame(rows)
    best_row = results.loc[results["validation_mse"].idxmin()]
    return results, int(best_row["degree"]), float(best_row["alpha"])


def save_selection_plot(variant: str, results: pd.DataFrame) -> None:
    """Save degree-versus-validation-score plot."""
    fig, axis_mse = plt.subplots(figsize=(8, 4.8))
    axis_r2 = axis_mse.twinx()

    axis_mse.plot(
        results["degree"],
        results["validation_mse"],
        marker="o",
        color="tab:blue",
        label="Validation MSE",
    )
    axis_r2.plot(
        results["degree"],
        results["validation_r2"],
        marker="s",
        color="tab:orange",
        label="Validation R²",
    )
    best = results.loc[results["validation_mse"].idxmin()]
    axis_mse.axvline(
        best["degree"],
        color="tab:green",
        linestyle="--",
        alpha=0.8,
        label=f"Selected degree = {int(best['degree'])}",
    )
    axis_mse.set_xlabel("Polynomial degree")
    # A log scale keeps the small differences between good high-degree models visible.
    axis_mse.set_yscale("log")
    axis_mse.set_ylabel("Validation MSE (log scale)", color="tab:blue")
    axis_r2.set_ylabel("Validation R²", color="tab:orange")
    axis_mse.set_xticks(results["degree"])
    axis_mse.grid(alpha=0.25)
    lines1, labels1 = axis_mse.get_legend_handles_labels()
    lines2, labels2 = axis_r2.get_legend_handles_labels()
    axis_mse.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(f"{variant}: polynomial degree selection")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{variant}_model_selection.png", dpi=200)
    plt.close(fig)


def run_variant(variant: str, summary_rows: list[dict]) -> dict:
    """Select, retrain, predict, save, and verify one dataset variant."""
    settings = CONFIG[variant]
    train_path = DATA_DIR / f"{ROLL_NO}_train_{variant}.csv"
    test_path = DATA_DIR / f"{ROLL_NO}_test_{variant}.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    summary_rows.append(inspect_dataset(train_path.name, train))
    summary_rows.append(inspect_dataset(test_path.name, test))
    feature_columns = validate_input(train, test, settings["expected_features"])
    X = train[feature_columns]
    y = train["y"]

    results, best_degree, best_alpha = select_model(
        X, y, settings["max_degree"]
    )
    if best_degree > settings["max_degree"]:
        raise AssertionError("Selected degree exceeds the assignment limit.")

    results.to_csv(RESULTS_DIR / f"{variant}_validation_results.csv", index=False)
    save_selection_plot(variant, results)

    # Refit only after model selection, now using every labelled training row.
    final_model = make_model(best_degree, best_alpha)
    final_model.fit(X, y)
    test_predictions = final_model.predict(test[feature_columns])

    if len(test_predictions) != len(test):
        raise AssertionError("Prediction row count does not match the test data.")
    if not np.isfinite(test_predictions).all():
        raise AssertionError("Predictions contain NaN or infinity.")

    prediction_path = Path(f"{ROLL_NO}_pred_{variant}.csv")
    pd.DataFrame({"y": test_predictions}).to_csv(prediction_path, index=False)

    # Read the saved file back to test the actual deliverable.
    saved = pd.read_csv(prediction_path)
    if list(saved.columns) != ["y"]:
        raise AssertionError("Prediction CSV must contain exactly one column named 'y'.")
    if len(saved) != len(test) or not np.isfinite(saved["y"]).all():
        raise AssertionError("Saved prediction CSV failed validation.")

    chosen = results.loc[results["degree"] == best_degree].iloc[0]
    print("\nValidation results:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.8f}"))
    print(
        f"Selected {variant}: degree={best_degree}, alpha={best_alpha:g}, "
        f"MSE={chosen['validation_mse']:.8f}, R2={chosen['validation_r2']:.8f}"
    )
    print(f"Saved {prediction_path} ({len(saved)} predictions)")
    return {
        "variant": variant,
        "degree": best_degree,
        "alpha": best_alpha,
        "validation_mse": float(chosen["validation_mse"]),
        "validation_r2": float(chosen["validation_r2"]),
    }


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if list(sample.columns) != ["y"]:
        raise ValueError("The sample submission format is not a single 'y' column.")

    summary_rows = []
    selected_models = []
    for variant in CONFIG:
        selected_models.append(run_variant(variant, summary_rows))

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    pd.DataFrame(selected_models).to_csv(
        RESULTS_DIR / "selected_models.csv", index=False
    )
    print("\nAll input, model-limit, prediction, row-count, and CSV-format checks passed.")


if __name__ == "__main__":
    main()
