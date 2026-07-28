from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "features_test.csv"
RESULTS_ROOT = ROOT / "results" / "MNISTInputPassThroughTest"
INFERENCE_PATHS = [
    RESULTS_ROOT / "run_20260720_193925" / "mnist_inference_results.csv",
    RESULTS_ROOT / "run_20260720_211008" / "mnist_inference_results.csv",
]
OUTPUT_DIR = RESULTS_ROOT / "balanced_50_per_class_selection"
TARGET_PER_CLASS = 50
FIRST_UNUSED_SAMPLE_INDEX = 200


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    existing = pd.concat(
        [pd.read_csv(path) for path in INFERENCE_PATHS], ignore_index=True
    )

    if len(existing) != 200:
        raise ValueError(f"Expected 200 existing inference rows, found {len(existing)}")
    if existing["sample_index"].duplicated().any():
        raise ValueError("Existing inference results contain duplicate sample_index values")

    classes = sorted(existing["label"].astype(int).unique())
    existing_counts = existing["label"].astype(int).value_counts().reindex(classes, fill_value=0)
    required_counts = TARGET_PER_CLASS - existing_counts
    if (required_counts < 0).any():
        raise ValueError("At least one class already exceeds the requested target")

    candidates = features.loc[
        features["sample_index"].astype(int) >= FIRST_UNUSED_SAMPLE_INDEX
    ].copy()
    selected_parts = []
    for digit in classes:
        required = int(required_counts.loc[digit])
        matches = candidates.loc[candidates["label"].astype(int) == digit].head(required)
        if len(matches) != required:
            raise ValueError(
                f"Class {digit} needs {required} rows, but only {len(matches)} are available"
            )
        selected_parts.append(matches)

    continuation = pd.concat(selected_parts).sort_values("sample_index").reset_index(drop=True)
    existing_features = features.loc[
        features["sample_index"].isin(existing["sample_index"])
    ].copy()
    balanced = pd.concat([existing_features, continuation], ignore_index=True)
    balanced = balanced.sort_values("sample_index").reset_index(drop=True)

    final_counts = balanced["label"].astype(int).value_counts().reindex(classes, fill_value=0)
    if not (final_counts == TARGET_PER_CLASS).all():
        raise RuntimeError(f"Balanced count check failed: {final_counts.to_dict()}")

    summary = pd.DataFrame(
        {
            "digit": classes,
            "existing_count": existing_counts.values,
            "additional_count": required_counts.values,
            "final_count": final_counts.values,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    continuation.to_csv(OUTPUT_DIR / "features_continuation_200_balanced.csv", index=False)
    balanced.to_csv(OUTPUT_DIR / "features_balanced_400.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "selection_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Selected {len(continuation)} continuation samples")
    print(
        "Continuation sample_index range: "
        f"{int(continuation['sample_index'].min())}..{int(continuation['sample_index'].max())}"
    )
    print(f"Saved files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
