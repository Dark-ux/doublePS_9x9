from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_FILES = [
    ROOT / "run_20260720_193925" / "mnist_inference_results.csv",
    ROOT / "run_20260720_211008" / "mnist_inference_results.csv",
]
OUTPUT_DIR = ROOT / "combined_200_samples_plots"
CLASSES = np.arange(8)


def load_results() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in INPUT_FILES]
    results = pd.concat(frames, ignore_index=True)
    if len(results) != 200:
        raise ValueError(f"Expected 200 samples, found {len(results)}")
    return results


def plot_confusion_matrix(results: pd.DataFrame) -> None:
    matrix = pd.crosstab(results["label"], results["predicted_class"])
    matrix = matrix.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    values = matrix.to_numpy()
    accuracy = (results["label"] == results["predicted_class"]).mean()

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    image = ax.imshow(values, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Sample count")

    threshold = values.max() / 2
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(
                col,
                row,
                str(values[row, col]),
                ha="center",
                va="center",
                color="white" if values[row, col] > threshold else "black",
                fontsize=11,
            )

    ax.set(
        xticks=CLASSES,
        yticks=CLASSES,
        xlabel="Predicted digit",
        ylabel="True digit",
        title=f"MNIST Confusion Matrix (200 samples, accuracy {accuracy:.1%})",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_200_samples.png", dpi=300)
    plt.close(fig)

    matrix.to_csv(OUTPUT_DIR / "confusion_matrix_200_samples.csv")


def plot_digit_five(results: pd.DataFrame, sample_index: int = 108) -> None:
    matches = results.loc[results["sample_index"] == sample_index]
    if len(matches) != 1:
        raise ValueError(f"Expected one sample with index {sample_index}, found {len(matches)}")
    sample = matches.iloc[0]
    if int(sample["label"]) != 5:
        raise ValueError(f"Sample {sample_index} has label {sample['label']}, not 5")

    scores = np.array([sample[f"output_power_norm_{digit}"] for digit in CLASSES])
    colors = ["#4C78A8"] * len(CLASSES)
    colors[5] = "#E45756"

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.bar(CLASSES, scores * 100, color=colors, width=0.72)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{score:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    predicted = int(sample["predicted_class"])
    ax.set(
        xticks=CLASSES,
        xlabel="Digit class",
        ylabel="Normalized output power (%)",
        title=f"Inference Scores for Sample {sample_index} (true: 5, predicted: {predicted})",
    )
    ax.set_ylim(0, max(55, scores.max() * 100 + 8))
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.5,
        -0.18,
        "Scores are normalized optical output powers, not calibrated softmax probabilities.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "digit_5_sample_108_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    plot_confusion_matrix(results)
    plot_digit_five(results)
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
