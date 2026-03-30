from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_ANALYSIS_DIR = ARTIFACTS_DIR / "dataset_analysis"
RAW_INFERENCE_DIR = ARTIFACTS_DIR / "raw_inference"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TEST_INFERENCE_DIR = OUTPUTS_DIR / "inference_result"
RAW_OVERLAY_DIR = OUTPUTS_DIR / "final_inference_with_size"
PROJECT_PATHS = {
    "artifacts": ARTIFACTS_DIR,
    "dataset_analysis": DATASET_ANALYSIS_DIR,
    "raw_inference": RAW_INFERENCE_DIR,
    "models": MODEL_ARTIFACTS_DIR,
    "evaluation": EVALUATION_DIR,
    "outputs": OUTPUTS_DIR,
    "test_inference": TEST_INFERENCE_DIR,
    "raw_overlay": RAW_OVERLAY_DIR,
}

FEATURE_COLUMNS = ["area_px", "diag_px", "aspect_ratio"]
DEFAULT_SIZE_LABELS = ("Fries", "Fingerling", "Juvenile")


def _to_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def ensure_project_dirs() -> dict[str, Path]:
    for path in PROJECT_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    return PROJECT_PATHS.copy()


def reset_directory(path: str | Path, recreate: bool = False) -> Path:
    target = _to_path(path)
    if target.exists():
        shutil.rmtree(target)
        print(f"Removed existing folder: {target}")
    else:
        print(f"Folder not found, skip cleanup: {target}")
    if recreate:
        target.mkdir(parents=True, exist_ok=True)
    return target


def list_image_records(image_dir: str | Path, source_name: str | None = None) -> pd.DataFrame:
    target_dir = _to_path(image_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {target_dir}")

    rows: list[dict[str, str]] = []
    for path in sorted(target_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        row = {
            "filename": path.name,
            "filepath": str(path.resolve()),
        }
        if source_name is not None:
            row["source_split"] = source_name
        rows.append(row)

    if not rows:
        raise ValueError(f"Tidak ada file gambar pada folder: {target_dir}")

    return pd.DataFrame(rows)


def collect_dataset_image_index(
    dataset_dir: str | Path,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> pd.DataFrame:
    dataset_root = _to_path(dataset_dir)
    frames: list[pd.DataFrame] = []

    for split_name in splits:
        image_dir = dataset_root / split_name / "images"
        frames.append(list_image_records(image_dir, source_name=split_name))

    return pd.concat(frames, ignore_index=True)


def _find_matching_image(label_path: Path, image_dir: Path) -> Path:
    matches = [
        path
        for path in image_dir.glob(f"{label_path.stem}.*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not matches:
        raise FileNotFoundError(f"Gambar untuk label tidak ditemukan: {label_path}")
    return matches[0]


def collect_dataset_label_boxes(
    dataset_dir: str | Path,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> pd.DataFrame:
    dataset_root = _to_path(dataset_dir)
    rows: list[dict[str, object]] = []

    for split_name in splits:
        image_dir = dataset_root / split_name / "images"
        label_dir = dataset_root / split_name / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Folder split tidak lengkap: {split_name}")

        for label_path in sorted(label_dir.iterdir()):
            if label_path.suffix.lower() != ".txt":
                continue

            image_path = _find_matching_image(label_path, image_dir)
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            label_text = label_path.read_text(encoding="utf-8").strip()
            if not label_text:
                continue

            for annotation_index, line in enumerate(label_text.splitlines()):
                values = line.strip().split()
                if len(values) == 5:
                    class_id, x_center, y_center, box_width, box_height = map(float, values)
                    x_center_px = x_center * image_width
                    y_center_px = y_center * image_height
                    box_width_px = box_width * image_width
                    box_height_px = box_height * image_height
                    x1 = x_center_px - (box_width_px / 2)
                    y1 = y_center_px - (box_height_px / 2)
                    x2 = x_center_px + (box_width_px / 2)
                    y2 = y_center_px + (box_height_px / 2)
                elif len(values) >= 7 and len(values[1:]) % 2 == 0:
                    class_id = float(values[0])
                    polygon = np.asarray(list(map(float, values[1:])), dtype=float).reshape(-1, 2)
                    x1 = float(polygon[:, 0].min() * image_width)
                    y1 = float(polygon[:, 1].min() * image_height)
                    x2 = float(polygon[:, 0].max() * image_width)
                    y2 = float(polygon[:, 1].max() * image_height)
                    box_width_px = x2 - x1
                    box_height_px = y2 - y1
                else:
                    raise ValueError(f"Format label tidak valid pada {label_path}: {line}")

                rows.append(
                    {
                        "source_split": split_name,
                        "filename": image_path.name,
                        "filepath": str(image_path.resolve()),
                        "label_path": str(label_path.resolve()),
                        "annotation_index": annotation_index,
                        "class_id": int(class_id),
                        "box_source": "ground_truth_label",
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width_px": float(box_width_px),
                        "height_px": float(box_height_px),
                    }
                )

    if not rows:
        raise ValueError("Tidak ada anotasi bounding box pada dataset.")

    return pd.DataFrame(rows)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    work = df.copy()
    work["area_px"] = work.get("area_px", work["width_px"] * work["height_px"])
    work["diag_px"] = np.sqrt(work["width_px"] ** 2 + work["height_px"] ** 2)
    work["aspect_ratio"] = work["width_px"] / (work["height_px"] + 1e-8)
    matrix = work[FEATURE_COLUMNS].to_numpy(dtype=float)
    return work, matrix


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    unique_labels, counts = np.unique(labels, return_counts=True)
    row = {
        "n_samples": int(len(labels)),
        "n_clusters": int(len(unique_labels)),
        "silhouette_score": np.nan,
        "davies_bouldin_score": np.nan,
        "calinski_harabasz_score": np.nan,
        "largest_cluster_share": np.nan,
        "smallest_cluster_share": np.nan,
    }

    if len(unique_labels) >= 2 and len(labels) > len(unique_labels):
        row["silhouette_score"] = float(silhouette_score(X_scaled, labels))
        row["davies_bouldin_score"] = float(davies_bouldin_score(X_scaled, labels))
        row["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
        row["largest_cluster_share"] = float(counts.max() / len(labels))
        row["smallest_cluster_share"] = float(counts.min() / len(labels))

    return pd.DataFrame([row])


def build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    work, _ = build_feature_matrix(df)
    if "size_class" not in work.columns:
        work["size_class"] = work["cluster"].map(lambda value: f"Cluster {value}")

    summary = (
        work.groupby(["cluster", "size_class"], dropna=False)
        .agg(
            sample_count=("cluster", "size"),
            mean_width_px=("width_px", "mean"),
            mean_height_px=("height_px", "mean"),
            mean_area_px=("area_px", "mean"),
            median_area_px=("area_px", "median"),
            min_area_px=("area_px", "min"),
            max_area_px=("area_px", "max"),
            mean_diag_px=("diag_px", "mean"),
            mean_aspect_ratio=("aspect_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["mean_area_px", "cluster"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary["share"] = summary["sample_count"] / len(work)
    return summary


def _aspect_ratio_note(value: float) -> str:
    if value >= 1.2:
        return "bounding box cenderung lebih melebar secara horizontal"
    if value <= 0.85:
        return "bounding box cenderung lebih memanjang secara vertikal"
    return "bounding box relatif seimbang antara lebar dan tinggi"


def build_cluster_interpretation(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["Belum ada data cluster untuk diinterpretasikan."]

    ordered = " < ".join(
        f"{row.size_class} (cluster {int(row.cluster)})"
        for row in summary.itertuples(index=False)
    )
    lines = [f"Urutan ukuran berdasarkan rata-rata area adalah {ordered}."]

    for row in summary.itertuples(index=False):
        lines.append(
            (
                f"{row.size_class} pada cluster {int(row.cluster)} mencakup {int(row.sample_count)} deteksi "
                f"({row.share:.1%}) dengan rata-rata area {row.mean_area_px:,.0f} px2, median "
                f"{row.median_area_px:,.0f} px2, rentang {row.min_area_px:,.0f}-{row.max_area_px:,.0f} px2, "
                f"dan {_aspect_ratio_note(row.mean_aspect_ratio)}."
            )
        )

    return lines


def build_size_mapping(
    df: pd.DataFrame,
    labels: tuple[str, ...] = DEFAULT_SIZE_LABELS,
) -> dict[int, str]:
    cluster_means = df.groupby("cluster")["area_px"].mean().sort_values()
    if len(cluster_means) != len(labels):
        raise ValueError(
            f"Jumlah cluster ({len(cluster_means)}) tidak cocok dengan jumlah label ukuran ({len(labels)})."
        )
    return {
        int(cluster_id): label
        for cluster_id, label in zip(cluster_means.index.tolist(), labels)
    }


def save_cluster_artifacts(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    interpretation_lines: list[str],
    output_dir: str | Path = EVALUATION_DIR,
) -> dict[str, Path]:
    base_dir = _to_path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_path = base_dir / "cluster_interpretation_summary.csv"
    metrics_path = base_dir / "clustering_evaluation_summary.csv"
    text_path = base_dir / "cluster_interpretation.txt"

    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    text_path.write_text("\n".join(interpretation_lines), encoding="utf-8")

    return {
        "summary_csv": summary_path,
        "metrics_csv": metrics_path,
        "interpretation_txt": text_path,
    }


def save_detection_summary(metrics_by_split: dict[str, object], output_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, metrics in metrics_by_split.items():
        results_dict = getattr(metrics, "results_dict", {}) or {}
        rows.append(
            {
                "split": split_name,
                "precision": float(results_dict.get("metrics/precision(B)", np.nan)),
                "recall": float(results_dict.get("metrics/recall(B)", np.nan)),
                "mAP50": float(results_dict.get("metrics/mAP50(B)", np.nan)),
                "mAP50_95": float(results_dict.get("metrics/mAP50-95(B)", np.nan)),
                "fitness": float(results_dict.get("fitness", np.nan)),
                "save_dir": str(getattr(metrics, "save_dir", "")),
            }
        )

    summary = pd.DataFrame(rows)
    output_file = _to_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_file, index=False)
    return summary


def generate_size_overlays(
    df: pd.DataFrame,
    output_folder: str | Path,
    clear_existing: bool = False,
) -> tuple[Path, int]:
    output_dir = _to_path(output_folder)
    if clear_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_images = 0
    for filename, group in df.groupby("filename", sort=True):
        image_path = Path(group["filepath"].iloc[0])
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images += 1
            continue

        for row in group.itertuples(index=False):
            x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
            label = str(row.size_class)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(str(output_dir / filename), image)

    return output_dir, missing_images
