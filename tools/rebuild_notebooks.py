from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT_DIR = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }


def build_train_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        metadata=notebook_metadata(),
        cells=[
            md(
                """
                # Setup Project Environment
                Instal dan import pustaka yang dibutuhkan, pastikan versi Python serta path workspace benar.
                """
            ),
            code(
                """
                import sys
                from pathlib import Path
                import random
                import csv

                import torch
                import numpy as np
                import pandas as pd
                from ultralytics import YOLO
                from tqdm import tqdm
                import joblib
                from PIL import Image
                from IPython.display import display
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans

                from project_helpers import (
                    DATASET_ANALYSIS_DIR,
                    EVALUATION_DIR,
                    MODEL_ARTIFACTS_DIR,
                    OUTPUTS_DIR,
                    RAW_INFERENCE_DIR,
                    RAW_OVERLAY_DIR,
                    ROOT_DIR,
                    TEST_INFERENCE_DIR,
                    build_cluster_interpretation,
                    build_cluster_summary,
                    build_feature_matrix,
                    build_size_mapping,
                    collect_dataset_label_boxes,
                    ensure_project_dirs,
                    evaluate_clustering,
                    generate_size_overlays,
                    list_image_records,
                    reset_directory,
                    save_cluster_artifacts,
                    save_detection_summary,
                )

                print(f"Python: {sys.version}")
                print(f"PyTorch: {torch.__version__}")
                print(f"CWD: {Path.cwd()}")
                print(f"Project root: {ROOT_DIR}")
                """
            ),
            md(
                """
                # Implement Code Fixes
                Perbaikan: hapus `runs` sebelum rerun, tambah evaluasi `test`, rapikan output ke folder `artifacts` dan `outputs`, lalu gunakan anotasi ground truth seluruh dataset untuk analisis clustering agar semua sampel berlabel ikut terwakili. Inferensi `Catfish_baby_images` tetap dipisah sebagai jalur deployment.
                """
            ),
            code(
                """
                SEED = 42
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED)

                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                DEVICE = 0 if torch.cuda.is_available() else "cpu"
                DATASET_DIR = ROOT_DIR / "Dataset"
                DATA_YAML = DATASET_DIR / "data.yaml"
                DATASET_SPLITS = ("train", "valid", "test")
                TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
                RAW_INPUT_DIR = ROOT_DIR / "Catfish_baby_images"
                RUNS_DIR = ROOT_DIR / "runs"
                PROJECT_PATHS = ensure_project_dirs()

                if not DATA_YAML.exists():
                    raise FileNotFoundError(f"Dataset YAML tidak ditemukan: {DATA_YAML}")
                if not TEST_IMAGES_DIR.exists():
                    raise FileNotFoundError(f"Folder test images tidak ditemukan: {TEST_IMAGES_DIR}")

                print(f"Data YAML: {DATA_YAML}")
                print(f"Dataset splits: {DATASET_SPLITS}")
                print(f"Test images: {TEST_IMAGES_DIR}")
                print(f"Raw input: {RAW_INPUT_DIR}")
                print(f"Device: {DEVICE}")
                print("Output directories:")
                for name, path in PROJECT_PATHS.items():
                    print(f"- {name}: {path}")
                """
            ),
            code(
                """
                # Bersihkan folder runs lama agar artefak rerun tidak tercampur
                if RUNS_DIR.exists():
                    reset_directory(RUNS_DIR)
                else:
                    print("Folder runs belum ada, lanjut tanpa cleanup.")
                """
            ),
            code(
                """
                # Stage 1: freeze backbone via train arg
                model = YOLO("yolov8s.pt")
                results_stage1 = model.train(
                    data=str(DATA_YAML),
                    epochs=10,
                    imgsz=640,
                    batch=8,
                    workers=2,
                    optimizer="AdamW",
                    lr0=0.0006,
                    lrf=0.01,
                    weight_decay=0.0008,
                    mosaic=0.15,
                    mixup=0.10,
                    copy_paste=0.10,
                    hsv_h=0.015,
                    hsv_s=0.6,
                    hsv_v=0.4,
                    scale=0.4,
                    fliplr=0.5,
                    freeze=list(range(10)),
                    seed=SEED,
                    name="train",
                    exist_ok=True,
                    device=DEVICE,
                )
                """
            ),
            code(
                """
                # Stage 2: fine-tune full model
                stage1_weights = RUNS_DIR / "detect" / "train" / "weights" / "last.pt"
                if not stage1_weights.exists():
                    raise FileNotFoundError(f"Stage-1 weights tidak ditemukan: {stage1_weights}")

                model = YOLO(str(stage1_weights))
                results_stage2 = model.train(
                    data=str(DATA_YAML),
                    epochs=60,
                    imgsz=640,
                    batch=8,
                    workers=2,
                    optimizer="AdamW",
                    lr0=0.0004,
                    weight_decay=0.0005,
                    mosaic=0.05,
                    mixup=0.05,
                    copy_paste=0.05,
                    scale=0.3,
                    fliplr=0.5,
                    flipud=0.2,
                    patience=20,
                    seed=SEED,
                    name="train2",
                    exist_ok=True,
                    device=DEVICE,
                )
                """
            ),
            code(
                """
                # Evaluation pada split valid dan test
                stage2_weights = RUNS_DIR / "detect" / "train2" / "weights" / "best.pt"
                if not stage2_weights.exists():
                    raise FileNotFoundError(f"Stage-2 weights tidak ditemukan: {stage2_weights}")

                model = YOLO(str(stage2_weights))
                valid_metrics = model.val(
                    data=str(DATA_YAML),
                    split="val",
                    device=DEVICE,
                    name="valid_eval",
                    exist_ok=True,
                )
                test_metrics = model.val(
                    data=str(DATA_YAML),
                    split="test",
                    device=DEVICE,
                    name="test_eval",
                    exist_ok=True,
                )

                detection_summary = save_detection_summary(
                    {"valid": valid_metrics, "test": test_metrics},
                    EVALUATION_DIR / "detection_evaluation_summary.csv",
                )
                display(detection_summary)
                """
            ),
            md(
                """
                # Run and Verify Unit Tests
                Jalankan inferensi singkat dan pastikan hasil batch inferensi tersimpan di dalam folder `outputs`.
                """
            ),
            code(
                """
                # Quick sanity inference pada satu gambar test
                test_candidates = sorted(
                    path for path in TEST_IMAGES_DIR.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if not test_candidates:
                    raise FileNotFoundError(f"Tidak ada gambar test pada {TEST_IMAGES_DIR}")

                test_img = test_candidates[0]
                model = YOLO(str(stage2_weights))
                res = model(str(test_img), conf=0.25, device=DEVICE)
                for result in res:
                    display(Image.fromarray(result.plot()))
                print(f"Sanity inference done untuk: {test_img.name}")
                """
            ),
            code(
                """
                # Inference batch test ke folder outputs/inference_result
                input_folder = TEST_IMAGES_DIR
                inference_output = TEST_INFERENCE_DIR
                if inference_output.exists():
                    reset_directory(inference_output)
                else:
                    print("Folder inference_result belum ada, akan dibuat saat inferensi.")

                results = model.predict(
                    source=str(input_folder),
                    save=True,
                    project=str(OUTPUTS_DIR),
                    name=inference_output.name,
                    conf=0.25,
                    exist_ok=True,
                    device=DEVICE,
                )
                print("Inference batch selesai. Folder:", inference_output)
                """
            ),
            md(
                """
                # Dataset-Wide Clustering Preparation
                Untuk memastikan seluruh sampel berlabel terwakili dalam analisis ukuran, tahap clustering memakai bounding box anotasi ground truth dari seluruh dataset (`train`, `valid`, `test`).
                """
            ),
            code(
                """
                # Kumpulkan seluruh bounding box dataset dari label ground truth
                DATASET_BOXES_CSV = DATASET_ANALYSIS_DIR / "dataset_boxes_all.csv"
                dataset_boxes_df = collect_dataset_label_boxes(DATASET_DIR, DATASET_SPLITS)
                dataset_boxes_df.to_csv(DATASET_BOXES_CSV, index=False)

                print("Box dataset penuh disimpan di:", DATASET_BOXES_CSV)
                display(
                    dataset_boxes_df.groupby("source_split")
                    .size()
                    .rename("box_count")
                    .reset_index()
                )
                display(dataset_boxes_df.head())
                """
            ),
            code(
                """
                # Ekstraksi fitur ukuran
                FEATURES_NPY = DATASET_ANALYSIS_DIR / "features_all_dataset.npy"
                dataset_boxes_df = pd.read_csv(DATASET_ANALYSIS_DIR / "dataset_boxes_all.csv")
                feature_df, X = build_feature_matrix(dataset_boxes_df)
                np.save(FEATURES_NPY, X)
                print("Fitur disimpan ke:", FEATURES_NPY, "Shape:", X.shape)
                display(feature_df[["filename", "area_px", "diag_px", "aspect_ratio"]].head())
                """
            ),
            code(
                """
                # Scaling + KMeans + evaluasi clustering
                FEATURES_NPY = DATASET_ANALYSIS_DIR / "features_all_dataset.npy"
                SCALER_PATH = MODEL_ARTIFACTS_DIR / "scaler_kmeans.pkl"
                KMEANS_PATH = MODEL_ARTIFACTS_DIR / "kmeans_size.pkl"

                X = np.load(FEATURES_NPY)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                joblib.dump(scaler, SCALER_PATH)

                kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                joblib.dump(kmeans, KMEANS_PATH)

                clustering_metrics = evaluate_clustering(X_scaled, clusters)
                cluster_counts = (
                    pd.Series(clusters)
                    .value_counts()
                    .sort_index()
                    .rename_axis("cluster")
                    .reset_index(name="count")
                )

                display(clustering_metrics)
                display(cluster_counts)
                print("Scaling & KMeans selesai. Center (scaled):")
                print(kmeans.cluster_centers_)
                """
            ),
            code(
                """
                # Mapping cluster -> label + interpretasi clustering
                dataset_boxes_df = pd.read_csv(DATASET_ANALYSIS_DIR / "dataset_boxes_all.csv")
                feature_df, X = build_feature_matrix(dataset_boxes_df)
                scaler = joblib.load(MODEL_ARTIFACTS_DIR / "scaler_kmeans.pkl")
                kmeans = joblib.load(MODEL_ARTIFACTS_DIR / "kmeans_size.pkl")
                clusters = kmeans.predict(scaler.transform(X))

                dataset_boxes_df["cluster"] = clusters
                dataset_boxes_df["area_px"] = feature_df["area_px"]
                mapping = build_size_mapping(dataset_boxes_df)
                dataset_boxes_df["size_class"] = dataset_boxes_df["cluster"].map(mapping)
                classified_path = DATASET_ANALYSIS_DIR / "classified_sizes_all_dataset.csv"
                dataset_boxes_df.to_csv(classified_path, index=False)

                cluster_summary = build_cluster_summary(dataset_boxes_df)
                clustering_metrics = evaluate_clustering(scaler.transform(X), clusters)
                interpretation_lines = build_cluster_interpretation(cluster_summary)
                artifact_paths = save_cluster_artifacts(
                    cluster_summary,
                    clustering_metrics,
                    interpretation_lines,
                    EVALUATION_DIR,
                )

                print("Cluster mapping:", mapping)
                display(cluster_summary)
                if "source_split" in dataset_boxes_df.columns:
                    display(
                        dataset_boxes_df.groupby(["source_split", "size_class"])
                        .size()
                        .rename("count")
                        .reset_index()
                    )
                print("Interpretasi hasil clustering:")
                for line in interpretation_lines:
                    print("-", line)
                print("File utama tersimpan:", classified_path)
                print("Artefak clustering:", artifact_paths)
                """
            ),
            code(
                """
                # Inferensi dan klasifikasi ukuran untuk Catfish_baby_images
                RAW_DETECTIONS_CSV = RAW_INFERENCE_DIR / "detections_raw_images.csv"
                RAW_CLASSIFIED_CSV = RAW_INFERENCE_DIR / "classified_sizes_raw_images.csv"
                raw_image_index = list_image_records(RAW_INPUT_DIR, source_name="raw_images")
                raw_paths = raw_image_index["filepath"].tolist()
                raw_split_lookup = dict(zip(raw_image_index["filepath"], raw_image_index["source_split"]))

                model = YOLO(str(stage2_weights))
                raw_results = model.predict(
                    source=str(RAW_INPUT_DIR),
                    conf=0.25,
                    stream=True,
                    device=DEVICE,
                    imgsz=640,
                    half=True if DEVICE != "cpu" else False,
                    batch=1,
                )

                wrote_any = False
                detected_raw_paths = set()
                with RAW_DETECTIONS_CSV.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=[
                            "source_split",
                            "filename",
                            "filepath",
                            "x1",
                            "y1",
                            "x2",
                            "y2",
                            "width_px",
                            "height_px",
                        ],
                    )
                    writer.writeheader()

                    for result in tqdm(raw_results, total=len(raw_paths), desc="Running raw-image inference"):
                        resolved_path = str(Path(result.path).resolve())
                        if result.boxes is None or len(result.boxes) == 0:
                            continue
                        detected_raw_paths.add(resolved_path)
                        for box in result.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = [float(value) for value in xyxy]
                            width, height = x2 - x1, y2 - y1
                            writer.writerow(
                                {
                                    "source_split": raw_split_lookup.get(resolved_path, "raw_images"),
                                    "filename": Path(result.path).name,
                                    "filepath": resolved_path,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "width_px": width,
                                    "height_px": height,
                                }
                            )
                            wrote_any = True

                if not wrote_any:
                    raise ValueError("Tidak ada deteksi pada Catfish_baby_images.")

                raw_detections_df = pd.read_csv(RAW_DETECTIONS_CSV)
                raw_feature_df, raw_X = build_feature_matrix(raw_detections_df)
                scaler = joblib.load(MODEL_ARTIFACTS_DIR / "scaler_kmeans.pkl")
                kmeans = joblib.load(MODEL_ARTIFACTS_DIR / "kmeans_size.pkl")
                mapping = (
                    pd.read_csv(DATASET_ANALYSIS_DIR / "classified_sizes_all_dataset.csv")[["cluster", "size_class"]]
                    .drop_duplicates()
                    .sort_values("cluster")
                    .set_index("cluster")["size_class"]
                    .to_dict()
                )
                raw_clusters = kmeans.predict(scaler.transform(raw_X))

                raw_detections_df["cluster"] = raw_clusters
                raw_detections_df["area_px"] = raw_feature_df["area_px"]
                raw_detections_df["size_class"] = raw_detections_df["cluster"].map(mapping)
                raw_detections_df.to_csv(RAW_CLASSIFIED_CSV, index=False)

                print("Deteksi raw images tersimpan di:", RAW_DETECTIONS_CSV)
                print("Klasifikasi raw images tersimpan di:", RAW_CLASSIFIED_CSV)
                print(f"Gambar raw dengan minimal 1 deteksi: {len(detected_raw_paths)} dari {len(raw_paths)}")
                """
            ),
            code(
                """
                # Generate overlay dengan label ukuran untuk Catfish_baby_images
                OUTPUT_FOLDER = RAW_OVERLAY_DIR
                classified_df = pd.read_csv(RAW_INFERENCE_DIR / "classified_sizes_raw_images.csv")
                output_folder, missing_images = generate_size_overlays(
                    classified_df,
                    OUTPUT_FOLDER,
                    clear_existing=True,
                )
                print("Overlay selesai. Folder:", output_folder)
                print("Jumlah gambar yang gagal dibaca:", missing_images)
                """
            ),
        ],
    )


def build_clustering_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        metadata=notebook_metadata(),
        cells=[
            md(
                """
                # Analisis Clustering Ukuran Benih Lele
                Notebook ini memuat ringkasan cluster, evaluasi kualitas clustering, dan interpretasi hasil berdasarkan seluruh bounding box anotasi ground truth pada dataset `train`, `valid`, dan `test`.
                """
            ),
            code(
                """
                import joblib
                import pandas as pd
                import matplotlib.pyplot as plt
                from IPython.display import display

                from project_helpers import (
                    DATASET_ANALYSIS_DIR,
                    EVALUATION_DIR,
                    MODEL_ARTIFACTS_DIR,
                    build_cluster_interpretation,
                    build_cluster_summary,
                    build_feature_matrix,
                    evaluate_clustering,
                )

                CLASSIFIED_PATH = DATASET_ANALYSIS_DIR / "classified_sizes_all_dataset.csv"
                SCALER_PATH = MODEL_ARTIFACTS_DIR / "scaler_kmeans.pkl"
                KMEANS_PATH = MODEL_ARTIFACTS_DIR / "kmeans_size.pkl"

                for path in [CLASSIFIED_PATH, SCALER_PATH, KMEANS_PATH]:
                    if not path.exists():
                        raise FileNotFoundError(f"Artefak tidak ditemukan: {path}")

                df = pd.read_csv(CLASSIFIED_PATH)
                feature_df, X = build_feature_matrix(df)
                scaler = joblib.load(SCALER_PATH)
                kmeans = joblib.load(KMEANS_PATH)
                X_scaled = scaler.transform(X)
                df["cluster"] = kmeans.predict(X_scaled)

                if "size_class" not in df.columns:
                    cluster_means = feature_df.assign(cluster=df["cluster"]).groupby("cluster")["area_px"].mean().sort_values()
                    mapping = {
                        cluster_means.index[0]: "Fries",
                        cluster_means.index[1]: "Fingerling",
                        cluster_means.index[2]: "Juvenile",
                    }
                    df["size_class"] = df["cluster"].map(mapping)

                print(f"Classified dataset file: {CLASSIFIED_PATH}")
                print(f"Evaluation folder: {EVALUATION_DIR}")
                print(f"Jumlah deteksi: {len(df)}")
                print(f"Cluster unik: {sorted(df['cluster'].unique())}")
                if "source_split" in df.columns:
                    display(
                        df.groupby("source_split")
                        .size()
                        .rename("detection_count")
                        .reset_index()
                    )
                display(df.head())
                """
            ),
            code(
                """
                cluster_summary = build_cluster_summary(df)
                interpretation_lines = build_cluster_interpretation(cluster_summary)

                display(cluster_summary)
                print("Interpretasi hasil clustering:")
                for line in interpretation_lines:
                    print("-", line)
                """
            ),
            code(
                """
                clustering_metrics = evaluate_clustering(X_scaled, df["cluster"].to_numpy())
                display(clustering_metrics)

                metric_row = clustering_metrics.iloc[0]
                silhouette = metric_row["silhouette_score"]
                db_index = metric_row["davies_bouldin_score"]
                ch_index = metric_row["calinski_harabasz_score"]

                if silhouette >= 0.5:
                    silhouette_note = "pemisahan antarkelompok kuat"
                elif silhouette >= 0.25:
                    silhouette_note = "pemisahan cluster cukup baik"
                else:
                    silhouette_note = "pemisahan cluster masih lemah"

                print(f"Silhouette score = {silhouette:.3f}, artinya {silhouette_note}.")
                print(f"Davies-Bouldin score = {db_index:.3f}; semakin kecil nilainya semakin baik.")
                print(f"Calinski-Harabasz score = {ch_index:.1f}; semakin besar nilainya semakin baik.")
                """
            ),
            code(
                """
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(
                    df["width_px"],
                    df["height_px"],
                    c=df["cluster"],
                    cmap="tab10",
                    alpha=0.7,
                )

                plt.xlabel("Bounding Box Width (pixel)")
                plt.ylabel("Bounding Box Height (pixel)")
                plt.title("Visualisasi Distribusi Cluster Ukuran Benih Lele")
                plt.colorbar(scatter, label="Cluster")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                """
            ),
            code(
                """
                plot_order = cluster_summary["size_class"].tolist()

                plt.figure(figsize=(8, 6))
                df.boxplot(column="area_px", by="size_class", grid=True)
                plt.xlabel("Size Class")
                plt.ylabel("Bounding Box Area (pixel2)")
                plt.title("Distribusi Ukuran Benih Lele Berdasarkan Cluster")
                plt.suptitle("")
                plt.tight_layout()
                plt.show()
                """
            ),
            code(
                """
                plt.figure(figsize=(8, 6))

                for cluster_id in sorted(df["cluster"].unique()):
                    subset = df[df["cluster"] == cluster_id]
                    label = subset["size_class"].mode().iloc[0]
                    plt.hist(
                        subset["area_px"],
                        bins=20,
                        alpha=0.5,
                        label=f"Cluster {cluster_id} - {label}",
                    )

                plt.xlabel("Bounding Box Area (pixel2)")
                plt.ylabel("Jumlah Sampel")
                plt.title("Histogram Distribusi Ukuran Benih Lele per Cluster")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                """
            ),
        ],
    )


def write_notebook(path: Path, notebook: nbf.NotebookNode) -> None:
    path.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Notebook updated: {path}")


def main() -> None:
    write_notebook(ROOT_DIR / "train_lele_optimized.ipynb", build_train_notebook())
    write_notebook(ROOT_DIR / "clustering.ipynb", build_clustering_notebook())


if __name__ == "__main__":
    main()
