# Deteksi dan Clustering Ukuran Benih Lele Menggunakan YOLOv8 dan K-Means

## Ringkasan

Repositori ini mendokumentasikan pipeline computer vision untuk mendeteksi benih lele pada citra dan mengelompokkan ukurannya ke dalam tiga kategori, yaitu `Fries`, `Fingerling`, dan `Juvenile`. Sistem dibangun dengan dua komponen utama:

1. model deteksi objek YOLOv8 untuk menemukan lokasi benih lele, dan
2. clustering K-Means berbasis fitur geometri bounding box untuk analisis ukuran.

Secara metodologis, proyek dibagi ke dalam dua jalur yang saling melengkapi:

- jalur evaluasi deteksi, yang memakai dataset berlabel `train/valid/test`;
- jalur analisis ukuran dataset, yang memakai seluruh anotasi ground truth agar seluruh sampel berlabel ikut terwakili;
- jalur inferensi deployment, yang memakai gambar mentah pada `Catfish_baby_images` untuk simulasi penggunaan lapangan.

Versi repositori saat ini tidak hanya memuat kode, tetapi juga dataset, bobot hasil pelatihan, artefak evaluasi, dan output visual. Karena ukuran aset penelitian cukup besar, file besar disimpan menggunakan Git LFS agar tetap kompatibel dengan GitHub.

---

## Status Repositori dan Git LFS

Repositori GitHub proyek ini berada di:

- `https://github.com/AlwanFauzi/finalfixpkl`

Konten yang saat ini ikut dalam repositori meliputi:

- kode sumber dan notebook;
- dataset berlabel pada `Dataset/`;
- gambar mentah deployment pada `Catfish_baby_images/`;
- hasil pelatihan pada `runs/`;
- artefak analisis pada `artifacts/`;
- output visual pada `outputs/`.

Karena banyak file berukuran besar, repositori ini memakai Git LFS untuk tipe file berikut:

- `*.jpg`
- `*.jpeg`
- `*.png`
- `*.pt`
- `*.pkl`
- `*.npy`

Konfigurasi Git LFS tersimpan pada file `.gitattributes`. Oleh sebab itu, pengguna yang melakukan clone atau pull dari repositori ini harus memasang Git LFS terlebih dahulu.

---

## Latar Belakang

Klasifikasi ukuran benih lele merupakan aspek penting dalam budidaya karena ukuran ikan berhubungan langsung dengan kepadatan tebar, strategi pemberian pakan, dan manajemen pemeliharaan. Dalam praktik manual, pengukuran populasi benih dalam jumlah besar membutuhkan tenaga, waktu, dan konsistensi yang tinggi.

Pendekatan computer vision memberikan peluang otomatisasi pada dua level:

- deteksi objek, untuk mengetahui lokasi benih lele pada citra; dan
- analisis ukuran, untuk mengubah informasi geometrik bounding box menjadi kelompok ukuran yang dapat diinterpretasikan secara operasional.

Proyek ini tidak berhenti pada pelatihan model deteksi, tetapi juga membangun alur analisis ukuran yang lebih kuat secara akademik. Revisi metodologis terpenting pada versi ini adalah penggunaan seluruh anotasi ground truth dataset untuk analisis clustering ukuran, sehingga distribusi ukuran tidak bias terhadap kegagalan prediksi model.

---

## Tujuan Penelitian dan Implementasi

Tujuan proyek ini adalah:

1. melatih model YOLOv8 untuk mendeteksi objek benih lele;
2. mengevaluasi model pada split validasi dan pengujian;
3. mengekstraksi fitur ukuran berbasis bounding box;
4. mengelompokkan ukuran benih menggunakan K-Means;
5. memetakan cluster ke label ukuran yang bermakna secara praktis; dan
6. menyediakan hasil inferensi visual untuk kebutuhan analisis dan demonstrasi deployment.

---

## Kontribusi Utama Versi Saat Ini

Perbaikan utama pada versi repositori saat ini meliputi:

- penambahan evaluasi eksplisit pada split `test`;
- pembersihan folder `runs` sebelum retraining agar eksperimen tidak tercampur;
- pemisahan output proyek ke dalam `artifacts/` dan `outputs/`;
- penambahan evaluasi clustering menggunakan `silhouette`, `Davies-Bouldin`, dan `Calinski-Harabasz`;
- penambahan interpretasi tekstual hasil clustering;
- pemisahan analisis dataset dan inferensi deployment;
- penggunaan seluruh ground truth boxes untuk clustering dataset penuh;
- publikasi dataset, hasil, dan bobot training melalui Git LFS.

---

## Deskripsi Dataset

Dataset utama tersimpan di `Dataset/` dan berasal dari ekspor Roboflow dalam format YOLOv8.

Karakteristik dataset:

- jumlah citra: `725`
- jumlah kelas: `1`
- nama kelas: `lele`
- split data:
  - `train`: `515` citra
  - `valid`: `70` citra
  - `test`: `140` citra

Jumlah anotasi hasil parsing seluruh label:

- `train`: `2081` objek
- `valid`: `101` objek
- `test`: `249` objek
- total: `2431` objek

Selain dataset berlabel, terdapat folder `Catfish_baby_images/` yang berisi `753` gambar mentah untuk kebutuhan inferensi deployment dan visualisasi.

Catatan metodologis:

- dataset mengandung anotasi bertipe bounding box dan polygon;
- anotasi polygon dikonversi menjadi bounding box minimum enclosing rectangle;
- konversi ini memastikan seluruh anotasi tetap dapat dipakai dalam analisis ukuran.

---

## Metodologi

## 1. Pelatihan Model Deteksi

Model dasar yang digunakan adalah `yolov8s.pt`.

Pelatihan dilakukan dalam dua tahap.

### Tahap 1: Frozen Backbone Training

Tahap pertama bertujuan melakukan adaptasi awal model pada domain citra benih lele dengan menjaga sebagian backbone tetap stabil.

Konfigurasi utama:

- model awal: `yolov8s.pt`
- epoch: `10`
- optimizer: `AdamW`
- learning rate awal: `0.0006`
- backbone freeze: layer `0-9`
- augmentasi: mosaic, mixup, copy-paste, HSV, scaling, dan flip

### Tahap 2: Fine-Tuning Full Model

Tahap kedua melanjutkan pelatihan dari bobot hasil tahap pertama.

Konfigurasi utama:

- bobot awal: `runs/detect/train/weights/last.pt`
- epoch: `60`
- optimizer: `AdamW`
- learning rate awal: `0.0004`
- patience: `20`
- augmentasi: lebih konservatif dibanding tahap pertama

Tahap ini bertujuan meningkatkan generalisasi model setelah representasi domain mulai terbentuk.

---

## 2. Evaluasi Model Deteksi

Model dievaluasi pada split `valid` dan `test` menggunakan metrik:

- Precision
- Recall
- mAP50
- mAP50-95

Hasil evaluasi terbaru:

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Valid | 0.8818 | 0.9604 | 0.9778 | 0.8094 |
| Test | 0.9497 | 0.9105 | 0.9565 | 0.7489 |

Interpretasi singkat:

- model menunjukkan performa deteksi yang baik pada validasi dan pengujian;
- recall yang tinggi menandakan sebagian besar objek berhasil ditemukan;
- penurunan mAP50-95 pada split test tetap berada dalam rentang yang wajar untuk skenario generalisasi.

---

## 3. Ekstraksi Fitur Ukuran

Untuk setiap bounding box, proyek mengekstraksi tiga fitur geometri:

1. luas bounding box;
2. panjang diagonal bounding box; dan
3. rasio aspek bounding box.

Secara operasional:

- `area_px = width * height`
- `diag_px = sqrt(width^2 + height^2)`
- `aspect_ratio = width / (height + epsilon)`

Ketiga fitur ini dipilih karena:

- luas merepresentasikan skala total objek;
- diagonal merepresentasikan ukuran spasial gabungan;
- rasio aspek merepresentasikan kecenderungan bentuk memanjang atau lebih seimbang.

---

## 4. Clustering Ukuran dengan K-Means

Sebelum clustering, fitur dinormalisasi menggunakan `StandardScaler`.

Algoritma clustering yang digunakan:

- model: `KMeans`
- jumlah cluster: `k = 3`
- `random_state = 42`
- `n_init = 10`

Cluster kemudian dipetakan ke label ukuran berdasarkan rata-rata area:

- area terkecil -> `Fries`
- area menengah -> `Fingerling`
- area terbesar -> `Juvenile`

---

## 5. Evaluasi Clustering

Evaluasi clustering dilakukan menggunakan tiga metrik:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

Hasil evaluasi clustering terbaru:

| n_samples | n_clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---:|---:|---:|---:|---:|
| 2431 | 3 | 0.4239 | 0.7956 | 2031.3427 |

Interpretasi:

- `silhouette = 0.4239` menunjukkan kualitas pemisahan cluster pada tingkat cukup baik;
- `Davies-Bouldin = 0.7956` menunjukkan overlap antar cluster masih terkendali;
- `Calinski-Harabasz = 2031.3427` mendukung bahwa struktur cluster cukup stabil untuk analisis deskriptif ukuran.

---

## 6. Interpretasi Hasil Clustering

Ringkasan cluster terbaru:

- `Fries` atau cluster `1`: `727` sampel atau `29.9%`
- `Fingerling` atau cluster `2`: `464` sampel atau `19.1%`
- `Juvenile` atau cluster `0`: `1240` sampel atau `51.0%`

Interpretasi umum:

- `Fries` memiliki rata-rata area terkecil dan bentuk relatif seimbang;
- `Fingerling` memiliki area menengah dan cenderung lebih memanjang secara horizontal;
- `Juvenile` memiliki area terbesar dan mendominasi distribusi anotasi dataset.

Interpretasi tekstual lengkap tersimpan pada:

- `artifacts/evaluation/cluster_interpretation.txt`

---

## Keputusan Metodologis Penting

### A. Analisis dataset penuh

Untuk analisis dataset penuh, proyek menggunakan:

- `artifacts/dataset_analysis/dataset_boxes_all.csv`
- `artifacts/dataset_analysis/classified_sizes_all_dataset.csv`

Sumber datanya adalah seluruh anotasi ground truth dataset.

Alasan akademiknya:

- seluruh sampel berlabel harus ikut terwakili;
- analisis distribusi ukuran tidak boleh bias terhadap kegagalan deteksi model;
- clustering dataset dimaksudkan sebagai analisis deskriptif ukuran populasi berlabel.

### B. Inferensi deployment

Untuk simulasi deployment, proyek menggunakan:

- `artifacts/raw_inference/detections_raw_images.csv`
- `artifacts/raw_inference/classified_sizes_raw_images.csv`
- `outputs/final_inference_with_size/`

Sumber datanya adalah prediksi model pada `Catfish_baby_images/`.

Alasan praktisnya:

- folder ini merepresentasikan skenario penggunaan lapangan;
- hasilnya menunjukkan bagaimana model bekerja pada citra mentah yang tidak menjadi bagian dari split evaluasi resmi;
- overlay visual lebih relevan untuk demonstrasi operasional.

---

## Struktur Direktori

Struktur proyek yang dipublikasikan saat ini adalah:

```text
finalfixpkl/
|-- .gitattributes
|-- .gitignore
|-- README.md
|-- project_helpers.py
|-- train_lele_optimized.ipynb
|-- clustering.ipynb
|-- requirements.txt
|-- requirements_gpu.txt
|-- tools/
|   `-- rebuild_notebooks.py
|-- Dataset/
|-- Catfish_baby_images/
|-- runs/
|-- artifacts/
|   |-- dataset_analysis/
|   |-- evaluation/
|   |-- models/
|   `-- raw_inference/
`-- outputs/
    |-- inference_result/
    `-- final_inference_with_size/
```

---

## File dan Komponen Utama

### `train_lele_optimized.ipynb`

Notebook utama yang menjalankan:

- setup lingkungan;
- pembersihan folder `runs`;
- training tahap 1 dan tahap 2;
- evaluasi `valid` dan `test`;
- inferensi test set;
- parsing seluruh ground truth dataset;
- ekstraksi fitur dan clustering;
- interpretasi cluster;
- inferensi deployment pada `Catfish_baby_images`;
- pembuatan overlay hasil klasifikasi ukuran.

### `clustering.ipynb`

Notebook analisis yang berfokus pada:

- pemuatan artefak clustering;
- ringkasan distribusi cluster;
- evaluasi kualitas clustering;
- interpretasi cluster;
- visualisasi sebaran dan histogram.

### `project_helpers.py`

Modul utilitas yang menangani:

- pembuatan direktori proyek;
- pengumpulan indeks gambar dan parsing anotasi;
- ekstraksi fitur ukuran;
- evaluasi clustering;
- interpretasi cluster;
- penyimpanan artefak;
- pembuatan overlay visual.

### `.gitattributes`

File ini mengatur tipe file besar yang harus ditangani Git LFS. Selama strategi penyimpanan repositori tidak berubah, file ini harus tetap dikomit bersama perubahan data besar.

---

## Lingkungan, Dependensi, dan Cara Clone

### Prasyarat

Sebelum menjalankan proyek, pastikan tersedia:

- Python `3.11`
- Git
- Git LFS
- pip

### Instalasi dependensi Python

Opsi CPU:

```bash
pip install -r requirements.txt
```

Opsi GPU CUDA 12.8:

```bash
pip install -r requirements_gpu.txt
```

Dependensi utama yang digunakan:

- PyTorch
- Ultralytics
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Jupyter

### Cara clone repositori ini

Karena repositori memakai Git LFS, alur yang direkomendasikan adalah:

```bash
git lfs install
git clone https://github.com/AlwanFauzi/finalfixpkl.git
cd finalfixpkl
git lfs pull
```

Catatan:

- `git lfs install` cukup dijalankan sekali per mesin;
- `git lfs pull` memastikan file besar benar-benar terunduh, bukan hanya pointer;
- metode `Download ZIP` dari GitHub tidak direkomendasikan untuk repositori ini karena aset besar lebih aman diambil lewat alur Git LFS.

### Catatan tentang bobot dasar

Bobot dasar `yolov8s.pt` sengaja tidak dipublikasikan dalam repositori agar ukuran repo tetap terkontrol. Saat notebook dijalankan pada lingkungan yang sesuai, Ultralytics dapat mengunduh bobot dasar ini kembali sesuai kebutuhan.

---

## Cara Menjalankan Proyek

## 1. Pelatihan dan pembuatan artefak

Jalankan:

- `train_lele_optimized.ipynb`

Notebook harus dijalankan dari atas ke bawah tanpa melompati sel, karena notebook:

- menghapus `runs` lama di awal;
- melatih model ulang;
- menghasilkan artefak evaluasi;
- membangun model clustering;
- membuat hasil inferensi deployment.

## 2. Analisis clustering

Setelah notebook utama selesai, jalankan:

- `clustering.ipynb`

Notebook ini membaca artefak dari:

- `artifacts/dataset_analysis/`
- `artifacts/models/`
- `artifacts/evaluation/`

---

## Output Penting

### Evaluasi

- `artifacts/evaluation/detection_evaluation_summary.csv`
- `artifacts/evaluation/clustering_evaluation_summary.csv`
- `artifacts/evaluation/cluster_interpretation_summary.csv`
- `artifacts/evaluation/cluster_interpretation.txt`

### Analisis dataset

- `artifacts/dataset_analysis/dataset_boxes_all.csv`
- `artifacts/dataset_analysis/features_all_dataset.npy`
- `artifacts/dataset_analysis/classified_sizes_all_dataset.csv`

### Model clustering

- `artifacts/models/scaler_kmeans.pkl`
- `artifacts/models/kmeans_size.pkl`

### Inferensi deployment

- `artifacts/raw_inference/detections_raw_images.csv`
- `artifacts/raw_inference/classified_sizes_raw_images.csv`
- `outputs/final_inference_with_size/`

### Inferensi test set

- `outputs/inference_result/`

---

## Kekuatan Proyek

Kekuatan utama proyek ini adalah:

- pipeline end-to-end dari training sampai visualisasi deployment sudah lengkap;
- evaluasi deteksi tersedia untuk `valid` dan `test`;
- clustering dataset penuh mencakup seluruh anotasi berlabel;
- output proyek dipisahkan secara rapi antara analisis, model, dan visualisasi;
- interpretasi cluster tersedia dalam bentuk numerik dan tekstual;
- repositori GitHub sudah memuat kode, dataset, bobot hasil, dan output sehingga mudah direplikasi.

---

## Keterbatasan

Keterbatasan yang masih perlu dicatat:

- clustering masih berbasis fitur geometri bounding box, belum memakai fitur visual ROI yang lebih kaya;
- pemetaan cluster ke label ukuran masih bertumpu pada rata-rata area, sehingga interpretasi biologisnya perlu validasi ahli;
- masih ada gap performa antara valid dan test pada model deteksi;
- konversi anotasi polygon ke bounding box dapat menghilangkan sebagian informasi bentuk;
- ukuran repositori menjadi besar karena memuat data dan hasil penelitian, meskipun telah dibantu Git LFS.

---

## Repositori GitHub, Data, dan Git LFS

Repositori ini sekarang memang dimaksudkan sebagai repositori penelitian yang lengkap, bukan hanya repositori kode.

Konsekuensinya:

- dataset dan hasil eksperimen ikut dipublikasikan;
- file besar disimpan melalui Git LFS;
- riwayat branch `main` sudah disesuaikan agar kompatibel dengan GitHub dan penyimpanan file besar.

Prinsip pengelolaan file:

- file kode, dokumentasi, dan notebook disimpan sebagai Git biasa;
- file aset besar seperti gambar, bobot training, model clustering, dan array numerik besar ditangani Git LFS;
- file lingkungan lokal seperti `venv/`, cache Python, dan file editor tetap diabaikan oleh `.gitignore`.

Jika di masa depan strategi repositori berubah, maka:

- `.gitattributes` perlu diperbarui lebih dahulu bila pola file LFS berubah;
- `.gitignore` perlu diperbarui bila ada folder lokal baru yang tidak layak dipublikasikan.

---

## Reproducibility Notes

Proyek ini menggunakan `SEED = 42` untuk:

- Python random;
- NumPy;
- PyTorch.

Selain itu, mode deterministic CuDNN diaktifkan untuk meningkatkan konsistensi eksperimen. Walaupun demikian, reproduksibilitas absolut tetap dapat dipengaruhi oleh:

- perbedaan perangkat keras;
- perbedaan versi driver CUDA;
- komponen non-deterministic tertentu pada library deep learning.

---

## Etika dan Penggunaan

Hasil clustering dalam proyek ini sebaiknya dipandang sebagai alat bantu analisis, bukan pengganti penuh penilaian ahli budidaya. Penggunaan hasil untuk keputusan operasional sebaiknya tetap mempertimbangkan:

- kualitas citra;
- kondisi pengambilan gambar;
- variasi populasi ikan;
- verifikasi lapangan oleh pihak yang kompeten.

---

## Sitasi Dataset dan Perangkat Lunak

Dataset utama berasal dari ekspor Roboflow:

- nama dataset: `Dataset Lele`
- format: YOLOv8
- jumlah citra: `725`
- sumber: Roboflow Universe

Jika proyek ini digunakan dalam laporan, skripsi, tesis, atau publikasi, disarankan mencantumkan sitasi terhadap:

- dataset;
- Roboflow;
- Ultralytics YOLOv8;
- Scikit-learn;
- PyTorch.

---

## Penutup

Secara keseluruhan, proyek ini telah berkembang menjadi pipeline analisis yang relatif matang, baik dari sisi metodologi maupun dokumentasi. Versi saat ini membedakan secara tegas evaluasi model, analisis ukuran dataset penuh, dan inferensi deployment, sekaligus menyediakan data serta hasil eksperimen langsung di repositori GitHub melalui Git LFS.

README ini dimaksudkan sebagai dokumen utama untuk memahami tujuan penelitian, struktur proyek, keputusan metodologis, hasil utama, serta tata cara replikasi proyek secara utuh.
