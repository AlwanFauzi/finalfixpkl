# Deteksi dan Clustering Ukuran Benih Lele Menggunakan YOLOv8 dan K-Means

## Ringkasan

Proyek ini membangun pipeline analisis citra untuk dua tujuan utama:

1. mendeteksi objek benih lele pada citra menggunakan model deteksi objek berbasis YOLOv8, dan
2. mengelompokkan ukuran benih lele ke dalam tiga kategori ukuran (`Fries`, `Fingerling`, dan `Juvenile`) menggunakan pendekatan unsupervised clustering berbasis fitur geometri bounding box.

Pipeline dirancang dalam dua jalur yang saling melengkapi:

- **Jalur evaluasi model deteksi**, yang menggunakan dataset berlabel `train/valid/test` untuk pelatihan dan evaluasi objektif model.
- **Jalur analisis ukuran dan deployment**, yang menggunakan:
  - seluruh anotasi ground truth dataset untuk analisis clustering yang representatif secara akademik, dan
  - gambar mentah pada `Catfish_baby_images` untuk inferensi lapangan dan visualisasi hasil klasifikasi ukuran.

Struktur proyek telah diperbarui agar lebih rapi, reproducible, dan mudah dipublikasikan ke GitHub, dengan pemisahan yang tegas antara kode sumber, artefak model, hasil evaluasi, dan output visual.

---

## Latar Belakang

Klasifikasi ukuran benih lele merupakan komponen penting dalam manajemen budidaya, karena ukuran ikan berkaitan langsung dengan kebutuhan pakan, kepadatan tebar, dan strategi pemeliharaan. Pada praktik lapangan, pengukuran manual terhadap ratusan hingga ribuan benih memerlukan waktu, tenaga, dan konsistensi yang tinggi.

Pendekatan computer vision menawarkan otomatisasi pada dua level:

- **deteksi objek**, untuk menemukan lokasi benih lele pada citra, dan
- **analisis ukuran**, untuk mengubah informasi geometrik bounding box menjadi kelompok ukuran yang lebih mudah diinterpretasikan.

Proyek ini tidak hanya melatih model deteksi, tetapi juga menyusun alur analisis yang dapat dipertanggungjawabkan secara metodologis. Perbaikan utama terbaru dalam proyek ini adalah penggunaan **seluruh bounding box anotasi ground truth** pada dataset untuk clustering ukuran, sehingga semua sampel berlabel ikut terwakili dalam analisis, tidak hanya sampel yang berhasil diprediksi model.

---

## Tujuan Penelitian dan Implementasi

Tujuan proyek ini adalah:

1. melatih model YOLOv8 untuk mendeteksi benih lele pada citra,
2. mengevaluasi performa model menggunakan split validasi dan pengujian,
3. mengekstraksi fitur ukuran berbasis bounding box,
4. mengelompokkan ukuran benih lele menggunakan K-Means,
5. memetakan cluster menjadi label ukuran yang bermakna secara praktis, dan
6. menyediakan hasil inferensi visual pada data mentah untuk kebutuhan demonstrasi atau deployment.

---

## Kontribusi Utama Proyek

Versi proyek saat ini memiliki beberapa kontribusi teknis penting:

- menambahkan evaluasi deteksi pada **test set** secara eksplisit,
- membersihkan folder `runs` sebelum retraining agar tidak tercampur dengan eksperimen sebelumnya,
- memindahkan output inferensi visual ke folder `outputs/` agar lebih rapi,
- menambahkan evaluasi clustering (`silhouette`, `Davies-Bouldin`, `Calinski-Harabasz`),
- menambahkan interpretasi tekstual hasil clustering,
- memisahkan artefak analisis dataset dan artefak inferensi deployment,
- memperbaiki metodologi clustering dataset dengan memakai **ground truth boxes** seluruh dataset, dan
- merapikan struktur direktori untuk persiapan publikasi ke GitHub.

---

## Deskripsi Dataset

Dataset utama tersimpan pada folder `Dataset/` dan berasal dari ekspor Roboflow.

Karakteristik dataset:

- Jumlah citra: **725**
- Format anotasi: **YOLOv8**
- Kelas objek: **1 kelas**, yaitu `lele`
- Split dataset:
  - `train`: 515 citra
  - `valid`: 70 citra
  - `test`: 140 citra

Jumlah anotasi bounding box hasil parsing seluruh label:

- `train`: 2081 objek
- `valid`: 101 objek
- `test`: 249 objek
- total: **2431 objek**

Catatan metodologis penting:

- Dataset mengandung campuran anotasi bertipe **bounding box** dan **segmentasi polygon**.
- Untuk menjaga konsistensi analisis ukuran, anotasi polygon dikonversi menjadi bounding box minimum-enclosing rectangle.
- Dengan pendekatan ini, seluruh anotasi tetap dapat digunakan untuk ekstraksi fitur ukuran.

Selain dataset berlabel, proyek juga memiliki folder `Catfish_baby_images/` yang berisi gambar mentah untuk inferensi deployment.

---

## Metodologi

## 1. Pelatihan Model Deteksi

Model dasar yang digunakan adalah `yolov8s.pt`.

Pelatihan dilakukan dalam dua tahap:

### Tahap 1: Frozen Backbone Training

Tujuan tahap ini adalah melakukan adaptasi awal model pada domain citra benih lele, sambil menjaga fitur backbone awal tetap stabil.

Konfigurasi utama:

- model awal: `yolov8s.pt`
- epoch: 10
- optimizer: `AdamW`
- learning rate awal: 0.0006
- backbone freeze: layer 0-9
- augmentasi ringan: mosaic, mixup, copy-paste, hsv, scaling, flip

### Tahap 2: Fine-Tuning Full Model

Setelah tahap awal selesai, model dilanjutkan ke fine-tuning seluruh parameter.

Konfigurasi utama:

- sumber bobot: `runs/detect/train/weights/last.pt`
- epoch: 60
- optimizer: `AdamW`
- learning rate awal: 0.0004
- patience: 20
- augmentasi lebih konservatif dibanding tahap 1

Tujuan tahap kedua adalah meningkatkan kemampuan generalisasi model setelah domain-specific representation mulai terbentuk.

---

## 2. Evaluasi Model Deteksi

Model dievaluasi pada split `valid` dan `test`.

Metrik yang digunakan:

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

- Model menunjukkan performa deteksi yang baik pada validasi dan pengujian.
- Nilai recall yang tinggi menunjukkan sebagian besar objek berhasil ditemukan.
- Penurunan mAP50-95 pada test dibanding valid mengindikasikan adanya gap generalisasi yang wajar, tetapi model masih cukup kuat untuk dipakai pada tahap inferensi ukuran.

---

## 3. Ekstraksi Fitur Ukuran

Untuk setiap bounding box, diekstraksi tiga fitur geometri utama:

1. **Luas bounding box**
   
   \[
   A = w \times h
   \]

2. **Panjang diagonal bounding box**

   \[
   d = \sqrt{w^2 + h^2}
   \]

3. **Rasio aspek**

   \[
   r = \frac{w}{h + \epsilon}
   \]

dengan:

- \(w\) = lebar bounding box dalam piksel
- \(h\) = tinggi bounding box dalam piksel
- \(\epsilon\) = konstanta kecil untuk menghindari pembagian nol

Fitur-fitur ini dipilih karena:

- luas menangkap skala total objek,
- diagonal menangkap ukuran spasial gabungan,
- rasio aspek menangkap kecenderungan bentuk memanjang horizontal atau vertikal.

---

## 4. Clustering Ukuran dengan K-Means

Sebelum clustering, fitur dinormalisasi menggunakan `StandardScaler`.

Algoritma clustering yang digunakan adalah `KMeans` dengan:

- jumlah cluster: `k = 3`
- `random_state = 42`
- `n_init = 10`

Tiga cluster ini kemudian dipetakan ke label ukuran:

- cluster dengan rata-rata area terkecil -> `Fries`
- cluster dengan rata-rata area menengah -> `Fingerling`
- cluster dengan rata-rata area terbesar -> `Juvenile`

---

## 5. Evaluasi Clustering

Evaluasi clustering dilakukan menggunakan tiga metrik:

- **Silhouette Score**: semakin tinggi semakin baik
- **Davies-Bouldin Index**: semakin rendah semakin baik
- **Calinski-Harabasz Index**: semakin tinggi semakin baik

Hasil evaluasi clustering terbaru:

| n_samples | n_clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---:|---:|---:|---:|---:|
| 2431 | 3 | 0.4239 | 0.7956 | 2031.3427 |

Interpretasi:

- `silhouette = 0.4239` menunjukkan pemisahan cluster berada pada kategori **cukup baik**, meskipun belum sangat kuat.
- `Davies-Bouldin = 0.7956` mengindikasikan cluster relatif terpisah dengan overlap yang masih terkendali.
- `Calinski-Harabasz = 2031.34` mendukung bahwa struktur cluster cukup stabil untuk analisis deskriptif ukuran.

---

## 6. Interpretasi Hasil Clustering

Berdasarkan hasil terbaru:

- `Fries` (cluster 1): 727 sampel, 29.9%
- `Fingerling` (cluster 2): 464 sampel, 19.1%
- `Juvenile` (cluster 0): 1240 sampel, 51.0%

Ringkasan interpretasi:

- `Fries` memiliki rata-rata area bounding box terkecil dan cenderung seimbang antara lebar dan tinggi.
- `Fingerling` memiliki area menengah dan bounding box lebih memanjang horizontal.
- `Juvenile` memiliki area terbesar dan mendominasi distribusi dataset.

Interpretasi lengkap disimpan pada:

- `artifacts/evaluation/cluster_interpretation.txt`

---

## Keputusan Metodologis Penting

Versi proyek ini secara eksplisit membedakan dua sumber data:

### A. Analisis dataset penuh

Untuk file:

- `artifacts/dataset_analysis/dataset_boxes_all.csv`
- `artifacts/dataset_analysis/classified_sizes_all_dataset.csv`

proyek menggunakan **seluruh anotasi ground truth** dari dataset berlabel.

Alasan akademiknya:

- seluruh sampel berlabel harus ikut terwakili,
- hasil clustering ukuran tidak boleh bias terhadap kegagalan deteksi model,
- analisis distribusi ukuran dataset harus didasarkan pada anotasi referensi, bukan hanya prediksi.

### B. Inferensi deployment

Untuk file:

- `artifacts/raw_inference/detections_raw_images.csv`
- `artifacts/raw_inference/classified_sizes_raw_images.csv`
- `outputs/final_inference_with_size/`

proyek menggunakan **prediksi model** pada `Catfish_baby_images`.

Alasan praktisnya:

- folder ini merepresentasikan data operasional atau deployment,
- hasilnya menggambarkan bagaimana model digunakan pada gambar baru,
- overlay visual lebih relevan untuk demonstrasi aplikasi lapangan.

---

## Struktur Direktori

Struktur proyek saat ini dirapikan sebagai berikut:

```text
finalfixpkl/
├── artifacts/
│   ├── dataset_analysis/
│   │   ├── dataset_boxes_all.csv
│   │   ├── features_all_dataset.npy
│   │   └── classified_sizes_all_dataset.csv
│   ├── evaluation/
│   │   ├── detection_evaluation_summary.csv
│   │   ├── clustering_evaluation_summary.csv
│   │   ├── cluster_interpretation_summary.csv
│   │   └── cluster_interpretation.txt
│   ├── models/
│   │   ├── scaler_kmeans.pkl
│   │   └── kmeans_size.pkl
│   └── raw_inference/
│       ├── detections_raw_images.csv
│       └── classified_sizes_raw_images.csv
├── outputs/
│   ├── inference_result/
│   └── final_inference_with_size/
├── Dataset/
├── Catfish_baby_images/
├── runs/
├── project_helpers.py
├── train_lele_optimized.ipynb
├── clustering.ipynb
├── requirements.txt
├── requirements_gpu.txt
└── README.md
```

---

## File dan Fungsi Utama

### `train_lele_optimized.ipynb`

Notebook utama yang menjalankan:

- setup lingkungan,
- cleanup folder `runs`,
- training tahap 1,
- training tahap 2,
- evaluasi valid dan test,
- inferensi test set,
- ekstraksi bounding box ground truth dataset penuh,
- ekstraksi fitur,
- clustering,
- interpretasi cluster,
- inferensi deployment pada `Catfish_baby_images`,
- pembuatan overlay hasil klasifikasi ukuran.

### `clustering.ipynb`

Notebook analisis clustering yang berfokus pada:

- pemuatan artefak clustering,
- ringkasan distribusi cluster,
- evaluasi kualitas clustering,
- interpretasi cluster,
- visualisasi sebaran dan histogram.

### `project_helpers.py`

Berisi fungsi utilitas untuk:

- pembuatan direktori proyek,
- pengumpulan indeks gambar,
- parsing anotasi label ground truth,
- ekstraksi fitur ukuran,
- evaluasi clustering,
- interpretasi cluster,
- penyimpanan artefak,
- pembuatan overlay visual.

---

## Lingkungan dan Dependensi

### Opsi CPU

```bash
pip install -r requirements.txt
```

### Opsi GPU CUDA 12.8

```bash
pip install -r requirements_gpu.txt
```

Dependensi inti:

- Python 3.11
- PyTorch
- Ultralytics
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Jupyter

---

## Cara Menjalankan Proyek

## 1. Pelatihan dan pembuatan seluruh artefak

Jalankan notebook:

- `train_lele_optimized.ipynb`

Urutan eksekusi harus dari atas ke bawah tanpa melompati sel, karena notebook:

- menghapus folder `runs` lama di awal,
- melatih model ulang,
- membuat artefak evaluasi,
- membangun artefak clustering,
- dan menghasilkan output deployment.

## 2. Analisis clustering

Setelah notebook utama selesai, jalankan:

- `clustering.ipynb`

Notebook ini membaca artefak yang sudah dihasilkan dari `artifacts/dataset_analysis/`, `artifacts/models/`, dan `artifacts/evaluation/`.

---

## Hasil Output Penting

### Output evaluasi

- `artifacts/evaluation/detection_evaluation_summary.csv`
- `artifacts/evaluation/clustering_evaluation_summary.csv`
- `artifacts/evaluation/cluster_interpretation_summary.csv`
- `artifacts/evaluation/cluster_interpretation.txt`

### Output analisis dataset

- `artifacts/dataset_analysis/dataset_boxes_all.csv`
- `artifacts/dataset_analysis/features_all_dataset.npy`
- `artifacts/dataset_analysis/classified_sizes_all_dataset.csv`

### Output model clustering

- `artifacts/models/scaler_kmeans.pkl`
- `artifacts/models/kmeans_size.pkl`

### Output inferensi deployment

- `artifacts/raw_inference/detections_raw_images.csv`
- `artifacts/raw_inference/classified_sizes_raw_images.csv`
- `outputs/final_inference_with_size/`

### Output inferensi test set

- `outputs/inference_result/`

---

## Kekuatan Proyek

Kekuatan utama proyek saat ini adalah:

- pipeline end-to-end dari training sampai visualisasi deployment sudah lengkap,
- evaluasi deteksi dilakukan pada valid dan test set,
- analisis clustering dataset penuh kini mencakup seluruh anotasi berlabel,
- struktur output rapi dan mudah direproduksi,
- interpretasi hasil tersedia dalam bentuk numerik dan tekstual,
- kode bantu dipisah dari notebook sehingga alur lebih terjaga.

---

## Keterbatasan

Beberapa keterbatasan yang masih perlu dicatat:

- clustering menggunakan fitur geometri bounding box, belum memakai fitur visual atau morfologis yang lebih kaya,
- pemetaan cluster ke label ukuran masih berbasis rata-rata area, sehingga interpretasi biologisnya perlu divalidasi oleh domain expert,
- model deteksi masih memiliki gap performa antara valid dan test,
- campuran anotasi box dan polygon memerlukan konversi ke bounding box, yang dapat menghilangkan sebagian detail bentuk.

---

## Rekomendasi Pengembangan Lanjutan

Beberapa arah pengembangan yang direkomendasikan:

1. menambahkan validasi domain untuk batas ukuran biologis tiap cluster,
2. mengevaluasi jumlah cluster optimal dengan Elbow Method dan Silhouette Analysis,
3. membandingkan K-Means dengan Gaussian Mixture Model atau Agglomerative Clustering,
4. menambahkan fitur visual dari ROI objek, bukan hanya fitur geometri,
5. menyusun skrip Python non-notebook agar pipeline dapat dijalankan dari command line,
6. menambahkan evaluasi statistik atau uji robustnes antar split data.

---

## Reproducibility Notes

Proyek ini menggunakan seed tetap (`SEED = 42`) untuk:

- Python random
- NumPy
- PyTorch

Selain itu, mode deterministic CuDNN diaktifkan untuk meningkatkan konsistensi hasil eksperimen.

Perlu dicatat bahwa reproduksibilitas absolut tetap dapat dipengaruhi oleh:

- perbedaan perangkat keras,
- perbedaan versi driver CUDA,
- dan karakteristik non-deterministic tertentu pada library deep learning.

---

## Etika dan Penggunaan

Hasil clustering pada proyek ini harus dipandang sebagai alat bantu analisis, bukan pengganti penuh penilaian ahli budidaya. Penggunaan hasil untuk keputusan operasional sebaiknya tetap mempertimbangkan:

- kualitas citra,
- kondisi pengambilan gambar,
- variasi populasi ikan,
- dan verifikasi lapangan.

---

## Sitasi Dataset

Dataset berasal dari ekspor Roboflow:

- Nama dataset: `Dataset Lele`
- Format: YOLOv8
- Jumlah citra: 725
- Sumber: Roboflow Universe

Jika proyek ini digunakan dalam laporan, skripsi, tesis, atau publikasi, disarankan mencantumkan sitasi terhadap dataset dan perangkat lunak yang digunakan, termasuk Roboflow, Ultralytics YOLOv8, Scikit-learn, dan PyTorch.

---

## Konten yang Direkomendasikan untuk GitHub

Untuk publikasi repo, file yang direkomendasikan untuk dilacak:

- notebook (`train_lele_optimized.ipynb`, `clustering.ipynb`)
- helper code (`project_helpers.py`)
- `tools/`
- `requirements.txt`
- `requirements_gpu.txt`
- `README.md`
- `.gitignore`

Sedangkan dataset, model, artefak hasil, dan output visual sebaiknya tidak dipush langsung ke GitHub karena ukuran besar, sifat sementara, atau potensi kerahasiaan data.

---

## Penutup

Secara keseluruhan, proyek ini telah berkembang dari sekadar notebook training menjadi pipeline analisis yang lebih matang secara metodologis. Revisi terbaru memperkuat validitas akademik analisis clustering, meningkatkan kerapian struktur proyek, dan memisahkan dengan jelas antara evaluasi dataset dan penggunaan model pada data deployment.

README ini dimaksudkan sebagai dokumentasi utama untuk replikasi, evaluasi, dan publikasi proyek.
