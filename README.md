# YOLO Performance Benchmark

Program benchmarking performa model YOLOv8 untuk skripsi **"ANALISIS PERBANDINGAN KINERJA MODEL YOLO PADA PERANGKAT APPLE SILICON M4 DAN PERANGKAT BERBASIS GPU NVIDIA RTX 3070"**.

Program ini menguji semua varian YOLOv8 (nano, small, medium, large, extra-large) secara otomatis pada video lalu lintas HD 1080p, mengukur metrik performa lengkap (FPS, inference time, CPU, GPU, RAM), menganalisis bottleneck per fase pemrosesan, dan menghasilkan data perbandingan antar perangkat secara otomatis.

## Prasyarat

- **Python 3.9+**
- **Video uji** dalam format MP4 (resolusi 1080p) ditempatkan di `data/video.mp4`
- **Perangkat yang didukung:**
  - MacBook Air dengan Apple Silicon M4 (menggunakan MPS)
  - PC dengan GPU NVIDIA RTX 3070 (menggunakan CUDA)
  - CPU-only sebagai fallback

## Instalasi

1. Clone repository ini:

```bash
git clone <repository-url>
cd yolo-performance-test
```

2. Buat virtual environment (opsional tapi disarankan):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# atau
.venv\Scripts\activate     # Windows
```

3. Install dependensi:

```bash
pip install -r requirements.txt
```

> **Catatan:** Untuk perangkat NVIDIA, pastikan CUDA toolkit sudah terinstall dan gunakan versi PyTorch yang sesuai dengan versi CUDA Anda. Lihat [pytorch.org](https://pytorch.org/get-started/locally/) untuk instruksi instalasi PyTorch dengan CUDA.

## Mengunduh Model

Unduh semua 5 varian model YOLOv8 (nano, small, medium, large, extra-large) ke direktori `models/`:

```bash
python download_models.py
```

Script ini akan:
- Mengunduh model yang belum ada di direktori `models/`
- Melewati model yang sudah terunduh sebelumnya
- Menampilkan ukuran file setiap model setelah selesai

Model yang diunduh:
| Model | File | Ukuran (approx.) |
|-------|------|-------------------|
| YOLOv8n (Nano) | `models/yolov8n.pt` | ~6 MB |
| YOLOv8s (Small) | `models/yolov8s.pt` | ~22 MB |
| YOLOv8m (Medium) | `models/yolov8m.pt` | ~50 MB |
| YOLOv8l (Large) | `models/yolov8l.pt` | ~84 MB |
| YOLOv8x (Extra-Large) | `models/yolov8x.pt` | ~131 MB |

## Menjalankan Benchmark

Pastikan video uji sudah ditempatkan di `data/video.mp4`, lalu jalankan:

```bash
python benchmark.py
```

Program akan secara otomatis:
1. Mendeteksi perangkat (Apple Silicon MPS / NVIDIA CUDA / CPU)
2. Menjalankan setiap varian model sebanyak `EXPERIMENT_REPEATS` kali
3. Melakukan warm-up configurable (`WARMUP_FRAMES`, default 50 frame) yang tidak masuk CSV hasil
4. Mengukur metrik per frame: `preprocess_ms`, `predict_ms`, `postprocess_ms`, CPU, GPU jika tersedia, RAM, dan RAM proses
5. Menyimpan hasil ke `results/{nama_perangkat}/{timestamp}/run_{n}/{model}/`

Hasil benchmark akan ditampilkan di console setelah setiap model selesai diuji.

### MacBook Apple Silicon M4

Untuk MacBook Apple Silicon M4, backend yang dipakai adalah MPS jika tersedia:

```bash
python benchmark.py
python generate_tables.py
```

Sebelum mengambil data final untuk skripsi, pastikan script sudah stabil dan hasil lama tidak dicampur dengan hasil final. Cara yang disarankan adalah memindahkan atau membersihkan folder `results/{nama_perangkat}/` secara manual setelah yakin tidak ada data penting di sana, lalu jalankan ulang benchmark final dari awal.

Apple Silicon GPU usage tidak diklaim jika pembacaannya tidak valid untuk benchmarking. Pada kondisi tersebut, kolom `gpu_percent` akan berisi `not_available`, dan sumber pengukuran dicatat di `metadata.json`.

### Definisi FPS

- `throughput_fps`: metrik FPS utama, dihitung dari `measured_frames / total_elapsed_seconds`.
- `fps_from_avg_total_ms`: metrik konsistensi, dihitung dari `1000 / avg_total_ms`.
- `mean_instant_fps`: rata-rata FPS instan per frame. Ini hanya metrik tambahan, bukan metrik utama untuk tabel performa.

## Perbandingan Antar Perangkat

Untuk membandingkan hasil dari dua perangkat (Apple Silicon M4 vs NVIDIA RTX 3070):

### Langkah 1: Jalankan benchmark di kedua perangkat

Jalankan `python benchmark.py` di masing-masing perangkat. Hasil akan tersimpan di:
- `results/apple_m4/` (atau nama perangkat Apple Silicon yang terdeteksi)
- `results/nvidia_rtx3070/` (atau nama perangkat NVIDIA yang terdeteksi)

### Langkah 2: Kumpulkan hasil di satu mesin

Salin direktori hasil dari perangkat kedua ke mesin yang akan menjalankan perbandingan. Pastikan kedua direktori hasil berada di dalam folder `results/`.

### Langkah 3: Jalankan perbandingan

```python
from analysis.compare import generate_comparison

generate_comparison(
    device1_dir="results/apple_m4",
    device2_dir="results/nvidia_rtx3070",
    output_dir="results/comparison/"
)
```

Atau jalankan langsung dari terminal:

```bash
python -c "from analysis.compare import generate_comparison; generate_comparison('results/apple_m4', 'results/nvidia_rtx3070')"
```

> **Catatan:** Sesuaikan nama direktori (`apple_m4`, `nvidia_rtx3070`) dengan nama perangkat yang terdeteksi secara otomatis oleh program saat benchmark dijalankan.

Hasil perbandingan yang dihasilkan:
- `results/comparison/comparison.csv` — Tabel perbandingan lengkap
- `results/comparison/fps_comparison.png` — Grafik perbandingan FPS
- `results/comparison/predict_time_comparison.png` — Grafik perbandingan waktu `model.predict()`
- `results/comparison/cpu_comparison.png` — Grafik perbandingan penggunaan CPU
- `results/comparison/gpu_comparison.png` — Grafik perbandingan penggunaan GPU
- `results/comparison/ram_comparison.png` — Grafik perbandingan penggunaan RAM

## Struktur Proyek

```
yolo-performance-test/
├── config.py                  # Konfigurasi benchmark
├── benchmark.py               # Entry point utama benchmark
├── download_models.py         # Script untuk mengunduh model YOLOv8
├── requirements.txt           # Dependensi Python
├── utils/
│   ├── device.py              # Deteksi perangkat (MPS/CUDA/CPU)
│   ├── monitor.py             # Monitoring sistem (CPU, GPU, RAM)
│   ├── timer.py               # Pengukuran waktu per fase
│   ├── logger.py              # Pencatatan metrik ke CSV
│   └── metadata.py            # Pengumpulan metadata perangkat
├── analysis/
│   ├── bottleneck.py          # Analisis bottleneck per fase
│   ├── summary.py             # Statistik agregat per model
│   └── compare.py             # Perbandingan antar perangkat
├── data/
│   └── video.mp4              # Video uji (1080p lalu lintas)
├── models/                    # Bobot model YOLOv8 (n, s, m, l, x)
├── results/                   # Output hasil benchmark
│   ├── {nama_perangkat}/      # Hasil per perangkat
│   └── comparison/            # Hasil perbandingan antar perangkat
└── tests/                     # Unit tests
```

## Konfigurasi

Pengaturan benchmark dapat diubah di file `config.py`:

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `VIDEO_PATH` | `"data/video.mp4"` | Path ke video uji |
| `MODEL_VARIANTS` | `["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]` | Daftar varian model yang akan diuji |
| `MODELS_DIR` | `"models/"` | Direktori penyimpanan model |
| `EXPERIMENT_REPEATS` | `3` | Jumlah pengulangan eksperimen per model |
| `WARMUP_FRAMES` | `50` | Jumlah frame warm-up (tidak diukur) |
| `CONFIDENCE_THRESHOLD` | `0.25` | Threshold confidence untuk deteksi |
| `IOU_THRESHOLD` | `0.7` | Threshold IoU untuk prediksi YOLO |
| `IMAGE_SIZE` | `640` | Ukuran input YOLO; `None` berarti memakai perilaku default/native Ultralytics |
| `RESULTS_BASE_DIR` | `"results/"` | Direktori dasar untuk menyimpan hasil |

## Format Output

### Hasil Per Eksperimen (`results/{nama_perangkat}/{timestamp}/`)

Setiap perangkat menghasilkan:

- **`run_{n}/{model}/frames.csv`** — Metrik per frame untuk setiap model dan repeat, dengan kolom:
  - `frame` — Nomor frame
  - `preprocess_ms` — Waktu pre-processing (ms)
  - `predict_ms` — Waktu pemanggilan `model.predict()` Ultralytics, bukan pure low-level inference
  - `postprocess_ms` — Waktu post-processing (ms)
  - `total_ms` — Total waktu pemrosesan frame (ms)
  - `cpu_percent` — Penggunaan CPU (%)
  - `gpu_percent` — Penggunaan GPU (%, atau `not_available` jika tidak tersedia/valid)
  - `ram_percent` — Penggunaan RAM (%)
  - `process_ram_mb` — RAM proses benchmark dalam MB
  - `detections` — Jumlah objek terdeteksi
  - `instant_fps` — FPS instan untuk frame tersebut

- **`summary_per_run.csv`** — Ringkasan per model per repeat, dibuat ulang dari `frames.csv`
- **`summary.csv` / `tabel_ringkasan.csv`** — Ringkasan agregat lintas repeat, menggunakan `throughput_fps_mean` sebagai FPS utama
- **`tabel_bottleneck.csv`** — Breakdown `preprocess_ms`, `predict_ms`, dan `postprocess_ms`

- **`run_{n}/{model}/metadata.json`** — Informasi perangkat, video, model, konfigurasi, run, dan sumber monitoring:
  - Nama dan model perangkat
  - Versi OS, Python, PyTorch, Ultralytics, OpenCV
  - Backend (`cpu`, `mps`, atau `cuda`), CUDA/cuDNN jika tersedia
  - Resolusi, FPS asli, jumlah frame, durasi, dan codec video
  - Parameter benchmark: warm-up, confidence threshold, IoU threshold, image size
  - Nama/path/ukuran file model
  - Repeat/run ID, jumlah frame terukur, dan statistik latency
  - Sumber pengukuran CPU/RAM/GPU/VRAM/temperatur/daya

### Hasil Perbandingan (`results/comparison/`)

- **`comparison.csv`** — Tabel perbandingan metrik kedua perangkat per model
- **`*.png`** — Grafik bar chart perbandingan (FPS, inference time, CPU, GPU, RAM)

## Menjalankan Tests

```bash
pytest tests/
```
# benchmark-yolo
