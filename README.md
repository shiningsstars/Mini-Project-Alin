# Mini-Project-Alin: Regresi Linear dengan Metode Least Squares

Proyek ini mengimplementasikan regresi linear sederhana menggunakan metode **Least Squares (Kuadrat Terkecil)** berbasis operasi matriks dengan NumPy.

---

## Deskripsi

Model dibangun untuk:
1. **Membangkitkan data acak** — 20 titik data dengan noise Gaussian (μ=0, σ=0.5) dari fungsi linear `y = 2 + 3x`.
2. **Menyusun matriks desain A** dan menyelesaikan **Persamaan Normal** (`Aᵀ·A·x̂ = Aᵀ·b`) secara manual.
3. **Mengevaluasi model** menggunakan SSE (Sum of Squared Errors) dan R² (Koefisien Determinasi).
4. **Memvisualisasikan hasil** dalam bentuk grafik regresi, residual plot, dan tabel data.

---

## Struktur File

```
Mini-Project-Alin/
├── least_squares_model.py   # Script utama: komputasi & visualisasi
├── requirements.txt         # Dependensi Python
├── analisis gabungan.png    # Output: gabungan 3 subplot
├── plot regresi.png         # Output: scatter plot + garis regresi
├── tabel validasi.png       # Output: tabel data lengkap
└── plot residu.png          # Output: residual plot
```

---

## Cara Menjalankan

### 1. Clone / download repositori ini

```bash
git clone https://github.com/username/Mini-Project-Alin.git
cd Mini-Project-Alin
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan script

```bash
python least_squares_model.py
```

Output akan dicetak di terminal dan 4 file gambar tersimpan di direktori yang sama.

---

## Dependensi

| Library      | Kegunaan                         |
|--------------|----------------------------------|
| `numpy`      | Operasi matriks & aljabar linear |
| `matplotlib` | Visualisasi grafik & tabel       |

---

## Metode: Persamaan Normal

Parameter regresi (β₀ dan β₁) dihitung dengan rumus:

```
x̂ = (AᵀA)⁻¹ · Aᵀb
```

di mana:
- **A** — Matriks desain berukuran 20×2 (kolom 1 berisi angka 1, kolom 2 berisi nilai x)
- **b** — Vektor target berukuran 20×1 (nilai y dengan noise)
- **x̂** — Vektor parameter [β₀, β₁]

---

## Metrik Evaluasi

| Metrik | Formula     | Keterangan                                               |
|--------|-------------|----------------------------------------------------------|
| SSE    | Σ(yᵢ - ŷᵢ)² | Total kuadrat error                                      |
| TSS    | Σ(yᵢ - ȳ)²  | Total variasi data                                       |
| R²     | 1 - SSE/TSS | Koefisien determinasi (semakin mendekati 1 semakin baik) |

---

## Contoh Output Terminal

```
=== HASIL PRINT OUT ===

--- HASIL NOMOR 2.1 ---
===================================================================
                  TABEL DATA DATASET MINIPROJECT 3
===================================================================
 No.      |    Koordinat X (Independen)    |    Koordinat Y (Target)
-------------------------------------------------------------------
 Titik 01 |                          1.00  |                   5.15
 ...
--- HASIL NOMOR 2.2 ---
Nilai beta_0 (Intersep) didapatkan = 2.xxxx
Nilai beta_1 (Slope) didapatkan    = 3.xxxx
Persamaan Garis Regresi: y = 2.xx + 3.xxx

--- HASIL NOMOR 2.3 ---
Sum of Squared Errors (SSE)      = x.xxxx
Koefisien Determinasi (R^2)      = 0.xxxx (xx.xx%)
```

---

## Catatan

- `np.random.seed(42)` digunakan agar data acak bersifat **reprodusibel**.
- Semua komputasi dilakukan secara **manual berbasis matriks** tanpa library regresi eksternal.
- Visualisasi disimpan otomatis ke file `.png` di direktori kerja.

---