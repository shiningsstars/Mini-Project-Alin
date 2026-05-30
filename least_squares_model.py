import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# BACKEND & LOGIKA KOMPUTASI
# =====================================================================

# Skenario 1: Menentukan Seed agar data acak bersifat konsisten saat di-print
np.random.seed(42) 

# 2.1. Membangkitkan 20 Data Acak (Data Generation)
m = 20
x = np.linspace(1, 10, m) # Mengambil 20 titik x dari rentang 1 sampai 10
noise = np.random.normal(0, 0.5, m) # miu=0, sigma=0.5 (varians = 0.5^2)
y = 2 + 3 * x + noise

# 2.2. Pembentukan Matriks Desain A dan Vektor b
A = np.vstack([np.ones(m), x]).T  # Kolom 1 isi angka 1 (intersep), Kolom 2 isi nilai x
b = y.reshape(-1, 1)              # Vektor b dari nilai y raksasa

# Komputasi Persamaan Normal (A^T * A * x_hat = A^T * b)
ATA = A.T @ A
ATb = A.T @ b

# Menyelesaikan SPL untuk mencari parameter x_hat (beta_0 dan beta_1)
x_hat = np.linalg.inv(ATA) @ ATb
beta_0_hat = x_hat[0][0]
beta_1_hat = x_hat[1][0]

# 2.3. Evaluasi Model (Metrik Kesalahan)
y_pred = A @ x_hat                 # Menghitung tebakan model (y topi)
y_mean = np.mean(y)                # Rata-rata dari data y asli

r = b - y_pred                     # Vektor residu / galat
SSE = np.sum(r**2)                 # Sum of Squared Errors
TSS = np.sum((b - y_mean)**2)      # Total Sum of Squares
R_squared = 1 - (SSE / TSS)        # Koefisien Determinasi R^2

# =====================================================================
# PRINT OUT 
# =====================================================================
print("=== HASIL PRINT UNTUK LAPORAN TANIA ===\n")
print(f"--- NOMOR 2.1 ---")
print("Data X:", np.round(x, 2))
print("Data Y:", np.round(y, 2))

print(f"\n--- NOMOR 2.2 ---")
print("Matriks A (Ukuran 20x2):\n", np.round(A, 2))
print("\nVektor b (Ukuran 20x1):\n", np.round(b, 2))
print("\nMatriks Normal A^T * A:\n", np.round(ATA, 2))
print("\nVektor A^T * b:\n", np.round(ATb, 2))
print(f"\nNilai beta_0 (Intersep) didapatkan = {beta_0_hat:.4f}")
print(f"Nilai beta_1 (Slope) didapatkan    = {beta_1_hat:.4f}")
print(f"Persamaan Garis Regresi: y = {beta_0_hat:.2f} + {beta_1_hat:.2f}x")

print(f"\n--- NOMOR 2.3 ---")
print(f"Sum of Squared Errors (SSE)      = {SSE:.4f}")
print(f"Total Sum of Squares (TSS)       = {TSS:.4f}")
print(f"Koefisien Determinasi (R^2)      = {R_squared:.4f} ({R_squared*100:.2f}%)")

# =====================================================================
# VISUALISASI GRAFIK DAN TABEL
# =====================================================================

# FIG 1: ANALISIS GABUNGAN
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(x, y, color='blue', label='Data Asli')
ax1.plot(x, beta_0_hat + beta_1_hat * x, color='red', linewidth=2, label='Garis Regresi')
for i in range(m):
    ax1.vlines(x[i], y[i], y_pred[i][0], colors='green', linestyles='dashed', alpha=0.5)
ax1.set_title('1. Regresi Linear & Residu', fontsize=12, pad=15)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

residu_flat = r.flatten()
ax2.scatter(x, residu_flat, color='purple', edgecolors='black', s=50, label='Residu ($e_i$)')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
for i in range(m):
    ax2.vlines(x[i], 0, residu_flat[i], colors='purple', linestyles='dotted', alpha=0.5)
ax2.set_title('2. Residual Plot (Ringkas)', fontsize=12, pad=15)
ax2.set_xlabel('X')
ax2.set_ylabel('Nilai Eror')
ax2.set_ylim(-1.5, 1.5)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

ax3.axis('off') 
tabel_data = []
for i in range(m):
    tabel_data.append([i+1, f"{x[i]:.2f}", f"{y[i].item():.2f}", f"{y_pred[i].item():.2f}", f"{r[i].item():.2f}"])
kolom_header = ['No', 'X', 'Y Asli', 'Prediksi', 'Residu']
tabel_fitur = ax3.table(cellText=tabel_data, colLabels=kolom_header, loc='center', cellLoc='center')
tabel_fitur.scale(0.9, 1.05)
ax3.set_title('3. Tabel Data', fontsize=12, y=1.02, fontweight='bold')

fig1.tight_layout()
fig1.savefig('analisis gabungan.png', dpi=300)


# FIG 2: PLOT REGRESI LINEAR
fig2, ax_regresi = plt.subplots(figsize=(8, 5))
ax_regresi.scatter(x, y, color='blue', label='Data Asli (with noise)')
ax_regresi.plot(x, beta_0_hat + beta_1_hat * x, color='red', linewidth=2, label=f'Garis Regresi: y={beta_0_hat:.2f}+{beta_1_hat:.2f}x')
for i in range(m):
    ax_regresi.vlines(x[i], y[i], y_pred[i][0], colors='green', linestyles='dashed', alpha=0.5)
ax_regresi.set_title('Visualisasi Regresi Linear & Residu', fontsize=14, pad=15)
ax_regresi.set_xlabel('X', fontsize=12)
ax_regresi.set_ylabel('Y', fontsize=12)
ax_regresi.legend()
ax_regresi.grid(True, linestyle='--', alpha=0.6)
fig2.tight_layout()
fig2.savefig('plot regresi.png', dpi=300)


# FIG 3: TABEL VALIDASI DATA 
fig3, ax_tabel = plt.subplots(figsize=(7, 6))
ax_tabel.axis('off')
tabel_fitur_tunggal = ax_tabel.table(cellText=tabel_data, colLabels=kolom_header, loc='center', cellLoc='center')
tabel_fitur_tunggal.scale(1.0, 1.2)
ax_tabel.set_title('Tabel Data (No, X, Y Asli, Prediksi Y, Residu)', fontsize=12, fontweight='bold', y=0.95)
fig3.tight_layout()
fig3.savefig('tabel validasi.png', dpi=300)


# FIG 4: PLOT RESIDU / ERROR
fig4, ax_residu = plt.subplots(figsize=(8, 5))
ax_residu.scatter(x, residu_flat, color='purple', edgecolors='black', s=70, label='Residu ($e_i$)')
ax_residu.axhline(y=0, color='black', linestyle='-', linewidth=2)
for i in range(m):
    ax_residu.vlines(x[i], 0, residu_flat[i], colors='purple', linestyles='--', alpha=0.6)
ax_residu.set_title('Visualisasi Error / Residu', fontsize=14, pad=15)
ax_residu.set_xlabel('Nilai X', fontsize=12)
ax_residu.set_ylabel('Nilai Error', fontsize=12)
ax_residu.set_ylim(-1.5, 1.5)
ax_residu.legend()
ax_residu.grid(True, linestyle='--', alpha=0.6)
fig4.tight_layout()
fig4.savefig('plot residu.png', dpi=300)

plt.show()