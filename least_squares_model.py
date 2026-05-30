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
# VISUALISASI DATA 
# =====================================================================
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data Asli (with noise)')
plt.plot(x, beta_0_hat + beta_1_hat * x, color='red', linewidth=2, label=f'Garis Regresi: y={beta_0_hat:.2f}+{beta_1_hat:.2f}x')
for i in range(m):
    plt.vlines(x[i], y[i], y_pred[i], colors='green', linestyles='dashed', alpha=0.5) # Garis Residu
plt.title('Visualisasi Regresi Linear & Residu (Least Squares)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()