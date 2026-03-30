import matplotlib.pyplot as plt

# DATA KAMU (isi sesuai hasil)
labels = ["CPU (Docker)", "MPS (GPU)"]

fps = [10.65, 50.96]
cpu = [31.79, 30.06]
ram = [11.67, 65.52]

# =========================
# FPS COMPARISON
# =========================
plt.figure()
plt.bar(labels, fps)
plt.title("Perbandingan FPS")
plt.ylabel("FPS")
plt.savefig("results/fps_compare.png")
plt.close()

# =========================
# CPU COMPARISON
# =========================
plt.figure()
plt.bar(labels, cpu)
plt.title("Perbandingan CPU Usage")
plt.ylabel("CPU (%)")
plt.savefig("results/cpu_compare.png")
plt.close()

# =========================
# RAM COMPARISON
# =========================
plt.figure()
plt.bar(labels, ram)
plt.title("Perbandingan RAM Usage")
plt.ylabel("RAM (%)")
plt.savefig("results/ram_compare.png")
plt.close()

print("Grafik perbandingan berhasil dibuat!")
