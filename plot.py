import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("results/performance.csv")

# =========================
# FPS PER FRAME
# =========================
df["fps"] = 1 / df["inference"]

# =========================
# PLOT FPS
# =========================
plt.figure()
plt.plot(df["fps"])
plt.title("FPS per Frame")
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.savefig("results/fps_plot.png")
plt.close()

# =========================
# CPU USAGE
# =========================
plt.figure()
plt.plot(df["cpu"])
plt.title("CPU Usage (%)")
plt.xlabel("Frame")
plt.ylabel("CPU %")
plt.savefig("results/cpu_plot.png")
plt.close()

# =========================
# RAM USAGE
# =========================
plt.figure()
plt.plot(df["ram"])
plt.title("RAM Usage (%)")
plt.xlabel("Frame")
plt.ylabel("RAM %")
plt.savefig("results/ram_plot.png")
plt.close()

print("Grafik berhasil dibuat di folder results/")
