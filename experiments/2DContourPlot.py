import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata

# Load dataset
df = pd.read_csv("Analysis.csv")  # columns: Program_UnexcusedAbsences, Threshold, RMSE

x = df["Program_UnexcusedAbsences"].to_numpy()
y = df["Threshold"].to_numpy()
z = df["RMSE"].to_numpy()

# Create grid for contour plot
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate RMSE values onto grid
Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

# Plot contour heatmap
plt.figure(figsize=(10,8))
contour = plt.contourf(Xi, Yi, Zi, levels=20, cmap="RdYlGn_r")  # green=low RMSE, red=high
plt.colorbar(contour, label="RMSE")

# Overlay scatter points (actual data)
plt.scatter(x, y, c="black", s=40, label="Observed Data")

# Highlight absolute minimum RMSE point
best_idx = np.argmin(z)
best_absences = x[best_idx]
best_threshold = y[best_idx]
best_rmse = z[best_idx]

plt.scatter(best_absences, best_threshold, c="gold", s=200, marker="*", label="Lowest RMSE")

# Annotate values directly on the plot
plt.text(best_absences, best_threshold,
         f"Absences={best_absences}\nThreshold={best_threshold}\nRMSE={best_rmse:.3f}",
         fontsize=10, color="black", ha="left", va="bottom",
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

plt.xlabel("Program_UnexcusedAbsences")
plt.ylabel("Threshold")
plt.title("RMSE Landscape with Observed Data and Best Candidate Highlight")
plt.legend()
plt.tight_layout()
plt.show()
