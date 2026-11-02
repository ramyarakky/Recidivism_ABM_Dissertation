import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load dataset
df = pd.read_csv("Analysis.csv")

x = df["Program_UnexcusedAbsences"].to_numpy()
y = df["Threshold"].to_numpy()
z = df["RMSE"].to_numpy()

# Find lowest RMSE
best_idx = np.argmin(z)
best_x, best_y, best_z = x[best_idx], y[best_idx], z[best_idx]

print("Lowest RMSE candidate:")
print(f"Absences={best_x}, Threshold={best_y}, RMSE={best_z}")

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Scatter all points
sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=50)

# Highlight lowest RMSE point with a large red star
ax.scatter(best_x, best_y, best_z,
           color="red", s=600, marker="*", edgecolor="black", label="Lowest RMSE")

# Annotate values
ax.text(best_x, best_y, best_z,
        f"Abs={best_x:.3f}\nThr={best_y:.3f}\nRMSE={best_z:.4f}",
        fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

ax.set_xlabel("Program_UnexcusedAbsences")
ax.set_ylabel("Threshold")
ax.set_zlabel("RMSE")
ax.set_title("Effect of Program Absences and Threshold on RMSE")

fig.colorbar(sc, ax=ax, label="RMSE")
ax.legend()
plt.show()
