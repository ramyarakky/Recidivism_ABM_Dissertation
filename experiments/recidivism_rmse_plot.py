import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load your dataset
df = pd.read_csv("Cal.csv")

# -------------------------------
# Find lowest RMSE candidate
# -------------------------------
best_idx = df["RMSE"].idxmin()
best_row = df.loc[best_idx]

print("Lowest RMSE candidate:")
print(best_row[["Iteration","Program_UnexcusedAbsences","Threshold","Recid_3yr","Recid_6yr","Recid_9yr","RMSE"]])

# -------------------------------
# 1. Scatter plots: Absences vs RMSE, Threshold vs RMSE
# -------------------------------
plt.figure(figsize=(12,5))

# Absences vs RMSE
plt.subplot(1,2,1)
sns.scatterplot(x="Program_UnexcusedAbsences", y="RMSE", data=df, hue="Recid_3yr", palette="viridis")
plt.scatter(best_row["Program_UnexcusedAbsences"], best_row["RMSE"],
            color="gold", s=200, marker="*", label="Lowest RMSE")
plt.text(best_row["Program_UnexcusedAbsences"], best_row["RMSE"],
         f"Abs={best_row['Program_UnexcusedAbsences']:.3f}\nRMSE={best_row['RMSE']:.4f}",
         fontsize=9, color="black", ha="left", va="bottom",
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
plt.title("Program Absences vs RMSE (colored by Recid_3yr)")
plt.xlabel("Program_UnexcusedAbsences")
plt.ylabel("RMSE")
plt.legend()

# Threshold vs RMSE
plt.subplot(1,2,2)
sns.scatterplot(x="Threshold", y="RMSE", data=df, hue="Recid_6yr", palette="plasma")
plt.scatter(best_row["Threshold"], best_row["RMSE"],
            color="gold", s=200, marker="*", label="Lowest RMSE")
plt.text(best_row["Threshold"], best_row["RMSE"],
         f"Thr={best_row['Threshold']:.3f}\nRMSE={best_row['RMSE']:.4f}",
         fontsize=9, color="black", ha="left", va="bottom",
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
plt.title("Threshold vs RMSE (colored by Recid_6yr)")
plt.xlabel("Threshold")
plt.ylabel("RMSE")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 2. 3D scatter plot: Absences, Threshold, RMSE
# -------------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

x = df["Program_UnexcusedAbsences"]
y = df["Threshold"]
z = df["RMSE"]

sc = ax.scatter(x, y, z, c=z, cmap="RdYlGn_r", s=50)
ax.scatter(best_row["Program_UnexcusedAbsences"], best_row["Threshold"], best_row["RMSE"],
           color="gold", s=300, marker="*", label="Lowest RMSE")

ax.text(best_row["Program_UnexcusedAbsences"], best_row["Threshold"], best_row["RMSE"],
        f"Abs={best_row['Program_UnexcusedAbsences']:.3f}\nThr={best_row['Threshold']:.3f}\nRMSE={best_row['RMSE']:.4f}",
        fontsize=9, color="black")

ax.set_xlabel("Program_UnexcusedAbsences")
ax.set_ylabel("Threshold")
ax.set_zlabel("RMSE")
ax.set_title("3D Scatter: Absences, Threshold, and RMSE")

fig.colorbar(sc, ax=ax, label="RMSE")
ax.legend()
plt.show()

# -------------------------------
# 3. Heatmaps: Absences vs Threshold colored by Recidivism rates
# -------------------------------
for year in [3,6,9]:
    pivot = df.pivot_table(index="Threshold", columns="Program_UnexcusedAbsences", values=f"Recid_{year}yr")
    plt.figure(figsize=(10,8))
    sns.heatmap(pivot, cmap="coolwarm", cbar_kws={"label": f"Recid_{year}yr"})
    plt.title(f"Heatmap of Recid_{year}yr by Absences and Threshold")
    plt.xlabel("Program_UnexcusedAbsences")
    plt.ylabel("Threshold")
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4. Annotated scatter Absences vs Threshold with RMSE
# -------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x="Program_UnexcusedAbsences", y="Threshold", size="RMSE", data=df,
                sizes=(20,200), hue="RMSE", palette="RdYlGn_r")
plt.scatter(best_row["Program_UnexcusedAbsences"], best_row["Threshold"],
            color="gold", s=300, marker="*", label="Lowest RMSE")
plt.text(best_row["Program_UnexcusedAbsences"], best_row["Threshold"],
         f"Abs={best_row['Program_UnexcusedAbsences']:.3f}\nThr={best_row['Threshold']:.3f}\nRMSE={best_row['RMSE']:.4f}",
         fontsize=9, color="black", ha="left", va="bottom",
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
plt.title("Absences vs Threshold (size/color = RMSE)")
plt.legend()
plt.tight_layout()
plt.show()
