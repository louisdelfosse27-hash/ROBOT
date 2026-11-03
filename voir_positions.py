import pandas as pd
import matplotlib.pyplot as plt

p = pd.read_csv("data/poses_ajustees.csv")
plt.scatter(p.x_m, p.y_m, s=80, color="tab:red")
for _, r in p.iterrows():
    plt.annotate(r.fichier.replace(".csv", ""), (r.x_m, r.y_m), xytext=(6, 6),
                 textcoords="offset points")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()