import pandas as pd
import matplotlib.pyplot as plt

import gridmap
from gridmap import nouvelle_grille, scan_local, mettre_a_jour

poses = pd.read_csv("data/poses_ajustees.csv")
grille = nouvelle_grille()

for _, r in poses.iterrows():
    df = pd.read_csv(f"data/{r.fichier}")
    for sid in sorted(df.scan_id.unique()):          # tous les tours renforcent la carte
        mettre_a_jour(grille, scan_local(df[df.scan_id == sid]),
                      (r.x_m, r.y_m, r.theta_deg))

h = gridmap.TAILLE_M / 2
plt.imshow(grille, cmap="gray_r", vmin=0, vmax=255, origin="lower", extent=[-h, h, -h, h])
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Grid map globale")
plt.show()