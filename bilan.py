import pandas as pd

SOURCES = {"grid map": "data/resultats_grid.csv",
           "scan contre scan": "data/resultats_scan_scan.csv",
           "carte vectorielle": "data/resultats_vecto.csv"}
EPS = 1e-9   # les erreurs du CSV sont arrondies : on compte "<=" le seuil, pas "<"

lignes = []
for nom, chemin in SOURCES.items():
    r = pd.read_csv(chemin)
    r = r[r.dx.abs() < 0.8]        # on écarte le décalage de 0,8 m : hors de la fenêtre de recherche
    for methode, g in r.groupby("methode"):
        lignes.append(dict(
            reference=nom, methode=methode, n=len(g),
            err_pos_cm=round(g.err_pos_m.mean() * 100, 1),
            err_ang_deg=round(g.err_ang_deg.mean(), 2),
            temps_s=round(g.temps_s.mean(), 2),
            ok_5cm=round(((g.err_pos_m <= 0.05 + EPS) & (g.err_ang_deg < 2)).mean() * 100),
            ok_10cm=round(((g.err_pos_m <= 0.10 + EPS) & (g.err_ang_deg < 2)).mean() * 100)))

bilan = pd.DataFrame(lignes)
print(bilan.to_string(index=False))
bilan.to_csv("data/bilan_localisation.csv", index=False)