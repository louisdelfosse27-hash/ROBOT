"""Détection d'un nouveau tour du lidar à partir des angles reçus.

Utilisé par acquisition.py. Il accumule l'angle parcouru (les angles du LD06 décroissent : 359° → 0°) ; dès qu'un tour complet (360°)
est atteint, il signale un nouveau tour. Cela permet de séparer deux révolutions
même si l'indice des points ne repart pas à zéro.
"""


class DetecteurTour:
    def __init__(self):
        self.reset()

    def reset(self):
        """Repart de zéro (à appeler quand l'indice des points repart plus bas)."""
        self._dernier = None
        self._cumul = 0.0

    def nouveau_tour(self, angle_deg):
        """Renvoie True si cet angle ferme un tour complet depuis le dernier départ."""
        if self._dernier is None:
            self._dernier = angle_deg
            return False
        delta = (self._dernier - angle_deg) % 360.0   # les angles du lidar décroissent
        self._dernier = angle_deg
        if delta > 180.0:          # petit recul (bruit) : ignoré
            return False
        self._cumul += delta
        if self._cumul >= 360.0:
            self._cumul = 0.0
            return True
        return False
