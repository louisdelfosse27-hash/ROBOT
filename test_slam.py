import numpy as np
import pytest

from slam import composer, relatif, odometrie_bruitee, odometrie_seule, erreurs, ecart_angle
import simulation as sim


def test_composer_est_l_inverse_de_relatif():
    p, q = (1.0, 2.0, 30.0), (-3.0, 0.5, -80.0)
    r = composer(p, relatif(p, q))
    assert r[0] == pytest.approx(q[0]) and r[1] == pytest.approx(q[1])
    assert ecart_angle(r[2], q[2]) == pytest.approx(0.0)


def test_relatif_dans_le_repere_du_robot():
    # robot tourné de 90° : avancer de 1 m dans son axe (x robot) = +1 m en y dans la salle
    d = relatif((0.0, 0.0, 90.0), (0.0, 1.0, 90.0))
    assert d == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_odometrie_sans_bruit_redonne_la_trajectoire():
    refs = [(0.0, 0.0, 0.0), (1.0, 0.5, 20.0), (2.0, 2.0, 170.0), (0.0, 2.0, -170.0)]
    odo = odometrie_bruitee(refs, np.random.default_rng(0), sig_fixe=0, sig_par_m=0, sig_th=0)
    est = odometrie_seule(refs[0], odo)
    assert all(e_pos < 1e-9 and e_ang < 1e-9 for e_pos, e_ang in erreurs(est, refs))


def test_sigma_croit_avec_la_distance():
    from slam import sigma_pas, SIG_ODO_FIXE
    assert sigma_pas((0.0, 0.0, 0.0)) == pytest.approx(SIG_ODO_FIXE)
    assert sigma_pas((3.0, 4.0, 10.0)) > sigma_pas((0.3, 0.4, 10.0))


def test_a_priori_ramene_vers_l_odometrie_quand_le_score_est_plat():
    """Grille uniforme (aucune information) : sans a priori le résultat est arbitraire,
    avec a priori on doit retrouver la pose prédite."""
    from gridmap import nouvelle_grille
    from slam import localiser_multires_prior
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    pose, _ = localiser_multires_prior(nouvelle_grille(), pts, (0.0, 0.0, 0.0), sigma=0.05)
    assert np.hypot(pose[0], pose[1]) < 0.02


def test_ecart_angle_passe_par_180():
    assert ecart_angle(179.0, -179.0) == pytest.approx(-2.0)


def test_scan_simule_dans_la_salle():
    murs = sim.murs_du_monde()
    pts = sim.scan_simule((-2.0, 0.0, 0.0), murs, np.random.default_rng(0))
    assert len(pts) > 0.9 * sim.N_FAISCEAUX * (1 - sim.P_PERTE)
    # le point le plus éloigné ne dépasse pas la diagonale de la salle
    assert np.linalg.norm(pts, axis=1).max() < 16.0
