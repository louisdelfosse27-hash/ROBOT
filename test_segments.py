import pytest
from segments import Segment, intersection, couper

A = Segment((0.0, 0.0), (10.0, 0.0))     # segment horizontal de 10 m


def test_intersection_simple():
    b = Segment((3.0, -1.0), (3.0, 1.0))
    pt, t, u = intersection(A, b)
    assert pt == pytest.approx((3.0, 0.0))
    assert t == pytest.approx(0.3)
    assert u == pytest.approx(0.5)


def test_pas_d_intersection_segment_trop_court():
    b = Segment((3.0, 1.0), (3.0, 2.0))   # la droite coupe A, mais pas le segment b
    assert intersection(A, b) is None
    assert couper(A, b) == A


def test_parallele():
    b = Segment((0.0, 1.0), (10.0, 1.0))
    assert intersection(A, b) is None


def test_decoupe_garde_le_plus_long():
    b = Segment((3.0, -1.0), (3.0, 1.0))
    r = couper(A, b)
    assert r.p1 == pytest.approx((3.0, 0.0))
    assert r.p2 == pytest.approx((10.0, 0.0))          # côté de 7 m gardé


def test_intersection_a_une_extremite_de_a():
    assert couper(A, Segment((0.0, -1.0), (0.0, 1.0))) == A     # au début
    assert couper(A, Segment((10.0, -1.0), (10.0, 1.0))) == A   # à la fin


def test_b_se_termine_sur_a_coupe_quand_meme():
    b = Segment((3.0, 0.0), (3.0, 5.0))   # extrémité de b posée à l'intérieur de A
    r = couper(A, b)
    assert r.p1 == pytest.approx((3.0, 0.0))
    assert r.p2 == pytest.approx((10.0, 0.0))


def test_colineaires_pas_decoupes():
    b = Segment((5.0, 0.0), (15.0, 0.0))
    assert couper(A, b) == A


def test_longueurs_egales_garde_le_premier_point():
    b = Segment((5.0, -1.0), (5.0, 1.0))
    r = couper(A, b)
    assert r.p1 == pytest.approx((0.0, 0.0))
    assert r.p2 == pytest.approx((5.0, 0.0))


def test_longueurs_egales_segment_inverse():
    a_inverse = Segment((10.0, 0.0), (0.0, 0.0))        # le "premier point" est maintenant (10, 0)
    b = Segment((5.0, -1.0), (5.0, 1.0))
    r = couper(a_inverse, b)
    assert r.p1 == pytest.approx((10.0, 0.0))
    assert r.p2 == pytest.approx((5.0, 0.0))


def test_longueurs_egales_a_la_tolerance_pres():
    b = Segment((5.00000001, -1.0), (5.00000001, 1.0))  # différence ~2e-8 m < TOL_LONGUEUR
    r = couper(A, b)
    assert r.p1 == pytest.approx((0.0, 0.0))            # on garde bien le côté du premier point