import numpy as np
import pytest
from cvn_rl.simulation.dqi_freshness import apply_dqi_freshness


def test_freshness_boundary_and_off():
    result = apply_dqi_freshness(.8, np.array([0, 50, 100, 101]), 0, 100, 1000)
    assert np.allclose(result['fresh'], [.8, .8*np.exp(-.5), .8/np.e, 0])
    assert result['expired'].tolist() == [False, False, False, True]
    off = apply_dqi_freshness(.8, [0, 10000], 0, 100, 1000, enabled=False)
    assert np.all(off['fresh'] == .8)
    assert not np.any(off['expired']) and off['would_expire'][-1]


def test_scale_and_coherence_are_independent_of_age():
    base = apply_dqi_freshness(.7, 50, [.1, .9], 100, 1000)
    half = apply_dqi_freshness(.7, 50, [.1, .9], 100, 1000, scale=.5)
    assert np.allclose(half['validity_ms'], base['validity_ms']/2)
    assert base['fresh'][1] > base['fresh'][0]
    with pytest.raises(ValueError):
        apply_dqi_freshness(.5, -1, .5, 100, 1000)
