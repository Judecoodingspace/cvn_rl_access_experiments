"""Pure, physical-time DQI freshness rule; callers control its opt-in use."""
from __future__ import annotations

import numpy as np


def apply_dqi_freshness(raw, age_ms, coherence, t_min_ms, t_max_ms, scale=1.0, enabled=True):
    if (not np.all(np.isfinite([t_min_ms, t_max_ms, scale])) or
            not 0 < float(t_min_ms) <= float(t_max_ms) or scale <= 0):
        raise ValueError('positive ordered validity endpoints and scale required')
    raw, age, rho = np.broadcast_arrays(np.asarray(raw, float), np.asarray(age_ms, float),
                                         np.asarray(coherence, float))
    if not all(np.all(np.isfinite(a)) for a in (raw, age, rho)) or np.any(age < 0):
        raise ValueError('freshness inputs must be finite; age must be nonnegative')
    validity = scale * (t_min_ms + (t_max_ms - t_min_ms) * np.clip(rho, 0, 1))
    would_expire = age > validity
    expired = would_expire if enabled else np.zeros_like(would_expire)
    discount = np.where(expired, 0.0, np.exp(-age / validity)) if enabled else np.ones_like(age)
    return dict(fresh=raw * discount, validity_ms=validity, discount=discount,
                expired=expired, would_expire=would_expire)
