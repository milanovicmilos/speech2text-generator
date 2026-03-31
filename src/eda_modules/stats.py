from __future__ import annotations

from typing import Dict

import numpy as np


def bonferroni_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    """Apply Bonferroni correction on a dictionary of p-values.

    Invalid and NaN p-values are ignored for multiplicity counting and
    propagated as NaN in the output.
    """
    finite_keys = [key for key, value in p_values.items() if value is not None and np.isfinite(value)]
    m = max(1, len(finite_keys))

    adjusted: Dict[str, float] = {}
    for key, value in p_values.items():
        if value is None or not np.isfinite(value):
            adjusted[key] = np.nan
            continue
        adjusted[key] = float(min(1.0, value * m))
    return adjusted
