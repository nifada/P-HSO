"""nimfa 1.4.0 (2019) uses NumPy names removed in NumPy 2.0. Restore them as aliases of
their documented replacements before importing nimfa; nimfa's code itself is unchanged."""
import numpy as np
_ALIASES = {"mat": np.asmatrix, "float": float, "int": int, "bool": bool, "Inf": np.inf,
            "NaN": np.nan, "infty": np.inf, "product": np.prod}
for _n, _v in _ALIASES.items():
    try:
        getattr(np, _n)
    except AttributeError:
        setattr(np, _n, _v)
