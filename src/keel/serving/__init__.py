"""M1 — Zero-config serving.

``keel setup`` gives a newcomer a fast, tuned local server (hardware profile → model/quant
pick with KV headroom → backend select → speed features → benchmark → auto-tune → write one
config). ``keel doctor`` explains any slowness. Wraps mlx-lm, oMLX and MTPLX behind one
interface; our value is hardware-aware selection, tuning and diagnosis, not a new engine.
"""

from __future__ import annotations
