# diagnose.py - run this standalone to check transition counts
# python diagnose.py

import sys, os
import numpy as np
import pandas as pd
import inspect

print("="*60)
print("FILE VERSIONS")
print("="*60)
import models.hmm as hmm_mod
import validate_hmm as vh_mod
print(f"hmm.py:          {inspect.getfile(hmm_mod)}")
print(f"validate_hmm.py: {inspect.getfile(vh_mod)}")

# Check for majority vote code
vh_src = inspect.getsource(vh_mod)
print(f"\nmajority-vote in validate_hmm: {'scipy_mode' in vh_src or 'PLOT_SMOOTH' in vh_src}")
print(f"transition count print:        {'n_transitions' in vh_src}")

hmm_src = inspect.getsource(hmm_mod.RegimeHMM.predict)
print(f"rolling mean in predict():     {'rolling' in hmm_src}")
print(f"min_persistence param:         {'min_persistence' in hmm_src}")

print("\n" + "="*60)
print("CURRENT CONFIG")
print("="*60)
import config
print(f"HMM_MIN_PERSISTENCE: {config.HMM_MIN_PERSISTENCE}")
print(f"HMM_N_RESTARTS:      {config.HMM_N_RESTARTS}")
print(f"HMM_COVARIANCE:      {config.HMM_COVARIANCE}")
