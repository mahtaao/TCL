import numpy as np
from pys import PyS

# Initialize PyS
pys = PyS()

# Fetch epo_I
epo_I = pys.fetch_epo("I")

# Assuming ts is a predefined variable
ts = 100
raw_I = np.random.rand(128, ts)  # np.array() (128 x ts)

# Solve inverse problem
VOL_INV_OPT_I, VOL_PARC_MTRX = pys.solve_inverse_prob(epo_I, mode='vol')

# or fetch inv opt
# VOL_INV_OPT_I, VOL_PARC_MTRX = pys.fetch_inv_opt("I", mode='vol')

# Assuming raw_i is a predefined variable
raw_i = np.random.rand(128, ts)
src_lvl_i = np.dot(VOL_INV_OPT_I, raw_i)
label_lvl_i = np.dot(VOL_PARC_MTRX, src_lvl_i)

# Metrics
#---------
# Apply linear operations before inverse operator & non-linear after

# Assuming lin_fnc is a predefined function
def lin_fnc(x):
    return x * 2

# Linear
lin_raw_i = lin_fnc(raw_i)
src_lvl_i = np.dot(VOL_INV_OPT_I, lin_raw_i)
label_lvl_i = np.dot(VOL_PARC_MTRX, src_lvl_i)

# Assuming non_lin_fnc is a predefined function
def non_lin_fnc(x):
    return x ** 2

# Non-Linear
non_lin_label_lvl_i = non_lin_fnc(label_lvl_i)