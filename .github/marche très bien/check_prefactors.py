import numpy as np
from scipy.constants import c, epsilon_0, m_e, hbar
from scipy.constants import elementary_charge as q_e

# --- inputs identical to the notebook run -----------------------------------
lam, n2, n0 = 1030e-9, 2.74e-20, 1.4498   # n0 from Sellmeier, refined below
tau_c, meff_drude_rel, E_tr_eV = 1.7e-15, 1.0, 4.2

# keldysh.py is pure numpy/scipy, so this needs neither cupy nor a GPU.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
from keldysh import n_sellmeier
n0 = float(n_sellmeier(lam))

w0f  = 2*np.pi*c/lam            # omega0
k0   = 2*np.pi/lam * n0         # komega
meff_drude = meff_drude_rel*m_e

print(f"n0 = {n0:.6f}   omega0 = {w0f:.6e}   k0 = {k0:.6e}\n")

# ---- 1. KERR -----------------------------------------------------------------
chi3      = 4/3*epsilon_0*n0**2*c*n2                 # config.py:237
kerr_code = 3*chi3*w0f**2/(8*k0*c**2)                # operators.py:66
# independent: du/dz = i (w0/c) n2 I u , with I = 0.5 n0 c eps0 |u|^2
kerr_ref  = (w0f/c)*n2*0.5*n0*c*epsilon_0
print("KERR      code = %.9e" % kerr_code)
print("          ref  = %.9e   (i (w0/c) n2 I u)" % kerr_ref)
print("          ratio= %.12f\n" % (kerr_code/kerr_ref))

# ---- 2. DRUDE cross section --------------------------------------------------
sig_code = ((k0*q_e**2*tau_c)/(n0**2*meff_drude*epsilon_0*w0f*(1+(w0f*tau_c)**2)))*1e4
sig_ref  = (k0*q_e**2*tau_c)/(n0**2*meff_drude*epsilon_0*w0f*(1+(w0f*tau_c)**2))
print("SIGMA_W   code = %.6e cm^2   (SI value %.6e m^2)" % (sig_code, sig_ref))
print("          plasma_pref = sigma/2*100 = %.6e  [m^-1 per cm^-3]" % (sig_code/2*100))
print("          check: units cm^2 * cm^-3 = cm^-1, x100 -> m^-1  OK\n")

# ---- 3. STE index ------------------------------------------------------------
rho_c_pump = (epsilon_0*m_e*w0f**2/q_e**2)*1e-6      # cm^-3   grids.py:133
w_tr  = E_tr_eV*q_e/1.054571817e-34                  # grids.py:134
f_ste = w0f**2/(w_tr**2 - w0f**2)                    # grids.py:135
ste_code = (w0f/c)*f_ste/(2.0*n0*rho_c_pump)         # grids.py:136
print("STE       rho_c(pump) = %.4e cm^-3   f_STE = %.6f" % (rho_c_pump, f_ste))
print("          ste_pref(code)          = %.6e" % ste_code)
# The docstring of run_filament.py used to drop the c and read
# w0/(2 n0 rho_c), which is not even dimensionally homogeneous. Kept here as a
# regression check: the ratio below must stay 1/c, and the docstring must
# carry the c.
print("          same, with c dropped    = %.6e   <-- the old wrong form" % (w0f*f_ste/(2*n0*rho_c_pump)))
print("          ratio                   = %.6e  = 1/c ? %.6e" % (ste_code/(w0f*f_ste/(2*n0*rho_c_pump)), 1/c))
print("          dn_STE at rho_s=1e20    = %+.3e" % (f_ste*1e20/(2*n0*rho_c_pump)))
