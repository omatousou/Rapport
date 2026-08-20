import numpy as np
from scipy.constants import c, epsilon_0, m_e, hbar
from scipy.constants import elementary_charge as q_e

# --- inputs identical to the notebook run -----------------------------------
lam, n2, n0 = 1030e-9, 2.74e-20, 1.4498   # n0 from Sellmeier, refined below
tau_c, meff_drude_rel, E_tr_eV = 1.7e-15, 1.0, 4.2

import sys; sys.path.insert(0, "/home/user/Rapport/.github/marche très bien/sim")
sys.path.insert(0, "/tmp/claude-0/-home-user-Rapport/faca0213-b3bc-5328-a268-d0ed76b3a490/scratchpad/verif/stub")
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
print("          w0/(2 n0 rho_c) f_STE   = %.6e   <-- what the docstring says" % (w0f*f_ste/(2*n0*rho_c_pump)))
print("          ratio code/docstring    = %.6e  = 1/c ? %.6e" % (ste_code/(w0f*f_ste/(2*n0*rho_c_pump)), 1/c))
print("          dn_STE at rho_s=1e20    = %+.3e" % (f_ste*1e20/(2*n0*rho_c_pump)))
