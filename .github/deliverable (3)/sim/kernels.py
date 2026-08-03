"""
CUDA kernel for the carrier rate equations (eq. 6-7).

Isolated so the (long, verbatim-C) kernel source does not sit in the middle of
the Python solver. Imports cupy only.
"""

import cupy as cp

# ================================================================================
#  3.  CUDA KERNEL -- plasma rate equation (Chimier 2011 corrected)
# ================================================================================
_RATE_KERNEL_SRC = r'''
__device__ __forceinline__ double interpolate_pi_rate(
    double I, const double* table, int size, double log_min, double inv_log_step, double Imin, double Imax)
{
    if (!isfinite(I) || I < 0.0) I = 0.0;
    if (I < Imin) I = Imin;
    if (I > Imax) I = Imax;
    double fidx = (log10(I) - log_min) * inv_log_step;
    int idx = (int)fidx;
    if (idx < 0) idx = 0;
    if (idx > size - 2) idx = size - 2;
    double r = table[idx] + (fidx - (double)idx) * (table[idx+1] - table[idx]);
    return (!isfinite(r) || r < 0.0) ? 0.0 : r;
}

__device__ __forceinline__ double exact_exp_step(double x, double S, double L, double dt)
{
    double Ldt = L * dt;
    double phi1 = (fabs(Ldt) > 1e-6) ? (exp(Ldt) - 1.0) / Ldt : 1.0 + Ldt * (0.5 + Ldt * (1.0/6.0 + Ldt / 24.0));
    double x_new = exp(Ldt) * x + S * dt * phi1;
    return (!isfinite(x_new) || x_new < 0.0) ? 0.0 : x_new;
}

extern "C" __global__
void solve_rate_equation_kernel(
    const double2* __restrict__ E_field,
    double* __restrict__ ne, double* __restrict__ ns,
    const int Nr, const int Nt, const double dt,
    const double field_to_I,
    const double beta_g, const double beta_s,
    const double na, const double inv_tau_r, const double inv_tau_ste,
    const int enable_ste,
    const double* __restrict__ W_PI_val, const double* __restrict__ W_PI_ste,
    const int table_size, const double log_min_I, const double inv_log_step, const double Imin, const double Imax)
{
    int ri = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (ri >= Nr) return;
    int base = ri * Nt;

    double ne_val = ne[base];
    double ns_val = ns[base];
    if (!isfinite(ne_val) || ne_val < 0.0) ne_val = 0.0;
    if (!isfinite(ns_val) || ns_val < 0.0) ns_val = 0.0;

    double2 f0 = E_field[base];
    double I_c = (f0.x*f0.x + f0.y*f0.y) * field_to_I;
    double W_c = interpolate_pi_rate(I_c, W_PI_val, table_size, log_min_I, inv_log_step, Imin, Imax);
    double Ws_c = enable_ste ? interpolate_pi_rate(I_c, W_PI_ste, table_size, log_min_I, inv_log_step, Imin, Imax) : 0.0;

    for (int step = 0; step < Nt - 1; ++step)
    {
        double2 f1 = E_field[base + step + 1];
        double I_n = (f1.x*f1.x + f1.y*f1.y) * field_to_I;
        double W_n = interpolate_pi_rate(I_n, W_PI_val, table_size, log_min_I, inv_log_step, Imin, Imax);
        double Ws_n = enable_ste ? interpolate_pi_rate(I_n, W_PI_ste, table_size, log_min_I, inv_log_step, Imin, Imax) : 0.0;

        double I_avg = 0.5 * (I_c + I_n);
        double W_avg = 0.5 * (W_c + W_n);
        double Ws_avg = 0.5 * (Ws_c + Ws_n);

        double depl = 1.0 - ne_val / na;
        if (depl < 0.0) depl = 0.0;
        if (depl > 1.0) depl = 1.0;

        double S_e = W_avg * depl + (enable_ste ? (Ws_avg + beta_s * I_avg * ne_val) * (ns_val / na) : 0.0);
        double L_e = beta_g * I_avg * depl - inv_tau_r;

        double S_s = enable_ste ? inv_tau_r * ne_val : 0.0;
        // STE loss: laser re-ionization back to the conduction band (Mao et al.,
        // Appl. Phys. A 79, 1695 (2004): -sigma_x N_STE I^m_x, here generalized
        // to a Keldysh rate at the STE gap Us) plus non-radiative decay to the
        // ground state with time tau_ste (Sakurai et al. tabulate 1 ps for
        // fused silica; inv_tau_ste = 0 disables it, the previous behaviour).
        double L_s = enable_ste ? -(Ws_avg + beta_s * I_avg * ne_val) / na - inv_tau_ste : 0.0;

        double ne_new = exact_exp_step(ne_val, S_e, L_e, dt);
        double ns_new = enable_ste ? exact_exp_step(ns_val, S_s, L_s, dt) : 0.0;

        if (ne_new + ns_new > na) {
            double scale = na / (ne_new + ns_new);
            ne_new *= scale;
            ns_new *= scale;
        }

        ne[base + step + 1] = ne_new;
        ns[base + step + 1] = ns_new;

        ne_val = ne_new;
        ns_val = ns_new;
        I_c = I_n;
        W_c = W_n;
        Ws_c = Ws_n;
    }
}
'''
rate_eq_kernel = cp.RawKernel(_RATE_KERNEL_SRC, 'solve_rate_equation_kernel')


