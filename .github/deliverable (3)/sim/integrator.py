"""
Integrator: allocates the recording buffers, marches the split-step scheme in
z, and writes result.npz / params.json.
"""

import os
import sys
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Union

import numpy as np
import cupy as cp
from tqdm.auto import tqdm

from scipy.constants import c, epsilon_0, m_e
from scipy.constants import elementary_charge as q_e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config, code_fingerprint   # noqa: E402
from grids import build_grids, ENVELOPES      # noqa: E402
from operators import LinearOperator, NonlinearOperator  # noqa: E402

# ================================================================================
#  8.  INTEGRATOR
# ================================================================================
class Integrator:
    def __init__(self, cfg: Config, envelope: Union[str, Callable] = "gaussian_focused"):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        g = build_grids(cfg)
        self.g = g
        self.dz, self.dt = cfg.dz, g["dt"]
        self.lin = LinearOperator(cfg, g)
        self.nl  = NonlinearOperator(cfg, g)

        fn = envelope if callable(envelope) else ENVELOPES[envelope]
        self.u = cp.ascontiguousarray(fn(g["rr"], g["tt"], cfg, g).astype(cp.complex128, copy=False))
        self.Nr, self.Nt = self.u.shape
        self.rho   = cp.zeros((self.Nr, self.Nt), dtype=cp.float64)
        self.rho_s = cp.zeros((self.Nr, self.Nt), dtype=cp.float64)
        self.mask_r = g["mask_r"]
        self.threads = 256
        self.blocks  = (self.Nr + self.threads - 1) // self.threads

        fl0 = cp.sum(cp.abs(self.u)**2, axis=1) * self.dt * g["invE2"] * 1e4
        self.U0_uJ = float(cp.sum(fl0 * 2.0 * cp.pi * g["rlist"]
                                  * cp.diff(g["rlist"], prepend=0.0))) * 1e6
        if cfg.verbose:
            print(f"[init] U_beam(0) = {self.U0_uJ:.3f} uJ ", flush=True)

        n_saves = cfg.nz // cfg.save_stride + 1
        self.fluence_rz = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        self.rho_rz     = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        self.rho_s_rz   = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        self.Imax_z     = np.zeros(n_saves, dtype=np.float64)
        self.z_saved    = np.zeros(n_saves, dtype=np.float64)
        # dE/dz (uJ/m) at each saved z, photoionization and plasma-absorption
        # channels kept separate (see NonlinearOperator.loss_rates) -- turned
        # into the cumulative fractional losses of Fig. 12 (Couairon 2005) in
        # _results(). No separate STE loss channel is modeled (the paper's
        # Fig. 12 has none either), so E_STE_z stays 0.
        self._dEdz_photo_uJm  = np.zeros(n_saves, dtype=np.float64)
        self._dEdz_plasma_uJm = np.zeros(n_saves, dtype=np.float64)
        self.E_STE_z    = np.zeros(n_saves, dtype=np.float64)
        self.k_save = 0

        tlist_fs = cp.asnumpy(g['tlist']) * 1e15
        self.t_full_fs = tlist_fs

        self._t_stride = max(1, cfg.rho_t_stride) if cfg.rho_t_stride > 0 else 0
        self._r_stride = max(1, cfg.rho_r_stride)
        if self._t_stride > 0:
            Nt_sub = (self.Nt - 1) // self._t_stride + 1
            Nr_sub = (self.Nr - 1) // self._r_stride + 1
            self.rho_rzt   = np.zeros((n_saves, Nr_sub, Nt_sub), dtype=np.float32)
            self.rho_s_rzt = np.zeros((n_saves, Nr_sub, Nt_sub), dtype=np.float32)
            self.I_rzt     = np.zeros((n_saves, Nr_sub, Nt_sub), dtype=np.float32)
            self.t_sub_fs  = tlist_fs[::self._t_stride]
            self.r_sub     = cp.asnumpy(g["rlist"])[::self._r_stride]
        else:
            self.rho_rzt, self.rho_s_rzt, self.I_rzt = None, None, None
            self.t_sub_fs, self.r_sub = None, None

        # The cube dominates the file size; say so before spending the run.
        cube_b = (3 * self.rho_rzt.nbytes) if self.rho_rzt is not None else 0
        rest_b = 4 * n_saves * (10 * 2 * self.Nr + 3 * self.Nt)
        tot_gb = (cube_b + rest_b) / 1024**3
        if cfg.verbose:
            print(f"[init] result.npz ~ {tot_gb:.2f} GB non compresse"
                  + (f" ({100*cube_b/(cube_b+rest_b):.0f} % = cube (z,r,t))" if cube_b else ""),
                  flush=True)
        if tot_gb > 2.0:
            print(f"[WARN] result.npz estime a {tot_gb:.1f} GB. Le cube (z,r,t) est pilote par "
                  f"rho_t_stride={cfg.rho_t_stride} / rho_r_stride={cfg.rho_r_stride} ; "
                  f"rho_t_stride=0 le desactive completement (les figures de l'article n'en "
                  f"ont pas besoin, elles utilisent les traces on-axis pleine resolution).",
                  flush=True)

        # On-axis (r index 0, closest to the axis), FULL time resolution --
        # independent of rho_t_stride and cheap (no radial dimension, unlike
        # rho_rzt/I_rzt above: ~Nt floats per saved z-plane). This is what
        # figures_article.py's 0D reintegration should read: a rho_t_stride
        # subsampled I_rzt can miss a narrow intensity spike between two
        # saved samples, and since multiphoton rate scales roughly as I^K
        # (K ~ 8-9 photons for a 9 eV gap at 1030 nm), missing that spike can
        # make the reintegrated density look many orders of magnitude below
        # what the CUDA kernel (which ran on the full grid) actually computed.
        # Absorbed energy density (J/cm^3) deposited locally at each (r, z):
        # integral over t of 2*alpha*I, alpha from NonlinearOperator.loss_rates.
        self.absorbed_rz = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        tl_fs = tlist_fs
        if cfg.absorb_time_bins_fs:
            edges = np.asarray(cfg.absorb_time_bins_fs, dtype=np.float64)
            self._absorb_masks = [cp.asarray((tl_fs >= a) & (tl_fs < b))
                                  for a, b in zip(edges[:-1], edges[1:])]
            self.absorb_bin_edges_fs = edges
            self.absorbed_rz_bins = cp.zeros((n_saves, len(self._absorb_masks), self.Nr),
                                             dtype=cp.float32)
        else:
            self._absorb_masks, self.absorb_bin_edges_fs, self.absorbed_rz_bins = None, None, None
        # Peak intensity at each (r, z) -- max over time, per radius (Imax_z is
        # the max over r AND t, a different quantity).
        self.Ipeak_rz = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        # Free-electron density at one instant rather than its max over time.
        if cfg.rho_snapshot_t_fs is not None:
            self._it_snap = int(np.argmin(np.abs(tl_fs - cfg.rho_snapshot_t_fs)))
            self.rho_rz_at = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        else:
            self._it_snap, self.rho_rz_at = None, None

        self.rho_onaxis_t   = np.zeros((n_saves, self.Nt), dtype=np.float32)
        self.rho_s_onaxis_t = np.zeros((n_saves, self.Nt), dtype=np.float32)
        self.I_onaxis_t     = np.zeros((n_saves, self.Nt), dtype=np.float32)

    def step(self, dz):
        u, rho, rho_s = self.u, self.rho, self.rho_s
        u = self.lin.half_linear(u, dz)
        _, a = self.nl.split(u, rho, rho_s); u = u * cp.exp(-0.5 * dz * a)
        k1, _ = self.nl.split(u,               rho, rho_s)
        k2, _ = self.nl.split(u + 0.5 * dz * k1, rho, rho_s)
        k3, _ = self.nl.split(u + 0.5 * dz * k2, rho, rho_s)
        k4, _ = self.nl.split(u +       dz * k3, rho, rho_s)
        u = u + dz / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _, a = self.nl.split(u, rho, rho_s); u = u * cp.exp(-0.5 * dz * a)
        u = self.lin.half_linear(u, dz)
        self.u = u * self.mask_r

    def propagate(self):
        cfg = self.cfg
        pbar = tqdm(range(cfg.nz + 1), desc="Filamentation", unit="step",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for i in pbar:
            self.nl.update_plasma(self.u, self.rho, self.rho_s, self.dt, self.blocks, self.threads)
            self.step(self.dz)
            z_now = cfg.begin + i * self.dz

            if i % cfg.save_stride == 0:
                self._record(z_now)

            if i > 0 and (i % cfg.ckpt_every == 0):
                k = self.k_save - 1
                I_peak = float(self.Imax_z[k])

                flu_cpu = cp.asnumpy(self.fluence_rz[k])
                r_cpu = cp.asnumpy(self.g["rlist"])
                dr_cpu = np.diff(r_cpu, prepend=0.0)
                U_now_uJ = float(np.sum(flu_cpu * 2.0 * np.pi * r_cpu * dr_cpu)) * 100.0
                pct_u = U_now_uJ / self.U0_uJ * 100.0

                pbar.set_postfix(z=f"{z_now*1e6:+.0f}µm", U=f"{pct_u:.1f}%", I_peak=f"{I_peak:.2e}")

        pbar.close()
        return self._results()

    def _record(self, z_now):
        k = self.k_save
        g = self.g
        absu2 = cp.abs(self.u)**2
        I_full = absu2 * g["invE2"]
        self.fluence_rz[k] = (cp.sum(absu2, axis=1) * self.dt * g["invE2"]).astype(cp.float32, copy=False)
        self.rho_rz[k]     = cp.max(self.rho,   axis=1).astype(cp.float32, copy=False)
        self.rho_s_rz[k]   = cp.max(self.rho_s, axis=1).astype(cp.float32, copy=False)
        self.Imax_z[k]     = float(I_full.max())
        self.z_saved[k]    = float(z_now)

        # Fig. 12 (Couairon 2005): dE/dz per channel, integrated over the
        # transverse plane the same way U0_uJ is (fluence x 2*pi*r*dr, cm^2
        # -> m^2, J -> uJ), with an extra 2*alpha_channel(r,t) weight since
        # dI/dz = -2*alpha*I for the field-amplitude decay applied in step().
        photo_al, plasma_al = self.nl.loss_rates(self.u, self.rho)

        # Local absorbed energy density (J/cm^3): dW/dV = int 2 alpha I dt.
        # I_full is W/cm^2 and alpha is 1/m, so 2*alpha*I*dt is J/(cm^2 m);
        # x100 converts the 1/m into 1/cm -> J/cm^3.
        dep = 2.0 * (photo_al + plasma_al) * I_full * self.dt * 100.0
        self.absorbed_rz[k] = cp.sum(dep, axis=1).astype(cp.float32, copy=False)
        if self._absorb_masks is not None:
            for b, m in enumerate(self._absorb_masks):
                self.absorbed_rz_bins[k, b] = cp.sum(dep * m, axis=1).astype(cp.float32, copy=False)
        self.Ipeak_rz[k] = cp.max(I_full, axis=1).astype(cp.float32, copy=False)
        if self._it_snap is not None:
            self.rho_rz_at[k] = self.rho[:, self._it_snap].astype(cp.float32, copy=False)

        r_cpu, dr_cpu = g["rlist"], cp.diff(g["rlist"], prepend=0.0)
        floss_photo  = cp.sum(2.0 * photo_al  * I_full, axis=1) * self.dt
        floss_plasma = cp.sum(2.0 * plasma_al * I_full, axis=1) * self.dt
        self._dEdz_photo_uJm[k]  = float(cp.sum(floss_photo  * 2.0 * cp.pi * r_cpu * dr_cpu)) * 1e4 * 1e6
        self._dEdz_plasma_uJm[k] = float(cp.sum(floss_plasma * 2.0 * cp.pi * r_cpu * dr_cpu)) * 1e4 * 1e6

        if self._t_stride > 0 and self.rho_rzt is not None:
            rs, ts = self._r_stride, self._t_stride
            self.rho_rzt  [k] = cp.asnumpy(self.rho  [::rs, ::ts].astype(cp.float32, copy=False))
            self.rho_s_rzt[k] = cp.asnumpy(self.rho_s[::rs, ::ts].astype(cp.float32, copy=False))
            self.I_rzt    [k] = cp.asnumpy(I_full[::rs, ::ts].astype(cp.float32, copy=False))

        self.rho_onaxis_t  [k] = cp.asnumpy(self.rho  [0].astype(cp.float32, copy=False))
        self.rho_s_onaxis_t[k] = cp.asnumpy(self.rho_s[0].astype(cp.float32, copy=False))
        self.I_onaxis_t    [k] = cp.asnumpy(I_full[0].astype(cp.float32, copy=False))

        self.k_save += 1

    def _cumulative_energy_fraction(self, dEdz_uJm):
        """Trapezoidal cumulative integral of dE/dz (uJ/m) over z_saved,
        normalized by U0_uJ -- the fractional cumulative energy loss curve
        of Fig. 12 (Couairon 2005)."""
        z = self.z_saved[:self.k_save]
        dEdz = dEdz_uJm[:self.k_save]
        if len(z) < 2:
            return np.zeros_like(dEdz)
        seg_uJ = 0.5 * (dEdz[:-1] + dEdz[1:]) * np.diff(z)
        cum_uJ = np.concatenate([[0.0], np.cumsum(seg_uJ)])
        return cum_uJ / self.U0_uJ

    def _results(self):
        cfg, g = self.cfg, self.g
        def _mirror(a):
            return np.hstack([a[:, ::-1], a])
        r_cpu     = cp.asnumpy(g["rlist"])
        flu_cpu   = cp.asnumpy(self.fluence_rz[:self.k_save])
        rho_cpu   = cp.asnumpy(self.rho_rz[:self.k_save])
        rho_s_cpu = cp.asnumpy(self.rho_s_rz[:self.k_save])

        E_MPI_z    = self._cumulative_energy_fraction(self._dEdz_photo_uJm)
        E_plasma_z = self._cumulative_energy_fraction(self._dEdz_plasma_uJm)
        E_total    = E_MPI_z + E_plasma_z

        out = dict(
            r=np.concatenate([-r_cpu[::-1], r_cpu]),
            rlist=r_cpu,
            z=self.z_saved[:self.k_save],
            fluence_rz=np.hstack([flu_cpu[:, ::-1], flu_cpu]),
            rho_rz=np.hstack([rho_cpu[:, ::-1], rho_cpu]),
            rho_s_rz=np.hstack([rho_s_cpu[:, ::-1], rho_s_cpu]),
            Imax_z=self.Imax_z[:self.k_save],
            E_plasma_z=E_plasma_z,
            E_MPI_z=E_MPI_z,
            E_STE_z=self.E_STE_z[:self.k_save],
            E_total_z=E_total,
            rho_rzt=(self.rho_rzt[:self.k_save] if self.rho_rzt is not None else None),
            rho_s_rzt=(self.rho_s_rzt[:self.k_save] if self.rho_s_rzt is not None else None),
            I_rzt=(self.I_rzt[:self.k_save] if self.I_rzt is not None else None),
            t_sub_fs=(self.t_sub_fs if self.t_sub_fs is not None else None),
            r_sub=(self.r_sub if self.r_sub is not None else None),
            absorbed_rz=_mirror(cp.asnumpy(self.absorbed_rz[:self.k_save])),
            Ipeak_rz=_mirror(cp.asnumpy(self.Ipeak_rz[:self.k_save])),
            rho_rz_at=(_mirror(cp.asnumpy(self.rho_rz_at[:self.k_save]))
                       if self.rho_rz_at is not None else None),
            absorbed_rz_bins=(np.stack([_mirror(cp.asnumpy(self.absorbed_rz_bins[:self.k_save, b]))
                                        for b in range(self.absorbed_rz_bins.shape[1])], axis=1)
                              if self.absorbed_rz_bins is not None else None),
            absorb_bin_edges_fs=self.absorb_bin_edges_fs,
            rho_onaxis_t=self.rho_onaxis_t[:self.k_save],
            rho_s_onaxis_t=self.rho_s_onaxis_t[:self.k_save],
            I_onaxis_t=self.I_onaxis_t[:self.k_save],
            t_full_fs=self.t_full_fs,
        )
        # Atomic write: a run that dies mid-save (disk full -> OSError errno 5)
        # must not leave a truncated result.npz behind, because the next run
        # would find it, try to load it, and crash with a zlib "invalid block
        # type" instead of simply recomputing.
        final = os.path.join(cfg.out_dir, "result.npz")
        tmp = final + ".tmp.npz"
        try:
            np.savez_compressed(tmp, **out)
            os.replace(tmp, final)
        except BaseException:
            for f in (tmp, tmp + ".npz"):
                try:
                    os.remove(f)
                except OSError:
                    pass
            raise
        self._dump_params()
        return out

    def _dump_params(self):
        """Companion params.json (probe optics + which physics channels were
        active), consumed by web/abel_phase_explorer.py to compute Delta n and
        to label ablation scenarios without re-deriving them from the npz."""
        cfg = self.cfg
        params = dict(
            n0=cfg.n0, n0_probe=cfg.n0_probe,
            nc_probe_cm3=cfg.nc_probe,
            lambda_probe_nm=cfg.lambda_probe * 1e9,
            wavelength_nm=cfg.wavelength * 1e9,
            n2=cfg.n2, U_g_eV=cfg.Ui_eV, Us_eV=cfg.Us_eV,
            energy_uJ=cfg.energy_uJ, w0_um=cfg.w0 * 1e6,
            delta_t_fs=cfg.delta_t * 1e15,
            # z_sim = 0 is always the gaussian geometric focus (see
            # envelope_gaussian_focused: curvature = 1 at begin = 0), so the
            # focus-to-interface distance in lab space is simply -begin_um
            # whenever `begin` was set to minus that distance (as in the
            # original notebook). Exposed here so abel_phase_explorer.py does
            # not need a hardcoded experimental-geometry constant.
            code_fingerprint=code_fingerprint(),
            begin_um=cfg.begin * 1e6, end_um=cfg.end * 1e6,
            z_focus_air_um=cfg.z_focus_air_um,
            toggles=dict(
                enable_kerr_instantaneous=cfg.enable_kerr_instantaneous,
                enable_kerr_raman=cfg.enable_kerr_raman,
                enable_self_steepening=cfg.enable_self_steepening,
                enable_photoionization_loss=cfg.enable_photoionization_loss,
                enable_plasma_defocusing=cfg.enable_plasma_defocusing,
                enable_plasma_absorption=cfg.enable_plasma_absorption,
                enable_space_time_focusing=cfg.enable_space_time_focusing,
                enable_spectral_filter=cfg.enable_spectral_filter,
                tau_ste_fs=(cfg.tau_ste * 1e15 if cfg.tau_ste else None),
                enable_ste_index=cfg.enable_ste_index,
                E_tr_eV=cfg.E_tr_eV, f_ste_pump=float(self.g["f_ste"]),
                enable_avalanche=cfg.enable_avalanche,
                enable_recombination=cfg.enable_recombination,
                enable_ste=cfg.enable_ste,
            ),
        )
        with open(os.path.join(cfg.out_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

