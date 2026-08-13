import matplotlib; matplotlib.use("Agg")
import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "sim")
import figures_filament as ff
res = ff.load_scenario_npz("runs_z0/z0_350um_4uJ")
PK = dict(E_tr_eV=4.2, n2=2.74e-20, tau_c_s=1.7e-15, tau_r_s=330e-15, tau_ste_s=1e-12)
z = np.asarray(res["z"])*1e6
t_sub = np.asarray(res["t_sub_fs"])
vg = 299.792458/1.4627

fig = plt.figure(figsize=(13.5, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.28)
a = fig.add_subplot(gs[0])
a.add_patch(Rectangle((z[0], t_sub[0]), z[-1]-z[0], t_sub[-1]-t_sub[0],
                      fc="#e8e8ee", ec="0.5", lw=1.2, zorder=0))
a.text(z[-1]*0.5, t_sub[-1]*0.62, "fenetre temporelle du cube",
       ha="center", fontsize=9.5, color="0.35", zorder=1)
cols = plt.cm.plasma(np.linspace(0.05, 0.8, 5))
for tau, c in zip((0, 500, 1000, 1500, 2000), cols):
    tl = tau - z/(vg*1e-3)
    ins = (tl >= t_sub[0]) & (tl <= t_sub[-1])
    a.plot(z[ins], tl[ins], lw=2.6, color=c, zorder=3)
    a.plot(z[~ins], tl[~ins], lw=1.8, color=c, ls=":", zorder=2)
    k = np.argmax(ins) if ins.any() else 0
    a.text(z[k]+6, tl[k]+30, rf"$\tau$={tau} fs", color=c, fontsize=10,
           fontweight="bold", zorder=4)
a.set_xlabel(r"$z$ [$\mu$m]", fontsize=11)
a.set_ylabel(r"temps local de la pompe  $t$ [fs]", fontsize=11)
a.set_title("Etape 1 — la sonde coupe le cube EN BIAIS\n"
            r"$t_{local}(z)\;=\;\tau\;-\;z/v_g$", fontsize=12)
lim = max(abs(t_sub[0]), abs(t_sub[-1]))*1.35
a.set_ylim(-lim*1.5, lim); a.set_xlim(z[0]-8, z[-1]+8)
a.grid(alpha=.3, zorder=0)
a.text(z[-1]*0.42, -lim*1.32,
       f"la pompe met {z[-1]/vg*1e3:.0f} fs pour traverser la boite\n"
       "pointille : hors fenetre, prolongement analytique",
       fontsize=9.5, color="0.3")

d = ff.probe_opl_transmittance(res, 1000.0, x_half_um=70.0, **PK)
rho = np.asarray(d["rho_e_rz"]); r = d["r_pos_um"]; m = r <= 25
b = fig.add_subplot(gs[1])
im = b.pcolormesh(z, r[m], rho[:, m].T, cmap="magma", shading="auto")
b.set_xlabel(r"$z$ [$\mu$m]", fontsize=11); b.set_ylabel(r"$r$ [$\mu$m]", fontsize=11)
b.set_title(r"la coupe donne une carte : $\rho_e(z,r)$ a $\tau=1000$ fs"
            "\n" r"[451, 512, 256]  $\rightarrow$  [451, 512]", fontsize=12)
fig.colorbar(im, ax=b, label=r"$\rho_e$ [cm$^{-3}$]")
fig.savefig("/tmp/pipeline_1.png", dpi=150, bbox_inches="tight")
from PIL import Image; print("taille :", Image.open("/tmp/pipeline_1.png").size)
