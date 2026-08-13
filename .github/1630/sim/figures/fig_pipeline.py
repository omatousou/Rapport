import matplotlib; matplotlib.use("Agg")
import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
sys.path.insert(0, "sim")
import figures_filament as ff
res = ff.load_scenario_npz("runs_z0/z0_350um_4uJ")
PK = dict(E_tr_eV=4.2, n2=2.74e-20, tau_c_s=1.7e-15, tau_r_s=330e-15, tau_ste_s=1e-12)

z = np.asarray(res["z"])*1e6
t_sub = np.asarray(res["t_sub_fs"])
n_g = 1.4627; vg = 299.792458/n_g          # µm/ps

# ============ FIGURE 1 : la coupe oblique ============
fig, ax = plt.subplots(1, 2, figsize=(13, 5),
                       gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]
a.add_patch(Rectangle((z[0], t_sub[0]), z[-1]-z[0], t_sub[-1]-t_sub[0],
                      fc="0.90", ec="0.55", lw=1.2, zorder=0))
a.text(0.5*(z[0]+z[-1]), t_sub[-1]-120, "fenetre temporelle du solveur",
       ha="center", fontsize=9, color="0.35")
cols = plt.cm.viridis(np.linspace(0.05, 0.85, 5))
for tau, c in zip((0, 500, 1000, 1500, 2000), cols):
    tl = tau - z/(vg*1e-3)
    inside = (tl >= t_sub[0]) & (tl <= t_sub[-1])
    a.plot(z[inside], tl[inside], lw=2.4, color=c)
    a.plot(z[~inside], tl[~inside], lw=2.0, color=c, ls=":")
    j = np.argmin(np.abs(z - z[-1]*0.06))
    a.text(z[j], tl[j]+55, rf"$\tau$ = {tau} fs", color=c, fontsize=9,
           fontweight="bold")
a.set_xlabel(r"$z$ [$\mu$m]"); a.set_ylabel(r"temps local de la pompe $t$ [fs]")
a.set_title("Etape 1 : la sonde coupe le cube EN BIAIS\n"
            r"$t_{local}(z)=\tau - z/v_g$", fontsize=11)
a.set_ylim(t_sub[0]-500, t_sub[-1]+400)
a.grid(alpha=.3)
a.annotate("", xy=(z[-1], -1500), xytext=(z[0], -1500),
           arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5))
a.text(0.5*(z[0]+z[-1]), -1400, f"la pompe met {z[-1]/vg*1e3:.0f} fs\n"
       f"pour traverser la boite", ha="center", fontsize=9, color="crimson")
a.text(z[-1]*0.55, t_sub[0]-380, "pointille = hors fenetre,\nprolongement analytique",
       fontsize=8.5, color="0.35")

d = ff.probe_opl_transmittance(res, 1000.0, x_half_um=70.0, **PK)
rho = np.asarray(d["rho_e_rz"]); r = d["r_pos_um"]
b = ax[1]
im = b.pcolormesh(z, r[r<=30], rho[:, r<=30].T, cmap="magma", shading="auto")
b.set_xlabel(r"$z$ [$\mu$m]"); b.set_ylabel(r"$r$ [$\mu$m]")
b.set_title(r"ce que la coupe donne : $\rho_e(z,r)$ a $\tau$ = 1000 fs"
            "\n" r"[451, 512, 256] $\rightarrow$ [451, 512]", fontsize=11)
fig.colorbar(im, ax=b, label=r"$\rho_e$ [cm$^{-3}$]")
fig.tight_layout(); fig.savefig("/tmp/pipeline_1.png", dpi=155, bbox_inches="tight")

# ============ FIGURE 2 : la chaine ============
from virtual_experiment import lowpass_NA, NOMARSKI_515
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
dn = np.asarray(d["dn_rz"]); al = np.asarray(d["alpha_cm_rz"])
mr = r <= 30
p = ax[0,0].pcolormesh(z, r[mr], dn[:,mr].T, cmap="bwr", shading="auto",
                       vmin=-np.abs(dn).max(), vmax=np.abs(dn).max())
ax[0,0].set_title(r"2a. $\Delta n(z,r)=\mathrm{Re}\sqrt{\varepsilon}-n_0$", fontsize=11)
fig.colorbar(p, ax=ax[0,0])
p = ax[0,1].pcolormesh(z, r[mr], al[:,mr].T, cmap="inferno", shading="auto")
ax[0,1].set_title(r"2b. $\alpha(z,r)=2\omega\,\mathrm{Im}\sqrt{\varepsilon}/c$", fontsize=11)
fig.colorbar(p, ax=ax[0,1], label="cm$^{-1}$")
for k in (0,1): ax[0,k].set_xlabel(r"$z$ [$\mu$m]"); ax[0,k].set_ylabel(r"$r$ [$\mu$m]")

iz = int(np.argmax(np.abs(d["opl_nm"]).max(axis=1)))
c = ax[0,2]
c.plot(r[mr], dn[iz][mr], color="crimson", lw=2, label=r"$\Delta n(r)$ local")
c.set_xlabel(r"$r$ ou $x$ [$\mu$m]"); c.set_ylabel(r"$\Delta n$", color="crimson")
c.tick_params(axis="y", colors="crimson")
c2 = c.twinx()
xx = d["x_um"]; mx = np.abs(xx) <= 30
c2.plot(xx[mx], d["opl_nm"][iz][mx], color="royalblue", lw=2)
c2.set_ylabel("OPL [nm]  (projete)", color="royalblue")
c2.tick_params(axis="y", colors="royalblue")
c.set_title(f"3. Abel direct, a z = {z[iz]:.0f} " r"$\mu$m"
            "\n" r"$F(x)=2\int_x^\infty f(r)\,r\,dr/\sqrt{r^2-x^2}$", fontsize=11)
c.grid(alpha=.3)

opl = np.asarray(d["opl_nm"])
p = ax[1,0].pcolormesh(z, xx, opl.T, cmap="bwr", shading="auto",
                       vmin=-np.abs(opl).max(), vmax=np.abs(opl).max())
ax[1,0].set_title("3b. vue de cote OPL$(z,x)$\n[451, 512] $\\rightarrow$ [451, 561]", fontsize=11)
fig.colorbar(p, ax=ax[1,0], label="nm")

dzs = float(np.mean(np.diff(z))); dxs = float(np.mean(np.diff(xx)))
lp = lowpass_NA(opl, dzs, dxs, 0.23, 0.515)
p = ax[1,1].pcolormesh(z, xx, lp.T, cmap="bwr", shading="auto",
                       vmin=-np.abs(opl).max(), vmax=np.abs(opl).max())
ax[1,1].axhline(0, color="k", lw=.8, ls=":")
ax[1,1].plot([150],[0], "o", ms=9, mfc="none", mec="lime", mew=2.5)
ax[1,1].set_title(f"4. passe-bas a NA/$\\lambda$ = {NOMARSKI_515.resolution_um:.2f} "
                  r"$\mu$m" "\npuis lecture au point vert", fontsize=11)
fig.colorbar(p, ax=ax[1,1], label="nm")
for k in (0,1):
    ax[1,k].set_xlabel(r"$z$ [$\mu$m]"); ax[1,k].set_ylabel(r"$x$ [$\mu$m]")
    ax[1,k].set_ylim(-30,30)

cur = np.loadtxt("/home/user/Rapport/.github/1630/notebooks/runs_z0/curve_4uJ.csv",
                 delimiter=",", skiprows=1)
e = ax[1,2]
e.axhline(0, color="0.6", lw=1)
e.plot(cur[:,0]*1e-3, cur[:,1], "o", ms=3.5, color="crimson")
e.set_xlabel("delai optique (ps)"); e.set_ylabel(r"$\delta\varphi$ (rad)")
e.set_title(r"5. un point par delai : $\delta\varphi(\tau)=2\pi\,$OPL$/\lambda$"
            "\n(courbe reelle du run 4 uJ)", fontsize=11)
e.grid(alpha=.3)
fig.suptitle("De $(z,r,t)$ a $\\delta\\varphi(\\tau)$ : la chaine complete",
             fontsize=13, y=1.00)
fig.tight_layout(); fig.savefig("/tmp/pipeline_2.png", dpi=145, bbox_inches="tight")
print("ok")
