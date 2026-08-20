# Changing the physics: where every term of the equations lives

This file is for someone who has a new idea about the model and wants to change
what the solver integrates. It answers one question: given a term in the
equations, which line of which file do I edit?

Read it together with the docstring at the top of `run_filament.py`, which
writes both equations out in full. This file is the map from that docstring to
the code.

Line numbers are given as a hint. They drift when the files are edited, so
search for the variable name if a number no longer matches.


## 1. Why the equation is spread over four files

The solver uses a split-step scheme. Each term is applied in the basis where it
is diagonal, because a diagonal operator is just a multiplication and needs no
matrix solve. That is good for speed and bad for reading, since the equation
gets cut into pieces that live in different places.

    config.py       every tunable number and every on/off flag
    grids.py        the axes, and every prefactor built once before the run
    operators.py    the two halves of the right hand side
    kernels.py      the carrier rate equations, as CUDA C
    integrator.py   the order the pieces are applied in, and the recording

The rule of thumb is: a number that does not change during the run is built in
`grids.py`, and a thing that depends on the field is computed in
`operators.py`.


## 2. Field equation, term by term

The equation, with the flag that switches each line off.

      U^ du/dz =   (i / 2k0) grad_perp^2 u                      always on
                 + i D^ U^ u                                    always on
                 + i T^^2 kerr_pref (1-f_R) |u|^2 u             enable_kerr_instantaneous
                 + i T^^2 kerr_pref f_R (R*|u|^2) u             enable_kerr_raman
                 - T^ (Ui W_PI / n0 c eps0 |u|^2) (1 - N/N_at) u
                                                                enable_photoionization_loss
                 - (sigma_w / 2) N u                            enable_plasma_absorption
                 - i (sigma_w w0 tau_c / 2) N u                 enable_plasma_defocusing
                 + i (w0 / 2 n0 rho_c) f_STE N_STE u            enable_ste_index

| Term | Computed in | Prefactor built in | Flag |
|---|---|---|---|
| diffraction, `grad_perp^2` | `operators.py:50`, the `rhorho**2 / (2*komega)` part of `phase` | `grids.py:44` (`rholist`) | none |
| dispersion `D^` | `operators.py:50`, the `delta_k` part of `phase` | `grids.py:59` | none |
| Kerr, instantaneous | `operators.py:110-111` (`kerr_I`) | `operators.py:66` (`kerr_pref`) | `enable_kerr_instantaneous` |
| Kerr, Raman | `operators.py:112-113` | `grids.py:113` (`R_f`) | `enable_kerr_raman` |
| photoionization loss | `operators.py:97` (`photo`) | `grids.py:140-161` (Keldysh tables) | `enable_photoionization_loss` |
| plasma absorption | `operators.py:132`, real part of `plasma_coeff` | `operators.py:67` (`plasma_pref`), `grids.py:121` (`sigmaomega`) | `enable_plasma_absorption` |
| plasma defocusing | `operators.py:133`, imaginary part of `plasma_coeff` | `operators.py:68` (`plasma_phase`) | `enable_plasma_defocusing` |
| STE index | `operators.py:138-139` | `grids.py:133-137` (`ste_pref`) | `enable_ste_index` |
| `T^`, self steepening | applied at `operators.py:141-142` | `grids.py:79-82` | `enable_self_steepening` |
| `U^`, space time focusing | applied at `operators.py:144` and `:50` | `grids.py:88-111` | `enable_space_time_focusing` |
| spectral mask | folded into `T_op` and `inv_U_nl` | `grids.py:66-73` | `enable_spectral_filter` |

Two remarks on the table.

The depletion factor `(1 - N/N_at)` on the photoionization loss is
`depl_field`, `operators.py:96`. It is clipped to `[0, 1]` so it can never go
negative and turn a loss into a gain.

The two plasma lines of the equation are **one** line of code. `plasma_coeff`
at `operators.py:132-133` is a complex number whose real part is the absorption
and whose imaginary part is the defocusing, and the two flags each contribute
one part of it. Do not look for two separate terms, there are none.


## 3. The one thing that will catch you out

`NonlinearOperator.split()` does **not** return `du/dz`.

Look at `Integrator.step()`, `integrator.py:138-149`. The dissipative part of
the equation is applied separately, as two exponential factors around the RK4
block.

    u = half_linear(u, dz)
    u = u * exp(-0.5 * dz * alpha)        <- absorption, zeroth order
    ... RK4 on split() ...
    u = u * exp(-0.5 * dz * alpha)        <- absorption, zeroth order
    u = half_linear(u, dz)

`alpha`, built at `operators.py:104-107`, is the sum of the photoionization
loss and the plasma absorption. Since those two channels are already applied by
the exponentials, they must not be applied a second time by the RK4. So they
are **subtracted back out** inside `split()`, and that is what the `+ alpha * u`
at the end of line 144 is doing.

Work through the plasma channel with both flags on. `plasma_coeff` becomes
`1 + i*w0*tau_c`, so

    extra      = -plasma_pref * (1 + i*w0*tau_c) * rho * u
    alpha      =  photo + plasma_pref * rho
    rhs        =  ... + extra + alpha * u

The real part `-plasma_pref * rho * u` cancels the `+plasma_pref * rho * u`
coming from `alpha * u`, exactly. What survives is
`-i * plasma_pref * w0 * tau_c * rho * u`, the defocusing, which is what should
be integrated by the RK4 because it is a phase and not a loss. The same
cancellation happens for `photo`: the `-fft(photo*u)*T_op` at line 142 cancels
the `photo` part of `alpha*u`, to the extent that `T^` and `U^` are the
identity.

**The consequence.** If you add a new absorbing channel and only add it to
`alpha`, it will be applied twice, once by the exponentials and once by the
RK4, and your absorption will be about twice too strong. Nothing will warn you.
Section 5 gives the recipe that avoids this.


## 4. Carrier equations, term by term

      dN/dt     = W_PI(I) (1 - N/N_at)                    always on
                + beta_g I N (1 - N/N_at)                 enable_avalanche
                + (W_STE + beta_s I N) N_STE/N_at         enable_ste
                - N / tau_r                               enable_recombination

      dN_STE/dt = N / tau_r                               enable_ste
                - (W_STE + beta_s I N) N_STE/N_at         enable_ste
                - N_STE / tau_ste                         tau_ste is not None

These are integrated on the GPU, one CUDA thread per radius, marching
sequentially in time. The source is the C string `_RATE_KERNEL_SRC` in
`kernels.py`, and the whole right hand side is packed into four variables.

| Variable | Line | Meaning |
|---|---|---|
| `S_e` | `kernels.py:77` | everything in `dN/dt` that does not multiply `N` |
| `L_e` | `kernels.py:78` | everything in `dN/dt` that does multiply `N` |
| `S_s` | `kernels.py:80` | same, for `dN_STE/dt` |
| `L_s` | `kernels.py:86` | same, for `dN_STE/dt` |

The split into a source `S` and a linear rate `L` is not cosmetic. The
integrator at `kernels.py:28-34` is `exact_exp_step`, which solves
`dx/dt = S + L x` exactly over one time step rather than approximating it. That
is what lets the kernel take large steps through the very stiff avalanche phase
without going unstable. So a new term has to be classified: if it is
proportional to the population being solved for, it belongs in `L`, otherwise
in `S`. Putting a term in the wrong one is not a small error, it changes the
stiffness handling.

The flags do not appear in the kernel. They are folded into the coefficients
back in `grids.py:124-127`: `enable_avalanche` off simply makes `beta_g` zero,
`enable_recombination` off makes `inv_tau_r` zero. This is why the kernel has
no branches for them.

**The flags of section 2 do not act here.** Turning off
`enable_photoionization_loss` stops the field from losing energy to ionization
but the kernel keeps making electrons at the same Keldysh rate. That is
deliberate, it is what lets you ask "what does this loss channel do to the
beam" separately from "how many electrons are there". It also means an ablation
study never conserves energy, by construction.


## 5. Recipes

### 5.1 Change a number

Do not edit any file. Pass it to `simulate()`.

    simulate(meff_drude_rel=0.5, tau_c_s=2.0e-15)

Every physical parameter is an argument there, and a misspelled name raises
instead of being silently ignored.

### 5.2 Change the form of a term that already exists

Edit the one line in `operators.py` from the table in section 2. For instance
to make the Kerr response saturate, replace `operators.py:111`

    kerr_I = kerr_I + (1.0 - self.f_R) * absu2

by

    kerr_I = kerr_I + (1.0 - self.f_R) * absu2 / (1.0 + absu2 / I_sat_field2)

Nothing else needs touching, because the Kerr term is a pure phase and does not
enter `alpha`.

### 5.3 Add a new term that is a pure phase, no loss

This is the easy case. Follow what the STE index term does, it is the most
recent addition and the cleanest example.

1. `config.py`: add the flag and any new parameter next to `enable_ste_index`.
2. `grids.py`: build the prefactor near `ste_pref`, `grids.py:133-137`, and add
   it to the dict returned at the end. Set it to `0.0` when the flag is off,
   which is how every other prefactor disables itself.
3. `operators.py`: read it in `__init__` next to `self.ste_pref`, and add your
   contribution to `extra` at `operators.py:137-139`.
4. `run_filament.py`: add the argument to `simulate()` and a line to
   `_FIELD_TERMS` so it shows up in the ON/OFF listing.
5. `integrator.py`: add the flag to the `toggles` dict at `:322-337`, so a
   finished run records which physics produced it.

Do **not** add it to `alpha`. A pure phase is not a loss.

### 5.4 Add a new term that absorbs energy

Same five steps, plus the bookkeeping of section 3. You must do both of these:

- add your absorption rate to `alpha`, next to `operators.py:105-106`, so the
  exponential factors apply it;
- add its **negative** to `extra` or to `NL_freq`, so the `+ alpha * u` at line
  144 does not apply it a second time.

If your term should also carry a phase, add the phase part only to `extra`,
exactly as `plasma_coeff` puts the absorption in the real part and the phase in
the imaginary part.

Then decide whether `T^` applies to it. Look at `operators.py:141-143`: the
Kerr bracket is multiplied by `T_op**2`, the ionization loss by `T_op`, and
`extra` by neither. Those powers come from Couairon 2005 Eq. (4) and are not
interchangeable. If your term needs a `T^`, give it its own `fft` line rather
than folding it into an existing one.

Finally, if the term is a real energy loss you probably want it in
`loss_rates()` too, `operators.py:148-163`. That function is only used for the
energy bookkeeping figures, not for the propagation, but it is written to
mirror `split()` exactly, and letting the two drift apart makes the energy
budget silently wrong.

### 5.5 Add a term to the carrier equations

Edit the CUDA source in `kernels.py`. Decide whether the term multiplies the
population being solved for. If yes it goes into `L_e` or `L_s`, if no into
`S_e` or `S_s`. Build any new coefficient in `grids.py` near
`avalanche_coef`, pass it through the argument list of `rate_eq_kernel` in
`operators.py:169-183`, and add the matching parameter to the kernel signature
at `kernels.py:37-46`.

The argument list is positional and unchecked. Adding a parameter in the middle
of the Python call without adding it in the same place in the C signature will
not raise, it will silently read the wrong numbers. Add new parameters at the
end of both.

### 5.6 Add a new initial field

`grids.py:180-185`. Write a function with the signature
`(rr, tt, cfg, g) -> complex array` and register it in `ENVELOPES`. It can then
be named by string, or passed directly as a callable.


## 6. Units

Mixed units are the second most common source of wrong results here, after the
`alpha` double counting.

| Quantity | Unit in the code |
|---|---|
| lengths, `z`, `r`, `w0` | m |
| time | s |
| carrier densities `rho`, `rho_s`, `rho_max` | **cm^-3** |
| intensity `I` | **W/cm^2** |
| absorption `alpha` | m^-1 |
| Keldysh rate out of the tables | cm^-3 s^-1, hence the `* 1e6` at `operators.py:97` |

The conversions are folded into three constants: `invE2` (`grids.py:128`) turns
`|u|^2` into W/cm^2, the `* 1e4` at the end of `sigmaomega` (`grids.py:123`)
and the `* 100.0` in `plasma_pref` (`operators.py:67`) turn the Drude cross
section into an absorption per metre acting on a density in cm^-3.

If you introduce a new density or a new cross section, write its unit in a
comment on the line where it is built. Every existing prefactor does.


## 7. Checking that a change did what you think

The solver has no unit tests, so verification is by ablation.

Run the same case twice, once with your new term on and once off, and compare.
`simulate()` prints the full ON/OFF listing before every run, so the two logs
are the record of what actually differed.

    simulate(run_tag="with",    enable_my_new_term=True)
    simulate(run_tag="without", enable_my_new_term=False)

Three checks worth doing on any new term.

**Energy.** `run_health_check` prints the beam energy against the initial
value. A pure phase term must leave it unchanged to the level of the numerical
noise. If adding a phase term changes the energy, you have put something in
`alpha` that does not belong there.

**Double counting.** Set your new absorption coefficient to a small value and
check that the extra loss scales linearly with it. If it comes out about twice
what you predicted analytically, re-read section 3.

**Step size.** Halve `Nz` and check the answer barely moves. A new stiff term
can quietly break the convergence of the split step, and the symptom is a
result that keeps changing as you refine `dz`.


## 8. A known rough edge

The structure above works but it is not pluggable. Adding one term means
editing five files, and the `alpha` bookkeeping of section 3 has to be got
right by hand every time.

A term registry, where each term is one object carrying its own flag, its
prefactor, its `T^` power and whether it is dissipative, would make
sections 5.3 and 5.4 a single edit and would make the double counting
structurally impossible. That is a real refactor of the hot path of a solver
that currently reproduces the published figures, so it has not been done. It is
worth doing if the model starts changing often.
