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
                 + i (w0 / 2 n0 c rho_c) f_STE N_STE u          enable_ste_index

| Term | Computed in | Prefactor built in | Flag |
|---|---|---|---|
| diffraction, `grad_perp^2` | `LinearOperator.half_linear`, the `rhorho**2 / (2*komega)` part of `phase` | `grids.py:44` (`rholist`) | none |
| dispersion `D^` | `LinearOperator.half_linear`, the `delta_k` part of `phase` | `grids.py:59` | none |
| Kerr, instantaneous | `_kerr_instantaneous` | `kerr_pref`, in `NonlinearOperator.__init__` | `enable_kerr_instantaneous` |
| Kerr, Raman | `_kerr_raman` | `grids.py:113` (`R_f`) | `enable_kerr_raman` |
| photoionization loss | `_Ctx.photo`, returned by `_photoionization_loss` | `grids.py:140-161` (Keldysh tables) | `enable_photoionization_loss` |
| plasma absorption | `_plasma_absorption` | `plasma_pref`, `grids.py:121` (`sigmaomega`) | `enable_plasma_absorption` |
| plasma defocusing | `_plasma_defocusing` | `plasma_phase` | `enable_plasma_defocusing` |
| STE index | `_ste_index` | `grids.py:133-137` (`ste_pref`) | `enable_ste_index` |
| `T^`, self steepening | applied by `split()`, per term `T_power` | `grids.py:79-82` | `enable_self_steepening` |
| `U^`, space time focusing | applied by `split()` and `half_linear` | `grids.py:88-111` | `enable_space_time_focusing` |
| spectral mask | folded into `T_op` and `inv_U_nl` | `grids.py:66-73` | `enable_spectral_filter` |

Each of the six nonlinear rows is one entry of `FIELD_TERMS` in `operators.py`,
the tuple `split()` loops over. The function named in the second column is the
term's `fn`, and the equation line in the first column is stored on the entry
itself as its `equation` field, which is what `run_filament.py` prints in its
ON/OFF listing.

Two remarks on the table.

The depletion factor `(1 - N/N_at)` on the photoionization loss is
`depl_field`, `operators.py:96`. It is clipped to `[0, 1]` so it can never go
negative and turn a loss into a gain.

The two plasma lines of the equation are two separate registry entries, one a
`loss` and one a `phase`, even though they are one physical process. They have
separate flags, so they have to be separately switchable. Before the registry
they were fused into a single complex coefficient called `plasma_coeff`.


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

**Why you no longer have to get this right by hand.** A dissipative term does
not declare its contribution to the field. It declares only its absorption
rate, and `split()` derives both halves from that single declaration: it adds
the rate to `alpha`, and it puts `-rate*u` into the sum. There is no way to
register one without the other. Before the registry both halves were written
out by hand for each channel, and adding a channel to `alpha` alone made the
absorption about twice too strong with no warning.

You still need to understand this section, because it is why `split()` returns
what it returns, and because it is the reason the equivalence test can compare
one array instead of two full simulations.


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
back in `grids.py:124-127`: `enable_avalanche` off makes both `beta_g` and
`beta_s` zero, so it also removes the `beta_s I N` half of the STE
re-ionization,
`enable_recombination` off makes `inv_tau_r` zero. This is why the kernel has
no branches for them.

**The flags of section 2 do not act here.** Turning off
`enable_photoionization_loss` stops the field from losing energy to ionization
but the kernel keeps making electrons at the same Keldysh rate. That is
deliberate, it is what lets you ask "what does this loss channel do to the
beam" separately from "how many electrons are there". It also means an ablation
study never conserves energy, by construction.


## 4b. Four things the solver does that are in no equation

These are deliberate, but you will not find them by reading the equations, and
two of them change the physics rather than only the numerics.

**Radial absorbing boundary.** `integrator.py:149` ends every z step with
`u = u * mask_r`, where `mask_r = exp(-(r / 0.9R)^20)` is built at
`grids.py:138`. It stops light that reaches the edge of the box from wrapping
around through the Hankel transform. It also removes energy, so a beam that
spreads close to `R` will not conserve energy, for a reason that has nothing to
do with ionization.

**Joint saturation clamp.** `kernels.py:91-95`. If `N + N_STE` exceeds `N_at`
at any time step, both are scaled down so they sum to `N_at`. Neither rate
equation contains this, and it switches itself on exactly in the strongly
ionized regime.

**The spectral mask is not a separate factor.** It is folded into `T_op`
(`grids.py:80`) and into `inv_U_nl` (`grids.py:109`). Two consequences. The
Kerr term carries the mask squared, since it is multiplied by `T_op**2`, while
the ionization term carries it once. And with `enable_self_steepening` off
`T_op` becomes a plain array of ones with no mask, so turning that flag off
also removes the spectral filter from the Kerr and ionization terms even though
`enable_spectral_filter` is still on. The linear step keeps its own mask
either way.

**`D^` is exact only inside the Sellmeier window.** `delta_k` is built at
`grids.py:59` from `omega_safe`, which `grids.py:46-47` clips to
lambda in `[0.18, 5]` um. Outside that band the dispersion is frozen at the
edge value rather than extrapolated. The mask has normally killed the field
there already, but the two are separate mechanisms and only one of them has a
flag.


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

### 5.3 Add a new term to the field equation

Write one `FieldTerm` and add it to the `FIELD_TERMS` tuple in `operators.py`.
That is the whole edit to the solver.

    def _my_new_term(op, ctx):
        return 1j * op.my_pref * ctx.rho_s * ctx.u        # a phase

    FieldTerm("my_new_term", "enable_my_new_term", "phase", 0,
              _my_new_term, "+ i my_pref N_STE u")

The four things the entry declares.

**`kind`** is `"phase"` or `"loss"`, and it is the field that matters.

- `"phase"`: `fn` returns the term's contribution to `du/dz` directly.
- `"loss"`: `fn` returns a non-negative absorption **rate**, in 1/m, and
  nothing else. `split()` then does both halves of the bookkeeping of
  section 3 for you, adding the rate to `alpha` and putting `-rate*u` into the
  sum. You cannot do one without the other, which is the point.

So an absorbing term is written exactly like `_plasma_absorption`, one line
returning a rate. Do not touch `alpha` yourself.

**`T_power`** is 0, 1 or 2, the power of `T^` in front of the term. Couairon
2005 Eq. (4) puts `T^2` on the Kerr bracket and `T^1` on the ionization loss.
Terms are grouped by this before transforming, so adding a term does not add an
FFT unless it uses a power nothing else uses.

**`flag`** is a `Config` field. A term whose flag is false is filtered out of
the registry at construction, so `split()` never branches on it.

**`equation`** is the line as it appears in the written equation. It is what
`run_filament.py` prints in its ON/OFF listing, so the listing cannot drift
from the code.

Anything the term needs from the field or the densities comes through `ctx`,
which carries `u`, `absu2`, `rho`, `rho_s`, and a lazily computed `photo`. Add
a cached property there if your term needs a new shared quantity, and it will
be computed at most once per call even if several terms want it.

Around that one edit, three bits of plumbing.

1. `config.py`: add the flag and any new parameter.
2. `grids.py`: build the prefactor near `ste_pref`, and add it to the returned
   dict. Set it to `0.0` when disabled, as every other prefactor does. Read it
   in `NonlinearOperator.__init__`.
3. `integrator.py`: add the flag to the `toggles` dict in `_dump_params`, so a
   finished run records which physics produced it. And add the argument to
   `simulate()` in `run_filament.py`.

Then run the equivalence test with your term disabled. It must still pass:
a new term that is off must change nothing.

### 5.4 Change a term without changing the physics

If you rearrange or optimize `split()` itself rather than adding a term, prove
it neutral with `sim/test_operators_equivalence.py`. It compares `split()`
against a frozen copy of a known-good version on random inputs, over all 64
combinations of the six field flags, and runs on the CPU in a second with no
GPU needed.

Comparing that one return value is enough. As section 3 shows, the entire
nonlinear contribution to `du/dz` over a step is `ifft(NL_freq * inv_U_nl)`, so
two implementations that return the same `(rhs, alpha)` integrate the same
equation. There is no need to run a propagation and compare pictures.

When you deliberately change the physics, the test will fail, which is correct.
Update the frozen reference copy in the test in the same commit, so it becomes
the record of what the new physics is.

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


## 8. What is still not pluggable

The field equation is a registry. The rest is not.

The carrier equations are still CUDA C with a positional, unchecked argument
list, and section 5.5 is still a five-place edit. That is the next thing worth
doing, and it is harder than the field equation was, because the `S`/`L`
packing is load bearing rather than incidental.

The prefactors are still built in `grids.py`, away from the term that uses
them. A term declares its `fn` but not where its constants come from, so
adding a term still means touching `grids.py` as well as `operators.py`.
Letting a `FieldTerm` carry its own prefactor builder would close that, and it
is a small change now that the registry exists.

`half_linear` is untouched. Diffraction and dispersion have no flags and are
never modified, so there was nothing to gain.
