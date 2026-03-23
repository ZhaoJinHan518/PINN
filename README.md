# PINN

This repository reproduces PINN examples for:

- the nonlinear Schrödinger equation (Section 3.1.1, continuous-time model)
- the Allen-Cahn equation (Section 3.2, discrete-time model)
- the 2D Navier-Stokes equation (Section 4.1.1, cylinder wake example)

## Setup

```bash
pip install -r requirements.txt
```

## Run the Schrödinger PINN example

```bash
python pinn_schrodinger.py --n0 50 --nb 50 --nf 20000 --lbfgs-max-iter 500 --save-predictions schrodinger_pinn.npz
```

The script trains a PINN with periodic boundary conditions and saves the predicted
solution on a regular grid (amplitude, real, and imaginary parts) into the `.npz`
file specified by `--save-predictions`.

## Run the discrete-time Allen-Cahn PINN example

```bash
python pinn_allen_cahn_discrete.py --q 100 --t-data 0.1 --t-target 0.9 --dt 0.8 --n-data 200 --lbfgs-max-iter 500 --save-predictions allen_cahn_pinn.npz
```

The script generates reference data from a spectral solver, trains a discrete-time
PINN with implicit Gauss-Legendre collocation, and saves the predicted solution
at the target time into the `.npz` file specified by `--save-predictions`. Note
that `--dt` is the total interval from `t-data` to `t-target`, while `--solver-dt`
controls the spectral solver time step used to generate reference data.

## Run the Navier-Stokes PINN example

```bash
python pinn_navier_stokes.py --data-path cylinder_wake.npz --n-train 5000 --lbfgs-max-iter 500 --save-predictions navier_stokes_pinn.npz
```

The script loads a cylinder wake dataset (scattered or gridded) from a `.npz` (or
`.mat`) file, trains a PINN that outputs the stream function and pressure, and
saves predicted velocities and pressure for sampled points. The dataset should
contain `t`, `x`, `y`, `u`, `v` arrays of matching size, or gridded `U`, `V`
arrays with `t`, `x`, `y` coordinates. Optional pressure data can be provided as
`p`/`P`/`P_star`.
