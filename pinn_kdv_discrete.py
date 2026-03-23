#!/usr/bin/env python3
import argparse
import math
import time

import numpy as np
import torch
from torch import nn


def gradients(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]


class DiscreteKdVPINN(nn.Module):
    def __init__(self, layers, lb, ub, lambda1, lambda2):
        """Initialize the discrete-time PINN network for the KdV equation."""
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))
        self.lambda1 = nn.Parameter(torch.tensor([lambda1], dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor([lambda2], dtype=torch.float32))
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x_norm = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_norm)


def gauss_legendre_collocation(q):
    """Return Gauss-Legendre collocation coefficients on [0, 1]."""
    nodes, _ = np.polynomial.legendre.leggauss(q)
    c = (nodes + 1.0) / 2.0
    a = np.zeros((q, q))
    b = np.zeros(q)
    for j in range(q):
        poly = np.poly1d([1.0])
        denom = 1.0
        for m in range(q):
            if m == j:
                continue
            poly *= np.poly1d([1.0, -c[m]])
            denom *= c[j] - c[m]
        poly /= denom
        poly_int = np.polyint(poly)
        b[j] = poly_int(1.0) - poly_int(0.0)
        for i in range(q):
            a[i, j] = poly_int(c[i]) - poly_int(0.0)
    return a, b, c


def kdv_rhs(u, k, lambda1, lambda2):
    u_hat = np.fft.fft(u)
    u_x = np.fft.ifft(1j * k * u_hat).real
    u_xxx = np.fft.ifft((1j * k) ** 3 * u_hat).real
    return -(lambda1 * u * u_x + lambda2 * u_xxx)


def solve_kdv(t_data, t_target, dt, nx, x_min, x_max, lambda1, lambda2):
    if t_target <= t_data:
        raise ValueError(
            f"t_target ({t_target}) must be larger than t_data ({t_data})."
        )
    length = x_max - x_min
    x = np.linspace(x_min, x_max, nx, endpoint=False)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    n_target = int(round(t_target / dt))
    if n_target < 1:
        raise ValueError(
            f"solver_dt ({dt}) is too large for target time {t_target}; "
            "must result in at least 1 time step."
        )
    dt = t_target / n_target
    n_data = int(round(t_data / dt))
    t_data_actual = n_data * dt
    if abs(t_data_actual - t_data) > 1e-12:
        print(
            f"Warning: t_data adjusted from {t_data:.6f} to {t_data_actual:.6f} "
            f"to align with dt={dt:.2e}."
        )
    u = np.cos(np.pi * x)
    u_data = u.copy() if n_data == 0 else None
    for step in range(1, n_target + 1):
        k1 = kdv_rhs(u, k, lambda1, lambda2)
        k2 = kdv_rhs(u + 0.5 * dt * k1, k, lambda1, lambda2)
        k3 = kdv_rhs(u + 0.5 * dt * k2, k, lambda1, lambda2)
        k4 = kdv_rhs(u + dt * k3, k, lambda1, lambda2)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if step == n_data:
            u_data = u.copy()
    return x, u_data, u


def periodic_interp(x, x_grid, u_grid, x_min, x_max):
    length = x_max - x_min
    x_mod = (x - x_min) % length + x_min
    x_ext = np.concatenate([x_grid, [x_max]])
    u_ext = np.concatenate([u_grid, [u_grid[0]]])
    return np.interp(x_mod, x_ext, u_ext)


def parse_layers(layers_str, output_dim):
    layers = [int(item) for item in layers_str.split(",") if item.strip()]
    if not layers:
        raise ValueError(
            "layers must contain at least 2 elements (input and output layer sizes)."
        )
    if layers[0] != 1:
        raise ValueError("input layer size must be 1 for x input.")
    if layers[-1] != output_dim:
        if layers[-1] <= 0:
            layers[-1] = output_dim
        else:
            print(
                f"Warning: adjusting output layer from {layers[-1]} to {output_dim} "
                "to match q+1 outputs."
            )
            layers[-1] = output_dim
    return layers


def to_tensor(array, device, requires_grad=False):
    return torch.tensor(array, dtype=torch.float32, device=device, requires_grad=requires_grad)


def derivatives_per_output(outputs, x):
    grads = []
    for idx in range(outputs.shape[1]):
        grads.append(gradients(outputs[:, idx : idx + 1], x))
    return torch.cat(grads, dim=1)


def add_noise(values, noise_level, rng):
    if noise_level <= 0.0:
        return values
    scale = np.std(values)
    if scale == 0.0:
        return values
    return values + noise_level * scale * rng.standard_normal(values.shape)


def build_training_data(
    x_grid,
    u_data,
    u_target,
    n_data,
    n_target,
    rng,
    x_min,
    x_max,
    noise_level,
):
    x_n = rng.uniform(x_min, x_max, size=(n_data, 1))
    u_n = periodic_interp(x_n[:, 0], x_grid, u_data, x_min, x_max)[:, None]
    x_np1 = rng.uniform(x_min, x_max, size=(n_target, 1))
    u_np1 = periodic_interp(x_np1[:, 0], x_grid, u_target, x_min, x_max)[:, None]
    u_n = add_noise(u_n, noise_level, rng)
    u_np1 = add_noise(u_np1, noise_level, rng)
    return x_n, u_n, x_np1, u_np1


def compute_collocation_stages(dt, eps):
    if not (0.0 < dt < 1.0):
        raise ValueError(
            "dt must be in (0, 1) to apply Eq. (28), since log(dt) must be negative."
        )
    if not (0.0 < eps < 1.0):
        raise ValueError(
            "rk-eps must be in (0, 1) to apply Eq. (28) for the RK stage count."
        )
    q = int(math.ceil(0.5 * math.log(eps) / math.log(dt)))
    return max(q, 1)


def train(model, data, rk_coeffs, config, device):
    x_n = to_tensor(data["x_n"], device, requires_grad=True)
    u_n = to_tensor(data["u_n"], device)
    x_np1 = to_tensor(data["x_np1"], device, requires_grad=True)
    u_np1 = to_tensor(data["u_np1"], device)
    x_left = to_tensor([[data["x_min"]]], device, requires_grad=True)
    x_right = to_tensor([[data["x_max"]]], device, requires_grad=True)
    a = rk_coeffs["a"].to(device)
    b = rk_coeffs["b"].to(device)
    dt = rk_coeffs["dt"]
    q = a.shape[0]

    def loss_fn():
        outputs = model(x_n)
        u_stage = outputs[:, :q]
        u_next = outputs[:, q : q + 1]
        u_x_stage = derivatives_per_output(u_stage, x_n)
        u_xx_stage = derivatives_per_output(u_x_stage, x_n)
        u_xxx_stage = derivatives_per_output(u_xx_stage, x_n)
        rhs_stage = -(model.lambda1 * u_stage * u_x_stage + model.lambda2 * u_xxx_stage)
        u_in = u_stage - dt * (rhs_stage @ a.T)
        u_next_pred = u_next - dt * (rhs_stage @ b)
        collocation = torch.cat([u_in, u_next_pred], dim=1)
        mse_n = torch.mean((collocation - u_n) ** 2)

        outputs_np1 = model(x_np1)
        u_np1_pred = outputs_np1[:, q : q + 1]
        mse_np1 = torch.mean((u_np1_pred - u_np1) ** 2)

        u_left = model(x_left)
        u_right = model(x_right)
        u_x_left = derivatives_per_output(u_left, x_left)
        u_x_right = derivatives_per_output(u_right, x_right)
        u_xx_left = derivatives_per_output(u_x_left, x_left)
        u_xx_right = derivatives_per_output(u_x_right, x_right)
        mse_b = (
            torch.mean((u_left - u_right) ** 2)
            + torch.mean((u_x_left - u_x_right) ** 2)
            + torch.mean((u_xx_left - u_xx_right) ** 2)
        )
        loss = mse_n + mse_np1 + mse_b
        return loss, mse_n, mse_np1, mse_b

    if config["adam_steps"] > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"])
        for step in range(config["adam_steps"]):
            optimizer.zero_grad()
            loss, mse_n, mse_np1, mse_b = loss_fn()
            loss.backward()
            optimizer.step()
            if (step + 1) % config["log_every"] == 0:
                print(
                    f"Adam step {step + 1:05d} | loss {loss.item():.3e} | "
                    f"mse_n {mse_n.item():.3e} | mse_np1 {mse_np1.item():.3e} | "
                    f"mse_b {mse_b.item():.3e} | "
                    f"lambda1 {model.lambda1.item():.4f} | lambda2 {model.lambda2.item():.4f}"
                )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config["lbfgs_max_iter"],
        tolerance_grad=config["lbfgs_tol_grad"],
        tolerance_change=config["lbfgs_tol_change"],
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    state = {"iter": 0}

    def closure():
        optimizer.zero_grad()
        loss, mse_n, mse_np1, mse_b = loss_fn()
        loss.backward()
        if state["iter"] % config["log_every"] == 0:
            print(
                f"LBFGS iter {state['iter']:05d} | loss {loss.item():.3e} | "
                f"mse_n {mse_n.item():.3e} | mse_np1 {mse_np1.item():.3e} | "
                f"mse_b {mse_b.item():.3e} | "
                f"lambda1 {model.lambda1.item():.4f} | lambda2 {model.lambda2.item():.4f}"
            )
        state["iter"] += 1
        return loss

    start = time.time()
    optimizer.step(closure)
    print(f"Training finished in {time.time() - start:.1f}s")


def evaluate(model, x_grid, u_exact, device, save_path, lambda_true):
    x_tensor = torch.tensor(x_grid[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(x_tensor).cpu().numpy()
    u_pred = outputs[:, -1]
    if u_exact is not None:
        rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
        print(f"Relative L2 error at target time: {rel_l2:.3e}")
    lambda1 = model.lambda1.item()
    lambda2 = model.lambda2.item()
    print(f"Estimated lambda1={lambda1:.6f}, lambda2={lambda2:.6f}")
    if lambda_true is not None:
        lambda1_true, lambda2_true = lambda_true
        err1 = abs(lambda1 - lambda1_true) / abs(lambda1_true)
        err2 = abs(lambda2 - lambda2_true) / abs(lambda2_true)
        print(f"Relative lambda errors: {err1:.3e}, {err2:.3e}")
    if save_path:
        np.savez_compressed(
            save_path,
            x=x_grid,
            u_pred=u_pred,
            u_exact=u_exact,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        print(f"Saved predictions to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discrete-time PINN for the Korteweg-de Vries equation (periodic BCs)."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument(
        "--q",
        type=int,
        default=0,
        help="Number of Gauss-Legendre collocation stages (0 to auto-compute).",
    )
    parser.add_argument(
        "--rk-eps",
        type=float,
        default=1e-10,
        help="Error tolerance for the RK stage count formula.",
    )
    parser.add_argument(
        "--t-data",
        type=float,
        default=0.2,
        help="Time at which training data is sampled.",
    )
    parser.add_argument(
        "--t-target",
        type=float,
        default=0.8,
        help="Time at which the solution is predicted.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.6,
        help=(
            "Time interval for the discrete-time PINN (must equal t-target - t-data; "
            "distinct from solver-dt)."
        ),
    )
    parser.add_argument(
        "--n-data",
        type=int,
        default=199,
        help="Number of spatial training samples at t-data.",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=201,
        help="Number of spatial training samples at t-target.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=-1.0,
        help="Left boundary of the spatial domain.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=1.0,
        help="Right boundary of the spatial domain.",
    )
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=1e-6,
        help="Time step for the spectral reference solver.",
    )
    parser.add_argument(
        "--solver-nx",
        type=int,
        default=512,
        help="Number of spatial grid points in the spectral solver.",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=6.0,
        help="True lambda1 value used to generate training data.",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=1.0,
        help="True lambda2 value used to generate training data.",
    )
    parser.add_argument(
        "--lambda1-init",
        type=float,
        default=1.0,
        help="Initial guess for lambda1.",
    )
    parser.add_argument(
        "--lambda2-init",
        type=float,
        default=1.0,
        help="Initial guess for lambda2.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Relative noise level to add to training data.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="1,50,50,50,50,0",
        help="Comma-separated layer sizes for the neural network.",
    )
    parser.add_argument(
        "--adam-steps",
        type=int,
        default=0,
        help="Number of Adam steps (0 to skip).",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=500,
        help="Maximum iterations for the L-BFGS optimizer.",
    )
    parser.add_argument(
        "--lbfgs-tol-grad",
        type=float,
        default=1e-7,
        help="Gradient tolerance for L-BFGS.",
    )
    parser.add_argument(
        "--lbfgs-tol-change",
        type=float,
        default=1e-9,
        help="Parameter change tolerance for L-BFGS.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Logging frequency (in optimization steps).",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default="",
        help="Output path for predictions (leave empty to skip).",
    )
    args = parser.parse_args()

    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")
    t_target = args.t_target
    interval = t_target - args.t_data
    if not np.isclose(interval, args.dt):
        raise ValueError(
            f"dt must equal t-target - t-data (expected {interval:.3f})."
        )
    if args.n_data <= 0 or args.n_target <= 0:
        raise ValueError("n-data and n-target must be positive.")
    if args.noise < 0.0:
        raise ValueError("noise must be non-negative.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x_grid, u_data_grid, u_target_grid = solve_kdv(
        args.t_data,
        t_target,
        args.solver_dt,
        args.solver_nx,
        args.x_min,
        args.x_max,
        args.lambda1,
        args.lambda2,
    )
    x_n, u_n, x_np1, u_np1 = build_training_data(
        x_grid,
        u_data_grid,
        u_target_grid,
        args.n_data,
        args.n_target,
        rng,
        args.x_min,
        args.x_max,
        args.noise,
    )

    if args.q <= 0:
        q = compute_collocation_stages(args.dt, args.rk_eps)
        print(f"Computed q={q} using rk-eps={args.rk_eps:.1e} and dt={args.dt:.3f}")
    else:
        q = args.q

    a_np, b_np, _ = gauss_legendre_collocation(q)
    b_np = b_np.reshape(-1, 1)
    rk_coeffs = {
        "a": torch.tensor(a_np, dtype=torch.float32),
        "b": torch.tensor(b_np, dtype=torch.float32),
        "dt": args.dt,
    }

    layers = parse_layers(args.layers, q + 1)
    device = torch.device(args.device)
    model = DiscreteKdVPINN(
        layers,
        lb=[args.x_min],
        ub=[args.x_max],
        lambda1=args.lambda1_init,
        lambda2=args.lambda2_init,
    ).to(device)
    config = {
        "adam_steps": args.adam_steps,
        "adam_lr": args.adam_lr,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "lbfgs_tol_grad": args.lbfgs_tol_grad,
        "lbfgs_tol_change": args.lbfgs_tol_change,
        "log_every": args.log_every,
    }
    data = {
        "x_n": x_n,
        "u_n": u_n,
        "x_np1": x_np1,
        "u_np1": u_np1,
        "x_min": args.x_min,
        "x_max": args.x_max,
    }

    train(model, data, rk_coeffs, config, device)
    evaluate(
        model,
        x_grid,
        u_target_grid,
        device,
        save_path=args.save_predictions or None,
        lambda_true=(args.lambda1, args.lambda2),
    )


if __name__ == "__main__":
    main()
