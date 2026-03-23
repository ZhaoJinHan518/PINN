#!/usr/bin/env python3
import argparse
import time

import numpy as np
import torch
from torch import nn

DIFFUSIVITY = 0.0001


def gradients(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]


class DiscretePINN(nn.Module):
    def __init__(self, layers, lb, ub):
        """Initialize the discrete-time PINN network."""
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))
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


def allen_cahn_rhs(u, k_sq, diffusivity=DIFFUSIVITY):
    u_hat = np.fft.fft(u)
    u_xx = np.fft.ifft(-k_sq * u_hat).real
    return diffusivity * u_xx - 5.0 * u**3 + 5.0 * u


def solve_allen_cahn(t_data, t_target, dt, nx, x_min, x_max):
    if t_target <= t_data:
        raise ValueError("t_target must be larger than t_data.")
    length = x_max - x_min
    x = np.linspace(x_min, x_max, nx, endpoint=False)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    k_sq = k**2
    n_target = int(round(t_target / dt))
    if n_target < 1:
        raise ValueError("solver dt is too large for the requested target time.")
    dt = t_target / n_target
    n_data = int(round(t_data / dt))
    t_data_actual = n_data * dt
    if abs(t_data_actual - t_data) > 1e-12:
        print(
            f"Warning: t_data adjusted from {t_data:.6f} to {t_data_actual:.6f} "
            f"to align with dt={dt:.2e}."
        )
    u = x**2 * np.cos(np.pi * x)
    u_data = u.copy() if n_data == 0 else None
    for step in range(1, n_target + 1):
        k1 = allen_cahn_rhs(u, k_sq)
        k2 = allen_cahn_rhs(u + 0.5 * dt * k1, k_sq)
        k3 = allen_cahn_rhs(u + 0.5 * dt * k2, k_sq)
        k4 = allen_cahn_rhs(u + dt * k3, k_sq)
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
        raise ValueError("layers must contain at least input and output sizes.")
    if layers[0] != 1:
        raise ValueError("input layer size must be 1 for x input.")
    if layers[-1] != output_dim:
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


def build_training_data(x_grid, u_data, n_samples, rng, x_min, x_max):
    x_samples = rng.uniform(x_min, x_max, size=(n_samples, 1))
    u_samples = periodic_interp(x_samples[:, 0], x_grid, u_data, x_min, x_max)[:, None]
    return x_samples, u_samples


def train(model, data, rk_coeffs, config, device):
    x_n = to_tensor(data["x_n"], device, requires_grad=True)
    u_n = to_tensor(data["u_n"], device)
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
        rhs_stage = DIFFUSIVITY * u_xx_stage - 5.0 * u_stage**3 + 5.0 * u_stage
        u_in = u_stage - dt * (rhs_stage @ a.T)
        u_next_pred = u_next - dt * (rhs_stage @ b)
        collocation = torch.cat([u_in, u_next_pred], dim=1)
        mse_n = torch.mean((collocation - u_n) ** 2)

        u_left = model(x_left)
        u_right = model(x_right)
        u_x_left = derivatives_per_output(u_left, x_left)
        u_x_right = derivatives_per_output(u_right, x_right)
        mse_b = torch.mean((u_left - u_right) ** 2) + torch.mean((u_x_left - u_x_right) ** 2)
        loss = mse_n + mse_b
        return loss, mse_n, mse_b

    if config["adam_steps"] > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"])
        for step in range(config["adam_steps"]):
            optimizer.zero_grad()
            loss, mse_n, mse_b = loss_fn()
            loss.backward()
            optimizer.step()
            if (step + 1) % config["log_every"] == 0:
                print(
                    f"Adam step {step + 1:05d} | loss {loss.item():.3e} | "
                    f"mse_n {mse_n.item():.3e} | mse_b {mse_b.item():.3e}"
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
        loss, mse_n, mse_b = loss_fn()
        loss.backward()
        if state["iter"] % config["log_every"] == 0:
            print(
                f"LBFGS iter {state['iter']:05d} | loss {loss.item():.3e} | "
                f"mse_n {mse_n.item():.3e} | mse_b {mse_b.item():.3e}"
            )
        state["iter"] += 1
        return loss

    start = time.time()
    optimizer.step(closure)
    print(f"Training finished in {time.time() - start:.1f}s")


def evaluate(model, x_grid, u_exact, device, save_path):
    x_tensor = torch.tensor(x_grid[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(x_tensor).cpu().numpy()
    u_pred = outputs[:, -1]
    if u_exact is not None:
        rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
        print(f"Relative L2 error at target time: {rel_l2:.3e}")
    if save_path:
        np.savez_compressed(save_path, x=x_grid, u_pred=u_pred, u_exact=u_exact)
        print(f"Saved predictions to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discrete-time PINN for the Allen-Cahn equation (periodic BCs)."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument(
        "--q",
        type=int,
        default=100,
        help="Number of Gauss-Legendre collocation stages.",
    )
    parser.add_argument(
        "--t-data",
        type=float,
        default=0.1,
        help="Time at which training data is sampled.",
    )
    parser.add_argument(
        "--t-target",
        type=float,
        default=0.9,
        help="Time at which the solution is predicted.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.8,
        help=(
            "Time interval for the discrete-time PINN (must equal t-target - t-data; "
            "distinct from solver-dt)."
        ),
    )
    parser.add_argument(
        "--n-data",
        type=int,
        default=200,
        help="Number of spatial training samples.",
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
        default=1e-4,
        help="Time step for the spectral reference solver.",
    )
    parser.add_argument(
        "--solver-nx",
        type=int,
        default=512,
        help="Number of spatial grid points in the spectral solver.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="1,200,200,200,200,101",
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
    dt = args.dt
    t_target = args.t_target
    interval = t_target - args.t_data
    if not np.isclose(interval, dt):
        raise ValueError(
            f"dt must equal t-target - t-data (expected {interval:.3f})."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x_grid, u_data_grid, u_target_grid = solve_allen_cahn(
        args.t_data,
        t_target,
        args.solver_dt,
        args.solver_nx,
        args.x_min,
        args.x_max,
    )
    x_n, u_n = build_training_data(
        x_grid, u_data_grid, args.n_data, rng, args.x_min, args.x_max
    )

    a_np, b_np, _ = gauss_legendre_collocation(args.q)
    b_np = b_np.reshape(-1, 1)
    rk_coeffs = {
        "a": torch.tensor(a_np, dtype=torch.float32),
        "b": torch.tensor(b_np, dtype=torch.float32),
        "dt": dt,
    }

    layers = parse_layers(args.layers, args.q + 1)
    device = torch.device(args.device)
    model = DiscretePINN(layers, lb=[args.x_min], ub=[args.x_max]).to(device)
    config = {
        "adam_steps": args.adam_steps,
        "adam_lr": args.adam_lr,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "lbfgs_tol_grad": args.lbfgs_tol_grad,
        "lbfgs_tol_change": args.lbfgs_tol_change,
        "log_every": args.log_every,
    }
    data = {"x_n": x_n, "u_n": u_n, "x_min": args.x_min, "x_max": args.x_max}

    train(model, data, rk_coeffs, config, device)
    evaluate(
        model,
        x_grid,
        u_target_grid,
        device,
        save_path=args.save_predictions or None,
    )


if __name__ == "__main__":
    main()
