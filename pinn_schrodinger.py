#!/usr/bin/env python3
import argparse
import math
import time

import numpy as np
import torch
from torch import nn


def latin_hypercube(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.random((n_samples, dim))
    points = np.zeros_like(u)
    for j in range(dim):
        points[:, j] = cut[:n_samples] + u[:, j] * (cut[1:] - cut[:n_samples])
        rng.shuffle(points[:, j])
    return points


def sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x)


class PINN(nn.Module):
    def __init__(self, layers, lb, ub):
        """Initialize the PINN network and normalization bounds."""
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

    def forward(self, t, x):
        x_in = torch.cat([t, x], dim=1)
        x_norm = 2.0 * (x_in - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_norm)


def gradients(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]


def net_uv_and_grads(model, t, x):
    uv = model(t, x)
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    u_x = gradients(u, x)
    v_x = gradients(v, x)
    return u, v, u_x, v_x


def schrodinger_residual(model, t, x):
    uv = model(t, x)
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    u_t = gradients(u, t)
    v_t = gradients(v, t)
    u_x = gradients(u, x)
    v_x = gradients(v, x)
    u_xx = gradients(u_x, x)
    v_xx = gradients(v_x, x)
    h_sq = u**2 + v**2
    f_u = -v_t + 0.5 * u_xx + h_sq * u
    f_v = u_t + 0.5 * v_xx + h_sq * v
    return f_u, f_v


def parse_layers(layers_str: str):
    """Parse a comma-separated list of integers into layer dimensions."""
    return [int(item) for item in layers_str.split(",") if item.strip()]


def build_training_data(n0, nb, nf, bounds, seed):
    """Generate training data for initial, boundary, and residual points."""
    rng = np.random.default_rng(seed)
    x_min, x_max, t_min, t_max = bounds
    x0 = x_min + (x_max - x_min) * latin_hypercube(n0, 1, rng)
    t0 = np.zeros_like(x0)
    h0 = 2.0 * sech(x0)
    u0 = h0
    v0 = np.zeros_like(h0)

    tb = t_min + (t_max - t_min) * latin_hypercube(nb, 1, rng)
    x_lb = x_min * np.ones_like(tb)
    x_ub = x_max * np.ones_like(tb)

    tf_xf = latin_hypercube(nf, 2, rng)
    tf = t_min + (t_max - t_min) * tf_xf[:, 0:1]
    xf = x_min + (x_max - x_min) * tf_xf[:, 1:2]

    return {
        "t0": t0,
        "x0": x0,
        "u0": u0,
        "v0": v0,
        "tb": tb,
        "x_lb": x_lb,
        "x_ub": x_ub,
        "tf": tf,
        "xf": xf,
    }


def to_tensor(array, device):
    return torch.tensor(array, dtype=torch.float32, device=device)


def train(model, data, config, device):
    """Train the PINN with optional Adam warmup and L-BFGS optimization."""
    t0 = to_tensor(data["t0"], device)
    x0 = to_tensor(data["x0"], device)
    u0 = to_tensor(data["u0"], device)
    v0 = to_tensor(data["v0"], device)
    tb = to_tensor(data["tb"], device).requires_grad_(True)
    x_lb = to_tensor(data["x_lb"], device).requires_grad_(True)
    x_ub = to_tensor(data["x_ub"], device).requires_grad_(True)
    tf = to_tensor(data["tf"], device).requires_grad_(True)
    xf = to_tensor(data["xf"], device).requires_grad_(True)

    def loss_fn():
        uv0 = model(t0, x0)
        u0_pred = uv0[:, 0:1]
        v0_pred = uv0[:, 1:2]
        mse0 = torch.mean((u0_pred - u0) ** 2 + (v0_pred - v0) ** 2)

        u_lb, v_lb, u_x_lb, v_x_lb = net_uv_and_grads(model, tb, x_lb)
        u_ub, v_ub, u_x_ub, v_x_ub = net_uv_and_grads(model, tb, x_ub)
        mse_b = torch.mean(
            (u_lb - u_ub) ** 2
            + (v_lb - v_ub) ** 2
            + (u_x_lb - u_x_ub) ** 2
            + (v_x_lb - v_x_ub) ** 2
        )

        f_u, f_v = schrodinger_residual(model, tf, xf)
        mse_f = torch.mean(f_u**2 + f_v**2)

        loss = mse0 + mse_b + mse_f
        return loss, mse0, mse_b, mse_f

    if config["adam_steps"] > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"])
        for step in range(config["adam_steps"]):
            optimizer.zero_grad()
            loss, mse0, mse_b, mse_f = loss_fn()
            loss.backward()
            optimizer.step()
            if (step + 1) % config["log_every"] == 0:
                print(
                    f"Adam step {step + 1:05d} | loss {loss.item():.3e} | "
                    f"mse0 {mse0.item():.3e} | mseb {mse_b.item():.3e} | msef {mse_f.item():.3e}"
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
        loss, mse0, mse_b, mse_f = loss_fn()
        loss.backward()
        if state["iter"] % config["log_every"] == 0:
            print(
                f"LBFGS iter {state['iter']:05d} | loss {loss.item():.3e} | "
                f"mse0 {mse0.item():.3e} | mseb {mse_b.item():.3e} | msef {mse_f.item():.3e}"
            )
        state["iter"] += 1
        return loss

    start = time.time()
    optimizer.step(closure)
    print(f"Training finished in {time.time() - start:.1f}s")


def evaluate(model, bounds, device, n_t, n_x, save_path):
    """Evaluate the trained model on a grid and optionally save predictions."""
    x_min, x_max, t_min, t_max = bounds
    t = np.linspace(t_min, t_max, n_t)
    x = np.linspace(x_min, x_max, n_x)
    t_grid, x_grid = np.meshgrid(t, x, indexing="ij")
    tx = np.stack([t_grid.reshape(-1), x_grid.reshape(-1)], axis=1)
    t_tensor = torch.tensor(tx[:, 0:1], dtype=torch.float32, device=device)
    x_tensor = torch.tensor(tx[:, 1:2], dtype=torch.float32, device=device)
    with torch.no_grad():
        uv = model(t_tensor, x_tensor).cpu().numpy()
    u = uv[:, 0].reshape(n_t, n_x)
    v = uv[:, 1].reshape(n_t, n_x)
    amp = np.sqrt(u**2 + v**2)
    if save_path:
        np.savez_compressed(
            save_path,
            t=t,
            x=x,
            u=u,
            v=v,
            amplitude=amp,
        )
        print(f"Saved predictions to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PINN for the nonlinear Schrödinger equation.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--nb", type=int, default=50)
    parser.add_argument("--nf", type=int, default=20000)
    parser.add_argument("--t-max", type=float, default=math.pi / 2.0)
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--layers", type=str, default="2,100,100,100,100,100,2")
    parser.add_argument("--adam-steps", type=int, default=0)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs-max-iter", type=int, default=500)
    parser.add_argument("--lbfgs-tol-grad", type=float, default=1e-7)
    parser.add_argument("--lbfgs-tol-change", type=float, default=1e-9)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-nt", type=int, default=100)
    parser.add_argument("--eval-nx", type=int, default=256)
    parser.add_argument("--save-predictions", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_min = 0.0
    bounds = (args.x_min, args.x_max, t_min, args.t_max)
    data = build_training_data(args.n0, args.nb, args.nf, bounds, args.seed)
    layers = parse_layers(args.layers)
    device = torch.device(args.device)

    model = PINN(layers, lb=[t_min, args.x_min], ub=[args.t_max, args.x_max]).to(device)
    config = {
        "adam_steps": args.adam_steps,
        "adam_lr": args.adam_lr,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "lbfgs_tol_grad": args.lbfgs_tol_grad,
        "lbfgs_tol_change": args.lbfgs_tol_change,
        "log_every": args.log_every,
    }

    train(model, data, config, device)
    evaluate(
        model,
        bounds,
        device,
        n_t=args.eval_nt,
        n_x=args.eval_nx,
        save_path=args.save_predictions or None,
    )


if __name__ == "__main__":
    main()
