#!/usr/bin/env python3
import argparse
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


class NavierStokesPINN(nn.Module):
    def __init__(self, layers, lb, ub, lambda1, lambda2):
        """Initialize the Navier-Stokes PINN network."""
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

    def forward(self, t, x, y):
        inputs = torch.cat([t, x, y], dim=1)
        x_norm = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_norm)


def parse_layers(layers_str: str):
    layers = [int(item) for item in layers_str.split(",") if item.strip()]
    if not layers:
        raise ValueError("layers must contain at least input and output sizes.")
    if layers[0] != 3:
        raise ValueError("input layer size must be 3 for (t, x, y) input.")
    if layers[-1] != 2:
        print("Warning: adjusting output layer size to 2 for (psi, p) outputs.")
        layers[-1] = 2
    return layers


def to_numpy(data):
    return np.asarray(data)


def maybe_load_mat(path):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load .mat files. Convert the dataset to .npz "
            "or install scipy."
        ) from exc
    raw = loadmat(path)
    return {key: value for key, value in raw.items() if not key.startswith("__")}


def load_dataset(path):
    if path.endswith(".npz"):
        raw = np.load(path)
        return {key: raw[key] for key in raw.files}
    if path.endswith(".mat"):
        return maybe_load_mat(path)
    raise ValueError("Unsupported dataset format. Use .npz or .mat.")


def pick_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def normalize_velocity(u_raw, v_raw):
    u = to_numpy(u_raw)
    v = to_numpy(v_raw) if v_raw is not None else None
    if v is None and u.ndim == 3:
        if u.shape[1] == 2:
            u, v = u[:, 0, :], u[:, 1, :]
        elif u.shape[2] == 2:
            u, v = u[:, :, 0], u[:, :, 1]
    if v is None:
        raise ValueError("Velocity data must include both u and v components.")
    return to_numpy(u), to_numpy(v)


def flatten_samples(t, x, y, u, v, p=None):
    t = to_numpy(t).reshape(-1)
    x = to_numpy(x).reshape(-1)
    y = to_numpy(y).reshape(-1)
    u = to_numpy(u)
    v = to_numpy(v)
    if u.ndim == 2 and u.shape[1] == 1 and v.ndim == 2 and v.shape[1] == 1:
        u = u.reshape(-1)
        v = v.reshape(-1)
        if p is not None:
            p = to_numpy(p)
            if p.ndim == 2 and p.shape[1] == 1:
                p = p.reshape(-1)
    if u.ndim == 1:
        if not (u.size == v.size == t.size == x.size == y.size):
            raise ValueError("Flattened arrays must share the same length.")
        p_flat = to_numpy(p).reshape(-1) if p is not None else None
        return (
            t.reshape(-1, 1),
            x.reshape(-1, 1),
            y.reshape(-1, 1),
            u.reshape(-1, 1),
            v.reshape(-1, 1),
            p_flat.reshape(-1, 1) if p_flat is not None else None,
        )

    if u.ndim == 2:
        if v.shape != u.shape:
            raise ValueError("u and v arrays must share the same shape.")
        if u.shape[0] == x.size and u.shape[1] == t.size:
            u_space_time = u
            v_space_time = v
        elif u.shape[0] == t.size and u.shape[1] == x.size:
            u_space_time = u.T
            v_space_time = v.T
        else:
            raise ValueError("Unable to align velocity data with coordinates.")
        n_points, n_times = u_space_time.shape
        if y.size != n_points:
            raise ValueError("x/y coordinate arrays must match spatial dimension.")
        t_grid = np.repeat(t[None, :], n_points, axis=0)
        x_grid = np.repeat(x[:, None], n_times, axis=1)
        y_grid = np.repeat(y[:, None], n_times, axis=1)
        p_flat = None
        if p is not None:
            p_arr = to_numpy(p)
            if p_arr.ndim == 2 and p_arr.shape == u_space_time.shape:
                p_flat = p_arr.reshape(-1, 1)
            elif p_arr.ndim == 1 and p_arr.size == u_space_time.size:
                p_flat = p_arr.reshape(-1, 1)
        return (
            t_grid.reshape(-1, 1),
            x_grid.reshape(-1, 1),
            y_grid.reshape(-1, 1),
            u_space_time.reshape(-1, 1),
            v_space_time.reshape(-1, 1),
            p_flat,
        )

    raise ValueError("Velocity arrays must be 1D or 2D.")


def extract_flow_data(data):
    t_key = pick_key(data, ["t", "t_star", "T"])
    if t_key is None:
        raise ValueError("Dataset must contain time array 't'.")
    t = data[t_key]

    if "X_star" in data:
        xy = to_numpy(data["X_star"])
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError("X_star must have shape (N, 2).")
        x = xy[:, 0]
        y = xy[:, 1]
    else:
        x_key = pick_key(data, ["x", "X"])
        y_key = pick_key(data, ["y", "Y"])
        if x_key is None or y_key is None:
            raise ValueError("Dataset must contain x/y coordinates or X_star.")
        x = data[x_key]
        y = data[y_key]

    u_key = pick_key(data, ["u", "U", "U_star"])
    v_key = pick_key(data, ["v", "V", "V_star"])
    if u_key is None:
        raise ValueError("Dataset must contain velocity data.")
    u_raw = data[u_key]
    v_raw = data[v_key] if v_key is not None else None
    u, v = normalize_velocity(u_raw, v_raw)
    p_key = pick_key(data, ["p", "P", "P_star"])
    p_raw = data[p_key] if p_key is not None else None
    return flatten_samples(t, x, y, u, v, p_raw)


def apply_bounds(t, x, y, u, v, p, bounds):
    t_min, t_max, x_min, x_max, y_min, y_max = bounds
    mask = (
        (t >= t_min)
        & (t <= t_max)
        & (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
    )
    t = t[mask]
    x = x[mask]
    y = y[mask]
    u = u[mask]
    v = v[mask]
    p = p[mask] if p is not None else None
    return t, x, y, u, v, p


def sample_training_data(t, x, y, u, v, p, n_train, noise, seed):
    rng = np.random.default_rng(seed)
    total = t.shape[0]
    sample_with_replacement = total < n_train
    idx = rng.choice(total, size=n_train, replace=sample_with_replacement)
    t_train = t[idx].reshape(-1, 1)
    x_train = x[idx].reshape(-1, 1)
    y_train = y[idx].reshape(-1, 1)
    u_train = u[idx].reshape(-1, 1)
    v_train = v[idx].reshape(-1, 1)
    if noise > 0.0:
        u_std = float(np.std(u_train))
        v_std = float(np.std(v_train))
        u_train = u_train + noise * u_std * rng.standard_normal(u_train.shape)
        v_train = v_train + noise * v_std * rng.standard_normal(v_train.shape)
    p_train = p[idx].reshape(-1, 1) if p is not None else None
    return {
        "t": t_train,
        "x": x_train,
        "y": y_train,
        "u": u_train,
        "v": v_train,
        "p": p_train,
    }


def to_tensor(array, device, requires_grad=False):
    return torch.tensor(array, dtype=torch.float32, device=device, requires_grad=requires_grad)


def net_uvp_and_residual(model, t, x, y):
    """Return velocity, pressure, and PDE residuals f/g for the Navier-Stokes system."""
    psi_p = model(t, x, y)
    psi = psi_p[:, 0:1]
    p = psi_p[:, 1:2]
    psi_x = gradients(psi, x)
    psi_y = gradients(psi, y)
    u = psi_y
    v = -psi_x
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)
    v_t = gradients(v, t)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    lambda1 = model.lambda1
    lambda2 = model.lambda2
    f = u_t + lambda1 * (u * u_x + v * u_y) + p_x - lambda2 * (u_xx + u_yy)
    g = v_t + lambda1 * (u * v_x + v * v_y) + p_y - lambda2 * (v_xx + v_yy)
    return u, v, p, f, g


def train(model, data, config, device):
    t_train = to_tensor(data["t"], device, requires_grad=True)
    x_train = to_tensor(data["x"], device, requires_grad=True)
    y_train = to_tensor(data["y"], device, requires_grad=True)
    u_train = to_tensor(data["u"], device)
    v_train = to_tensor(data["v"], device)

    def loss_fn():
        u_pred, v_pred, _, f, g = net_uvp_and_residual(model, t_train, x_train, y_train)
        mse_uv = torch.mean((u_pred - u_train) ** 2 + (v_pred - v_train) ** 2)
        mse_fg = torch.mean(f**2 + g**2)
        loss = mse_uv + mse_fg
        return loss, mse_uv, mse_fg

    if config["adam_steps"] > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"])
        for step in range(config["adam_steps"]):
            optimizer.zero_grad()
            loss, mse_uv, mse_fg = loss_fn()
            loss.backward()
            optimizer.step()
            if (step + 1) % config["log_every"] == 0:
                print(
                    f"Adam step {step + 1:05d} | loss {loss.item():.3e} | "
                    f"mse_uv {mse_uv.item():.3e} | mse_fg {mse_fg.item():.3e} | "
                    f"lambda1 {model.lambda1.item():.5f} | lambda2 {model.lambda2.item():.5f}"
                )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config["lbfgs_max_iter"],
        tolerance_grad=config["lbfgs_tol_grad"],
        tolerance_change=config["lbfgs_tol_change"],
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    lbfgs_state = {"iter": 0}

    def closure():
        optimizer.zero_grad()
        loss, mse_uv, mse_fg = loss_fn()
        loss.backward()
        if lbfgs_state["iter"] % config["log_every"] == 0:
            print(
                f"LBFGS iter {lbfgs_state['iter']:05d} | loss {loss.item():.3e} | "
                f"mse_uv {mse_uv.item():.3e} | mse_fg {mse_fg.item():.3e} | "
                f"lambda1 {model.lambda1.item():.5f} | lambda2 {model.lambda2.item():.5f}"
            )
        lbfgs_state["iter"] += 1
        return loss

    start = time.time()
    optimizer.step(closure)
    print(f"Training finished in {time.time() - start:.1f}s")


def evaluate(model, data, device, max_samples, save_path):
    t = data["t"].reshape(-1, 1)
    x = data["x"].reshape(-1, 1)
    y = data["y"].reshape(-1, 1)
    u_true = data["u"].reshape(-1, 1)
    v_true = data["v"].reshape(-1, 1)
    p_true = data["p"].reshape(-1, 1) if data["p"] is not None else None
    total = t.shape[0]
    if max_samples is not None and total > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(total, size=max_samples, replace=False)
        t = t[idx]
        x = x[idx]
        y = y[idx]
        u_true = u_true[idx]
        v_true = v_true[idx]
        p_true = p_true[idx] if p_true is not None else None

    t_tensor = to_tensor(t, device, requires_grad=True)
    x_tensor = to_tensor(x, device, requires_grad=True)
    y_tensor = to_tensor(y, device, requires_grad=True)
    u_pred, v_pred, p_pred, _, _ = net_uvp_and_residual(model, t_tensor, x_tensor, y_tensor)
    u_pred_np = u_pred.detach().cpu().numpy()
    v_pred_np = v_pred.detach().cpu().numpy()
    p_pred_np = p_pred.detach().cpu().numpy()

    def rel_l2(pred, truth):
        return np.linalg.norm(pred - truth) / np.linalg.norm(truth)

    if u_true is not None:
        print(f"Relative L2 error u: {rel_l2(u_pred_np, u_true):.3e}")
        print(f"Relative L2 error v: {rel_l2(v_pred_np, v_true):.3e}")

    print(f"Estimated lambda1: {model.lambda1.item():.6f}")
    print(f"Estimated lambda2: {model.lambda2.item():.6f}")

    if save_path:
        output = {
            "t": t,
            "x": x,
            "y": y,
            "u_pred": u_pred_np,
            "v_pred": v_pred_np,
            "p_pred": p_pred_np,
            "lambda1": np.array([model.lambda1.item()]),
            "lambda2": np.array([model.lambda2.item()]),
        }
        if u_true is not None:
            output["u_true"] = u_true
            output["v_true"] = v_true
        if p_true is not None:
            output["p_true"] = p_true
        np.savez_compressed(save_path, **output)
        print(f"Saved predictions to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PINN for the 2D Navier-Stokes example.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--layers", type=str, default="3,20,20,20,20,20,20,20,20,2")
    parser.add_argument("--lambda1-init", type=float, default=1.0)
    parser.add_argument("--lambda2-init", type=float, default=0.01)
    parser.add_argument("--adam-steps", type=int, default=0)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs-max-iter", type=int, default=500)
    parser.add_argument("--lbfgs-tol-grad", type=float, default=1e-7)
    parser.add_argument("--lbfgs-tol-change", type=float, default=1e-9)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-max-samples", type=int, default=20000)
    parser.add_argument("--save-predictions", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_dataset(args.data_path)
    t, x, y, u, v, p = extract_flow_data(dataset)

    t_min = np.min(t) if args.t_min is None else args.t_min
    t_max = np.max(t) if args.t_max is None else args.t_max
    x_min = np.min(x) if args.x_min is None else args.x_min
    x_max = np.max(x) if args.x_max is None else args.x_max
    y_min = np.min(y) if args.y_min is None else args.y_min
    y_max = np.max(y) if args.y_max is None else args.y_max
    t, x, y, u, v, p = apply_bounds(
        t,
        x,
        y,
        u,
        v,
        p,
        bounds=(t_min, t_max, x_min, x_max, y_min, y_max),
    )

    train_data = sample_training_data(t, x, y, u, v, p, args.n_train, args.noise, args.seed)
    layers = parse_layers(args.layers)
    device = torch.device(args.device)
    model = NavierStokesPINN(
        layers,
        lb=[t_min, x_min, y_min],
        ub=[t_max, x_max, y_max],
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
    train(model, train_data, config, device)

    eval_data = {"t": t, "x": x, "y": y, "u": u, "v": v, "p": p}
    evaluate(
        model,
        eval_data,
        device,
        max_samples=args.eval_max_samples if args.eval_max_samples > 0 else None,
        save_path=args.save_predictions or None,
    )


if __name__ == "__main__":
    main()
