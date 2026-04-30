import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS TO IMPLEMENT
# ─────────────────────────────────────────────────────────────────────────────
# The following two helper functions will be used in Tasks A, B and C.
# Implement them correctly once, and they'll work for all tasks!


def _gd_1d(grad_fn, x0, lr, num_iters):
    """Perform 1D gradient descent.

    Returns array of x after each step (length num_iters).
    """
    x = float(x0)
    traj = np.empty(num_iters)

    for i in range(num_iters):
        x = x - lr * grad_fn(x)
        traj[i] = x

    return traj


def _gd_2d(grad_fn, u0, v0, lr, num_iters):
    """Perform 2D gradient descent.

    Returns (u_traj, v_traj) as two numpy arrays, each length num_iters.
    """
    u, v = float(u0), float(v0)
    ut = np.empty(num_iters)
    vt = np.empty(num_iters)

    for i in range(num_iters):
        gu, gv = grad_fn(u, v)
        u = u - lr * gu
        v = v - lr * gv
        ut[i] = u
        vt[i] = v

    return ut, vt


# Quick test for you to verify your 1D helper functions.
# if __name__ == "__main__":
#     # Test 1D with simple function f(x)=x^2, df(x)=2x
#     def test_df(x):
#         return 2 * x
#     test_traj = _gd_1d(test_df, 10.0, 0.1, 5)
#     print("1D test:", test_traj)  # Should see decreasing values

# You can design another test to verify your 2D helper function as well.


# ─────────────────────────────────────────────────────────────────────────────
# Please replace "StudentMatriculationNumber" with your actual matric number
# in BOTH the filename and the function name below.
# e.g. if your matric number is A1234567R:
#   filename : A3_A1234567R.py
#   function : def A3_A1234567R(task, params)
# ─────────────────────────────────────────────────────────────────────────────
def A3_A0254465Y(task: str, params: dict) -> dict:
    """
    Input
    -----
    task   : str   one of "A", "B", "C"
    params : dict  task-specific inputs (see below)

    Returns
    -------
    dict of task-specific outputs (keys must match exactly)
    """

    if task == "A":
        return _task_A(params)
    elif task == "B":
        return _task_B(params)
    elif task == "C":
        return _task_C(params)
    else:
        raise ValueError(f"Unknown task '{task}'. Must be one of A, B, C.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK A
# Cost function : f(x) = (x - 3)^2 + 5
# Initialisation: x0 = 10.0
# ─────────────────────────────────────────────────────────────────────────────


def _task_A(params):
    lr_list = params["lr_list"]
    num_iters = params["num_iters"]

    X0 = 10.0  # fixed initialisation — do not change

    # --- cost function and gradient ---
    def f(x):
        return (x - 3) ** 2 + 5

    def df(x):
        return 2 * (x - 3)

    trajectories = {}
    final_f = {}

    for lr in lr_list:
        traj = _gd_1d(df, X0, lr, num_iters)
        trajectories[lr] = traj
        final_f[lr] = float(f(traj[-1]))

    # --- analysis outputs ---
    diverging_lrs = sorted([lr for lr in lr_list if abs(trajectories[lr][-1]) > 1e6])
    converged_lrs = sorted([lr for lr in lr_list if final_f[lr] < 5.1])

    f_final_slow = round(final_f[0.001], 2)

    return {
        "trajectories": trajectories,
        "final_f": final_f,
        "diverging_lrs": diverging_lrs,
        "converged_lrs": converged_lrs,
        "f_final_slow": f_final_slow,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK B
# Cost function : f(x) = x^4 - 8x^2 + x + 10
# ─────────────────────────────────────────────────────────────────────────────


def _task_B(params):
    init_list = params["init_list"]
    lr = params["lr"]
    num_iters = params["num_iters"]

    # --- cost function and gradient ---
    def f(x):
        return x**4 - 8 * x**2 + x + 10

    def df(x):
        return 4 * x**3 - 16 * x + 1

    final_x = {}
    final_f = {}
    trajectories = {}

    for x0 in init_list:
        traj = _gd_1d(df, x0, lr, num_iters)
        trajectories[x0] = traj
        final_x[x0] = float(traj[-1])
        final_f[x0] = float(f(traj[-1]))

    # --- analysis outputs ---
    left_basin_inits = sorted([x0 for x0 in init_list if final_x[x0] < 0])

    left_fs = [final_f[x0] for x0 in init_list if final_x[x0] < 0]
    right_fs = [final_f[x0] for x0 in init_list if final_x[x0] >= 0]
    left_mean = float(np.mean(left_fs)) if len(left_fs) > 0 else float("inf")
    right_mean = float(np.mean(right_fs)) if len(right_fs) > 0 else float("inf")
    global_min_side = "left" if left_mean < right_mean else "right"

    best_f = round(min(final_f.values()), 2)

    return {
        "final_x": final_x,
        "final_f": final_f,
        "trajectories": trajectories,
        "left_basin_inits": left_basin_inits,
        "global_min_side": global_min_side,
        "best_f": best_f,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK C
# Cost function : f(u, v) = sin(u)*cos(v) + 0.1*(u^2 + v^2)
# ─────────────────────────────────────────────────────────────────────────────


def _task_C(params):
    init_pairs = params["init_pairs"]
    lr = params["lr"]
    num_iters = params["num_iters"]

    # --- cost function and gradients ---
    def f(u, v):
        return np.sin(u) * np.cos(v) + 0.1 * (u**2 + v**2)

    def df_du(u, v):
        return np.cos(u) * np.cos(v) + 0.2 * u

    def df_dv(u, v):
        return -np.sin(u) * np.sin(v) + 0.2 * v

    final_uv = {}
    final_f = {}
    u_traj = {}  # no need to return, for debugging only
    v_traj = {}  # no need to return, for debugging only

    def grad_fn(u, v):
        return (df_du(u, v), df_dv(u, v))

    for pair in init_pairs:
        u0, v0 = pair[0], pair[1]
        key = str(pair)  # e.g. "[0.1, 0.1]"  — use this as the dict key

        ut, vt = _gd_2d(grad_fn, u0, v0, lr, num_iters)

        u_traj[key] = ut
        v_traj[key] = vt
        final_uv[key] = [float(ut[-1]), float(vt[-1])]
        final_f[key] = float(f(ut[-1], vt[-1]))

    # --- analysis outputs ---
    rounded_f = {k: round(v, 1) for k, v in final_f.items()}
    num_basins = len(set(rounded_f.values()))

    key0 = str([0.1, 0.1])
    same_basin_as_first = sorted(
        [k for k in rounded_f if k != key0 and rounded_f[k] == rounded_f[key0]]
    )

    best_init = min(final_f.items(), key=lambda kv: kv[1])[0]

    return {
        "final_uv": final_uv,
        "final_f": final_f,
        "num_basins": num_basins,
        "same_basin_as_first": same_basin_as_first,
        "best_init": best_init,
    }
