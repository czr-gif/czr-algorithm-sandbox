import numpy as np
import matplotlib

# 使用无 GUI 的后端，避免在无显示环境下崩溃
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Optimization_Manifold.problems.rayleigh_sphere import (
    RayleighQuotientSphereProblem,
)
from Optimization_Manifold.optimizers import RiemannianGradientDescent, RiemannianNewton
from typing import Optional


def run_rayleigh_on_sphere(
    dim: int = 10,
    step_size: float = 1e-2,
    max_iters: int = 1_000,
    tol: float = 1e-4,
    seed: Optional[int] = 42,
):
    """
    组合：
    - 流形：Sphere S^{n-1}
    - 问题：Rayleigh quotient minimization
    - 算法：Riemannian Gradient Descent (RGD)
    并绘制收敛曲线。
    """
    # 1. 构造测试问题（内部会创建 Sphere 流形和正定矩阵 A）
    problem = RayleighQuotientSphereProblem(dim=dim, seed=seed)
    manifold = problem.manifold

    # 2. 随机初始化一个球面上的起点
    x0 = problem.random_initial_point()

    # 3. 配置优化器
    optimizer = RiemannianGradientDescent(
        manifold=manifold,
        step_size=step_size,
        max_iters=max_iters,
        tol=tol,
    )

    # 4. 运行 RGD
    result = optimizer.solve(x0, problem)

    x_final = result["x"]
    f_final = result["f"]
    grad_norm_final = result["grad_norm"]
    history = result["history"]

    f_opt = problem.optimal_value()
    dist_to_opt = problem.distance_to_opt(x_final)

    print("=======================================")
    print("Riemannian Gradient Descent on Sphere")
    print("dim            :", dim)
    print("step_size      :", step_size)
    print("max_iters      :", max_iters)
    print("converged      :", result["converged"])
    print("final f(x)     :", f_final)
    print("optimal f*     :", f_opt)
    print("f(x) - f*      :", f_final - f_opt)
    print("||grad||_final :", grad_norm_final)
    print("dist(x, x*)    :", dist_to_opt)
    print("=======================================")

    # 5. 可视化收敛过程
    fs = np.array([h["f"] for h in history])
    grad_norms = np.array([h["grad_norm"] for h in history])
    iters = np.arange(len(history))

    # 避免负值或 0 带来的 log 问题
    gap = np.maximum(fs - f_opt, 1e-16)

    plt.figure(figsize=(10, 4))

    # (a) 目标函数值
    plt.subplot(1, 2, 1)
    plt.plot(iters, fs, label="f(x_k)")
    plt.axhline(f_opt, color="red", linestyle="--", label="f*")
    plt.xlabel("iteration")
    plt.ylabel("f(x)")
    plt.title("Rayleigh quotient on S^{n-1}")
    plt.legend()

    # (b) f(x_k) - f* 或 梯度范数（对数尺度）
    plt.subplot(1, 2, 2)
    plt.semilogy(iters, gap, label="f(x_k) - f*")
    plt.semilogy(iters, grad_norms, label="||grad||")
    plt.xlabel("iteration")
    plt.title("Convergence (log scale)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("rayleigh_sphere_rgd.png", dpi=150)  # 保存到当前目录
    # plt.show()  # 在非交互后端可以注释掉


def run_rayleigh_on_sphere_newton(
    dim: int = 10,
    max_iters: int = 50,
    tol: float = 1e-8,
    seed: Optional[int] = 42,
):
    """
    组合：
    - 流形：Sphere S^{n-1}
    - 问题：Rayleigh quotient minimization
    - 算法：Riemannian Newton Method
    并绘制收敛曲线。
    """
    problem = RayleighQuotientSphereProblem(dim=dim, seed=seed)
    manifold = problem.manifold
    x0 = problem.random_initial_point()

    optimizer = RiemannianNewton(
        manifold=manifold,
        max_iters=max_iters,
        tol=tol,
    )

    result = optimizer.solve(x0, problem)

    x_final = result["x"]
    f_final = result["f"]
    grad_norm_final = result["grad_norm"]
    history = result["history"]

    f_opt = problem.optimal_value()
    dist_to_opt = problem.distance_to_opt(x_final)

    print("=======================================")
    print("Riemannian Newton Method on Sphere")
    print("dim            :", dim)
    print("max_iters      :", max_iters)
    print("num_iters used :", result["num_iters"])
    print("converged      :", result["converged"])
    print("final f(x)     :", f_final)
    print("optimal f*     :", f_opt)
    print("f(x) - f*      :", f_final - f_opt)
    print("||grad||_final :", grad_norm_final)
    print("dist(x, x*)    :", dist_to_opt)
    print("=======================================")

    fs = np.array([h["f"] for h in history])
    grad_norms = np.array([h["grad_norm"] for h in history])
    iters = np.arange(len(history))
    gap = np.maximum(fs - f_opt, 1e-16)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(iters, fs, label="f(x_k)")
    plt.axhline(f_opt, color="red", linestyle="--", label="f*")
    plt.xlabel("iteration")
    plt.ylabel("f(x)")
    plt.title("Rayleigh quotient on S^{n-1} (Newton)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.semilogy(iters, gap, label="f(x_k) - f*")
    plt.semilogy(iters, grad_norms, label="||grad||")
    plt.xlabel("iteration")
    plt.title("Convergence (log scale)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("rayleigh_sphere_newton.png", dpi=150)


def run_comparison(
    dim: int = 10,
    seed: Optional[int] = 42,
):
    """
    在同一初始点上，对比 RGD 和 Riemannian Newton 的收敛速度。
    """
    problem = RayleighQuotientSphereProblem(dim=dim, seed=seed)
    manifold = problem.manifold
    x0 = problem.random_initial_point()
    f_opt = problem.optimal_value()

    # ---- RGD ----
    rgd = RiemannianGradientDescent(
        manifold=manifold, step_size=1e-2, max_iters=500, tol=1e-10
    )
    res_rgd = rgd.solve(x0, problem)

    # ---- Newton ----
    newton = RiemannianNewton(
        manifold=manifold, max_iters=50, tol=1e-10
    )
    res_newton = newton.solve(x0, problem)

    gap_rgd = np.maximum(
        [h["f"] for h in res_rgd["history"]], f_opt + 1e-16
    ) - f_opt
    gap_newton = np.maximum(
        [h["f"] for h in res_newton["history"]], f_opt + 1e-16
    ) - f_opt

    print("\n====== Comparison ======")
    print(f"RGD    : {res_rgd['num_iters']} iters, gap={res_rgd['f'] - f_opt:.2e}")
    print(f"Newton : {res_newton['num_iters']} iters, gap={res_newton['f'] - f_opt:.2e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(np.arange(len(gap_rgd)), gap_rgd, label="RGD")
    plt.semilogy(np.arange(len(gap_newton)), gap_newton, label="Newton")
    plt.xlabel("iteration")
    plt.ylabel("f(x_k) - f*")
    plt.title(f"RGD vs Riemannian Newton (dim={dim})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rayleigh_sphere_comparison.png", dpi=150)


if __name__ == "__main__":
    run_rayleigh_on_sphere(
        dim=10,
        step_size=1e-2,
        max_iters=1_000,
        tol=1e-6,
        seed=42,
    )

    run_rayleigh_on_sphere_newton(
        dim=10,
        max_iters=50,
        tol=1e-8,
        seed=42,
    )

    run_comparison(dim=10, seed=42)
