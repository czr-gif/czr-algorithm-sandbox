from __future__ import annotations

from typing import Callable, Dict, List, Optional, Any

import numpy as np

from Optimization_Manifold.manifolds.manifold import Manifold


class RiemannianNewton:
    """
    球面上的黎曼牛顿法 (Riemannian Newton Method)。

    算法概述
    --------
    每步迭代：
      1. 计算黎曼梯度 grad = rgrad(x)
      2. 构造黎曼 Hessian 矩阵 H_R（作用在 T_x M 上）
      3. 求解牛顿方程：H_R[η] = -grad，η ∈ T_x M
      4. Armijo 回退线搜索确定步长 t
      5. 收回：x_{k+1} = Retr_x(t * η)

    黎曼 Hessian（球面公式）
    -----------------------
    对于嵌入 R^n 的球面 S^{n-1}：
        Hess_R f(x)[u] = P_x(∇²f(x)[u]) - <∇f(x), x> * u
    其中 P_x(v) = v - <v, x> * x 是到切空间的正交投影。

    求解策略
    --------
    由于 H_R 作为 n×n 矩阵是奇异的（x 在其零空间内），
    使用正则化：H_reg = H_R + x x^T，使方程组非奇异，
    而解 η 自动满足 <η, x> = 0（即 η ∈ T_x M）。

    Levenberg–Marquardt 正则化
    --------------------------
    若 H_R 在 T_x M 上不正定（如鞍点附近），自动添加
    μ * P_x 使其变为正定，保证下降方向。

    参数
    ----
    manifold    : Sphere 流形对象（需实现 ehess2rhess）
    max_iters   : 最大迭代步数
    tol         : 梯度范数收敛阈值
    armijo_c    : Armijo 条件参数（0 < c < 1）
    armijo_beta : 线搜索退缩因子（0 < β < 1）
    max_ls      : 线搜索最大次数
    lm_lambda0  : LM 初始正则化强度（仅在 Hessian 不正定时激活）
    callback    : 可选回调函数 callback(k, x, f, grad_norm, info)
    """

    def __init__(
        self,
        manifold: Manifold,
        max_iters: int = 200,
        tol: float = 1e-6,
        armijo_c: float = 1e-4,
        armijo_beta: float = 0.5,
        max_ls: int = 30,
        lm_lambda0: float = 1e-4,
        callback: Optional[
            Callable[[int, np.ndarray, float, float, Dict[str, Any]], None]
        ] = None,
    ) -> None:
        self.manifold = manifold
        self.max_iters = max_iters
        self.tol = tol
        self.armijo_c = armijo_c
        self.armijo_beta = armijo_beta
        self.max_ls = max_ls
        self.lm_lambda0 = lm_lambda0
        self.callback = callback

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        objective: Any,
    ) -> Dict[str, Any]:
        """
        在流形上执行黎曼牛顿迭代。

        参数
        ----
        x0        : 初始点（已在流形上）
        objective : 目标问题对象，至少实现：
                    - f(x) -> float
                    - egrad(x) -> ndarray
                    - rgrad(x) -> ndarray
                    - ehess_vec(x, u) -> ndarray  [可选，缺失则用有限差分]

        返回
        ----
        result : dict，含 x, f, grad_norm, history, num_iters, converged
        """
        x = np.array(x0, dtype=float, copy=True)
        history: List[Dict[str, float]] = []
        converged = False

        for k in range(self.max_iters):
            fx = float(objective.f(x))
            grad = objective.rgrad(x)
            grad_norm = float(self.manifold.norm(x, grad))

            history.append({"f": fx, "grad_norm": grad_norm})

            if self.callback is not None:
                self.callback(k, x, fx, grad_norm, {"history": history})

            if grad_norm < self.tol:
                converged = True
                break

            # 1. 构造黎曼 Hessian 矩阵并求解牛顿方向
            eta = self._newton_direction(x, objective, grad)

            # 2. Armijo 回退线搜索
            t = self._line_search(x, eta, fx, grad, objective)

            # 3. 收回映射
            x = self.manifold.retract(x, t * eta)

        return {
            "x": x,
            "f": float(objective.f(x)),
            "grad_norm": float(self.manifold.norm(x, objective.rgrad(x))),
            "history": history,
            "num_iters": k + 1,
            "converged": converged,
        }

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _newton_direction(
        self,
        x: np.ndarray,
        objective: Any,
        rgrad: np.ndarray,
    ) -> np.ndarray:
        """
        求解牛顿方程 H_R[η] = -rgrad，η ∈ T_x M。

        使用正则化矩阵 H_reg = H_R + x*x^T，保证非奇异；
        若在 T_x M 上不正定，额外加 LM 正则项 μ * P_x。
        """
        n = len(x)
        egrad = objective.egrad(x) if hasattr(objective, "egrad") else None

        # ---- 逐列构造 n×n 黎曼 Hessian 矩阵 ----
        H_R = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            H_R[:, i] = self._rhess_vec(x, e_i, objective, egrad)

        # 正则化：H_reg = H_R + x x^T
        # 使得 H_reg 非奇异，且解沿 x 方向分量为 0
        H_reg = H_R + np.outer(x, x)

        # ---- LM 正则化：确保在 T_x M 上正定 ----
        P_x = np.eye(n) - np.outer(x, x)
        H_on_tangent = P_x @ H_R @ P_x  # H_R 在 T_x M 上的限制（对称）
        # 取最小特征值作为正定性检验
        eigvals = np.linalg.eigvalsh(H_on_tangent)
        # 注：P_x 本身有 eigenvalue=0（沿 x 方向），排除它
        tangent_eigvals = eigvals[eigvals > -0.5 * np.max(np.abs(eigvals))]
        min_ev = float(np.min(tangent_eigvals)) if len(tangent_eigvals) > 0 else 0.0

        if min_ev < self.lm_lambda0:
            # 添加 Levenberg–Marquardt 正则项
            mu = self.lm_lambda0 - min_ev + self.lm_lambda0
            H_reg = H_reg + mu * P_x

        # ---- 求解线性方程组 ----
        try:
            eta = np.linalg.solve(H_reg, -rgrad)
        except np.linalg.LinAlgError:
            # 退回到梯度下降方向
            eta = -rgrad

        # 投影回切空间（消除数值误差累积）
        eta = eta - np.dot(eta, x) * x

        # 验证是否为下降方向，若否则退回梯度方向
        if np.dot(eta, rgrad) > 0:
            eta = -rgrad

        return eta

    def _rhess_vec(
        self,
        x: np.ndarray,
        u: np.ndarray,
        objective: Any,
        egrad: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        计算黎曼 Hessian-向量积 Hess_R f(x)[u]。

        优先使用 objective.ehess_vec + manifold.ehess2rhess；
        否则使用有限差分近似（rgrad 在切向量方向的方向导数）。
        """
        if hasattr(objective, "ehess_vec") and hasattr(self.manifold, "ehess2rhess"):
            eg = egrad if egrad is not None else objective.egrad(x)
            ehess_u = objective.ehess_vec(x, u)
            return self.manifold.ehess2rhess(x, eg, ehess_u, u)

        # 有限差分后备：沿测地线方向差分 rgrad
        h = 1e-7
        # 将 u 投影到切空间作为步进方向
        u_tang = u - np.dot(u, x) * x
        u_norm = np.linalg.norm(u_tang)
        if u_norm < 1e-14:
            return np.zeros_like(x)
        u_tang = u_tang / u_norm

        x_eps = self.manifold.exp(x, h * u_tang)
        g_eps = objective.rgrad(x_eps)
        # 平行传输回 x 处的切空间
        g_eps_t = self.manifold.transport(x_eps, x, g_eps)
        g = objective.rgrad(x)
        return (g_eps_t - g) / h * u_norm

    def _line_search(
        self,
        x: np.ndarray,
        eta: np.ndarray,
        fx: float,
        rgrad: np.ndarray,
        objective: Any,
    ) -> float:
        """
        Armijo 回退线搜索。

        接受条件：f(Retr_x(t * η)) ≤ f(x) + c * t * <grad, η>
        其中 <grad, η> 应为负值（下降方向）。
        """
        slope = self.manifold.inner(x, rgrad, eta)  # 应 < 0
        t = 1.0
        for _ in range(self.max_ls):
            x_new = self.manifold.retract(x, t * eta)
            if objective.f(x_new) <= fx + self.armijo_c * t * slope:
                return t
            t *= self.armijo_beta
        return t
