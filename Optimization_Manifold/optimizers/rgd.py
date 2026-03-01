from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol, Any

import numpy as np

from Optimization_Manifold.manifolds.manifold import Manifold


class RiemannianObjective(Protocol):
    """
    最小化问题接口约定：
    - f(x): 返回标量损失
    - rgrad(x): 返回在 x 处的黎曼梯度 (切向量)
    """

    def f(self, x: np.ndarray) -> float: ...

    def rgrad(self, x: np.ndarray) -> np.ndarray: ...


class RiemannianGradientDescent:
    """
    最基本的黎曼梯度下降 (RGD)：
        x_{k+1} = Retr_x(-η_k * grad f(x_k))

    只依赖三个几何算子：
    - egrad2rgrad / rgrad: 计算黎曼梯度
    - norm: 评估梯度大小
    - retract: 收回映射，保证迭代点仍在流形上
    """

    def __init__(
        self,
        manifold: Manifold,
        step_size: float = 1e-2,
        max_iters: int = 1_000,
        tol: float = 1e-6,
        callback: Optional[
            Callable[[int, np.ndarray, float, float, Dict[str, Any]], None]
        ] = None,
    ) -> None:
        self.manifold = manifold
        self.step_size = step_size
        self.max_iters = max_iters
        self.tol = tol
        self.callback = callback

    def solve(
        self,
        x0: np.ndarray,
        objective: RiemannianObjective,
    ) -> Dict[str, Any]:
        """
        在给定流形上，对指定目标函数执行黎曼梯度下降。

        参数
        ----
        x0 : np.ndarray
            初始点，需已在流形上。
        objective : RiemannianObjective
            满足接口的目标问题对象，至少实现 f(x) 与 rgrad(x)。

        返回
        ----
        result : dict
            - "x": 最终点
            - "f": 最终函数值
            - "grad_norm": 最后一次迭代的梯度范数
            - "history": 每步 (f, grad_norm) 记录
            - "num_iters": 实际迭代步数
            - "converged": 是否因为梯度范数 < tol 提前收敛
        """
        x = np.array(x0, dtype=float, copy=True)
        history: List[Dict[str, float]] = []
        converged = False

        for k in range(self.max_iters):
            fx = objective.f(x)
            grad = objective.rgrad(x)
            grad_norm = self.manifold.norm(x, grad)

            history.append({"f": float(fx), "grad_norm": float(grad_norm)})

            if self.callback is not None:
                self.callback(k, x, fx, grad_norm, {"history": history})

            if grad_norm < self.tol:
                converged = True
                break

            # 梯度下降一步，并通过收回映射保证落在流形上
            step = -self.step_size * grad
            x = self.manifold.retract(x, step)

        return {
            "x": x,
            "f": float(objective.f(x)),
            "grad_norm": float(self.manifold.norm(x, objective.rgrad(x))),
            "history": history,
            "num_iters": k + 1,
            "converged": converged,
        }

