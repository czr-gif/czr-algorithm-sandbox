import numpy as np
from typing import Optional

from Optimization_Manifold.manifolds.manifold import Sphere


class RayleighQuotientSphereProblem:
    """
    球面上瑞利商最小化问题:
        min_{x in S^{n-1}} f(x) = x^T A x
    其中 A 为对称正定矩阵。
    """

    def __init__(self, dim: int, seed: Optional[int] = None):
        """
        参数
        ----
        dim : int
            向量维度 n，对应流形 S^{n-1}。
        seed : int | None
            随机种子，方便复现实验。
        """
        self.dim = dim
        self.manifold = Sphere(n=dim)  # Sphere(n) 表示嵌入维度为 n，流形为 S^{n-1}

        rng = np.random.default_rng(seed)

        # 构造一个对称正定矩阵 A = B^T B，保证有良好的谱性质
        B = rng.standard_normal((dim, dim))
        self.A = B.T @ B

        # 预先计算最小特征值及对应特征向量，作为“真解”基准
        eigvals, eigvecs = np.linalg.eigh(self.A)
        self.lambda_min = eigvals[0]
        self.x_star = eigvecs[:, 0]  # 最小特征值对应特征向量

    # ===========================
    # 目标函数与梯度
    # ===========================
    def f(self, x: np.ndarray) -> float:
        """目标函数 f(x) = x^T A x，要求 x 在单位球面上。"""
        return float(x.T @ self.A @ x)

    def egrad(self, x: np.ndarray) -> np.ndarray:
        """欧氏梯度 ∇f(x) = (A + A^T) x = 2 A x（A 已对称）。"""
        return 2.0 * (self.A @ x)

    def rgrad(self, x: np.ndarray) -> np.ndarray:
        """黎曼梯度：将欧氏梯度投影到球面切空间。"""
        return self.manifold.egrad2rgrad(x, self.egrad(x))

    # ===========================
    # 初始化与评估
    # ===========================
    def random_initial_point(self) -> np.ndarray:
        """在球面上随机初始化一个点。"""
        return self.manifold.random()

    def optimal_value(self) -> float:
        """返回理论最优目标值 λ_min。"""
        return float(self.lambda_min)

    def optimal_point(self) -> np.ndarray:
        """返回理论最优点（最小特征值对应的单位特征向量）。"""
        return self.x_star

    def distance_to_opt(self, x: np.ndarray) -> float:
        """
        用夹角的正弦来度量当前点与最优特征向量的偏差：
            d(x, x*) = sqrt(1 - (x^T x*)^2)
        该量在 0（方向相同或相反）到 1 之间。
        """
        cos_theta = float(x.T @ self.x_star)
        cos_theta = max(min(cos_theta, 1.0), -1.0)
        return np.sqrt(1.0 - cos_theta**2)

