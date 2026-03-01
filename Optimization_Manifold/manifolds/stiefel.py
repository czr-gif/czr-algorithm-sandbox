"""
Stiefel Manifold V_{k,n} = {X ∈ R^{n×k} | X^T X = I_k}

Author: czr-algorithm-sandbox
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .manifold import Manifold


class Stiefel(Manifold):
    """
    Stiefel 流形 V_{k,n}：所有 n×k 列正交矩阵的集合。

    约束条件：X^T X = I_k（k 个标准正交列向量）

    几何算子说明
    -----------------
    - 切空间：T_X V_{k,n} = {U ∈ R^{n×k} | X^T U + U^T X = 0}
    - 黎曼梯度：通过正交投影将欧氏梯度投影到切空间

    采用 retraction 的两种常用方式：
    1. QR-based: [X + U] = qr_flip(X + U)
    2. Polar: via svd on (X + U) = U Σ V^T
    """

    def __init__(self, n: int, k: Optional[int] = None) -> None:
        """
        参数
        ----
        n : int
            矩阵行数
        k : int | None
            矩阵列数。如果为 None，则 k=n（方阵情况，退化为正交群）
        """
        self._n = n
        self._k = k if k is not None else n

    @property
    def dim(self) -> int:
        """Stiefel 流形的内在维度: n*k - k*(k+1)/2"""
        return self._n * self._k - self._k * (self._k + 1) // 2

    @property
    def point_shape(self) -> tuple[int, int]:
        return (self._n, self._k)

    def inner(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.sum(u * v))

    def norm(self, x: np.ndarray, u: np.ndarray) -> float:
        return float(np.linalg.norm(u, ord="fro"))

    def egrad2rgrad(self, x: np.ndarray, egrad: np.ndarray) -> np.ndarray:
        """
        将欧氏梯度投影到切空间。

        公式：P_X(egrad) = egrad - X * Sym(X^T egrad)

        其中 Sym(Y) = (Y + Y^T) / 2

        推导：
        - 切空间约束：X^T U + U^T X = 0
        - 分解 X^T egrad = Sym + Skew
        - 要满足约束，需减去 Sym 部分
        """
        # 计算 X^T egrad
        xtg = x.T @ egrad  # k×k 矩阵

        # 计算对称部分: (xtg + xtg^T) / 2
        sym_xtg = (xtg + xtg.T) / 2.0

        # 黎曼梯度：egrad - X * sym_xtg
        rgrad = egrad - x @ sym_xtg

        return rgrad

    def retract(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        QR-based retraction: retr_X(u) = normalize_columns(X + u)

        使用修正的 QR 分解来保持数值稳定性。
        """
        y = x + u

        # QR 分解
        Q, R = np.linalg.qr(y)

        # 修正 QR（处理列翻转）
        # 当 R 的对角线为负时，翻转对应列的符号
        d = np.diag(np.sign(np.diag(R) + 1e-12))
        Q = Q @ d
        R = d @ R

        return Q

    def exp(self, x: np.ndarray, u: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        使用指数映射：利用 Cayley 变换。

        exp_X(t*u) = X + ∫... 的一种近似实现。

        简化方案：递归 QR 分解
        """
        y = x + t * u
        return self.retract(x, t * u)

    def transport(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        平行传输：利用 QR 分解

        P_{x→y}(u) = retr_y(u_proj)

        其中 u_proj 是 u 在 y 切空间的投影。
        """
        # 这是一个简化实现
        return u

    def random(self) -> np.ndarray:
        """
        在 Stiefel 流形上生成随机点：

        1. 生成 n×k 随机矩阵
        2. QR 分解得到正交化结果
        """
        A = np.random.randn(self._n, self._k)
        Q, _ = np.linalg.qr(A)
        return Q

    def inner_vec(self, x: np.ndarray, u: np.ndarray, v: np.ndarray, axis=None) -> np.ndarray:
        return np.sum(u * v, axis=axis, keepdims=True)

    def norm_vec(self, x: np.ndarray, u: np.ndarray, axis=None) -> np.ndarray:
        return np.linalg.norm(u, ord="fro", axis=axis, keepdims=True)