from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Manifold(ABC):
    """
    抽象流形基类，定义黎曼优化所需的几何算子。
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """流形的内在维度。"""
        ...

    @property
    @abstractmethod
    def point_shape(self) -> tuple[int, ...] | None:
        """流形上点的形状，None 表示任意形状。"""
        ...

    @abstractmethod
    def inner(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        """
        计算切向量 u 和 v 在点 x 处的内积。
        """
        ...

    @abstractmethod
    def norm(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        计算切向量 u 在点 x 处的范数。
        """
        ...

    @abstractmethod
    def egrad2rgrad(self, x: np.ndarray, egrad: np.ndarray) -> np.ndarray:
        """
        将欧氏梯度转换为黎曼梯度。

        参数
        ----
        x : np.ndarray
            流形上的当前点。
        egrad : np.ndarray
            欧氏梯度。

        返回
        ----
        rgrad : np.ndarray
            黎曼梯度（位于 x 处的切空间）。
        """
        ...

    @abstractmethod
    def retract(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        收回映射：将切向量 u 从点 x 处的切空间映射回流形。

        参数
        ----
        x : np.ndarray
            流形上的当前点。
        u : np.ndarray
            切向量。

        返回
        ----
        x_new : np.ndarray
            流形上的新点，满足 Retr_x(u) ≈ x + u。
        """
        ...

    @abstractmethod
    def exp(self, x: np.ndarray, u: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        黎曼指数映射：沿切向量 u 方向在流形上前进距离 t。

        参数
        ----
        x : np.ndarray
            流形上的当前点。
        u : np.ndarray
            切向量。
        t : float
            前进步长。

        返回
        ----
        x_new : np.ndarray
            流形上的新点。
        """
        ...

    @abstractmethod
    def transport(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        并行传输：将点 x 处的切向量 u 平行传输到点 y 处的切空间。
        """
        ...

    @abstractmethod
    def random(self) -> np.ndarray:
        """
        在流形上生成一个随机点。
        """
        ...

    @abstractmethod
    def inner_vec(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        向量化版本的 inner，沿某个轴计算内积。
        """
        ...

    @abstractmethod
    def norm_vec(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        向量化版本的 norm，沿某个轴计算范数。
        """
        ...


class Sphere(Manifold):
    """
    单位球面流形：S^{n-1} = {x ∈ R^n | ||x||_2 = 1}

    由于约束 ||x||^2 = 1，在 x 处的切空间为：
        T_x S^{n-1} = {u ∈ R^n | <u, x> = 0}

    内积使用标准的欧氏内积。

    几何算子
    --------
    - 内积：标准欧氏内积
    - 黎曼梯度：P_x(egrad) = egrad - <egrad, x> * x (正交投影)
    - Retract：x + u / ||x + u||_2
    - 指数映射：cos(||u||_2) * x + sin(||u||_2) * u / ||u||_2
    - 平行传输：利用旋转公式
    """

    def __init__(self, n: int) -> None:
        """
        参数
        ----
        n : int
            球面的维度参数，流形实际为 S^{n-1}。
        """
        self._n = n

    @property
    def dim(self) -> int:
        return self._n - 1

    @property
    def point_shape(self) -> tuple[int, ...]:
        return (self._n,)

    def inner(self, x: np.ndarray, u: np.ndarray, v: np.ndarray, *, keepdim: bool = False) -> float:
        return float(np.sum(u * v, axis=-1, keepdims=keepdim))

    def norm(self, x: np.ndarray, u: np.ndarray, *, keepdim: bool = False) -> float:
        return float(np.linalg.norm(u, axis=-1, keepdims=keepdim))

    def egrad2rgrad(self, x: np.ndarray, egrad: np.ndarray) -> np.ndarray:
        """黎曼梯度 = 欧氏梯度 - <欧氏梯度, x> * x"""
        inner_prod = np.sum(egrad * x, axis=-1, keepdims=True)
        return egrad - inner_prod * x

    def retract(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Retract = (x + u) / ||x + u||"""
        new_x = x + u
        norm = np.linalg.norm(new_x, axis=-1, keepdims=True)
        return new_x / (norm + 1e-12)  # 避免除零

    def exp(self, x: np.ndarray, u: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        指数映射：cos(t*||u||) * x + sin(t*||u||) * u / ||u||
        """
        u_norm = np.linalg.norm(u, axis=-1, keepdims=True)
        u_norm = np.maximum(u_norm, 1e-12)

        # 避免零向量
        if np.all(u_norm < 1e-12):
            return x.copy()

        theta = t * u_norm
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * x + sin_theta * (u / u_norm)

    def transport(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        平行传输：使用旋转公式将 u 从 x 传输到 y
        """
        # 如果 x 和 y 在一条射线上，传输就是恒等变换
        if np.allclose(x, y) or np.allclose(x, -y):
            return u.copy()

        # 旋转轴（正交于 x 和 y 张成的平面）
        axis = np.cross(x, y)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            # x 和 y 平行或反平行
            if np.dot(x, y) > 0:
                return u.copy()
            else:
                # 反平行：沿法线方向翻转
                normal = self.normal(x)
                return u - 2 * np.dot(u, normal) * normal

        axis = axis / axis_norm
        axis = axis.reshape(-1, 1) if axis.ndim == 1 else axis

        #Rodrigues 旋转公式
        cos_angle = np.clip(np.dot(x.flatten(), y.flatten()), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # 一般情况下的平行传输（简化版）
        # 利用投影分解
        x_flat = x.flatten()
        y_flat = y.flatten()

        # 平行于 x 的分量（被移除）
        parallel_u = np.dot(u.flatten(), x_flat) * x_flat

        # 垂直于 x 的分量（需要旋转到 y 的垂直空间）
        perp_u = u.flatten() - parallel_u

        if np.linalg.norm(perp_u) < 1e-12:
            return u.copy()

        # 将 perp_u 投影到 y 的垂直空间
        perp_u_parallel_y = np.dot(perp_u, y_flat) * y_flat

        perp_u_ortho_y = perp_u - perp_u_parallel_y
        perp_u_ortho_y_norm = np.linalg.norm(perp_u_ortho_y)

        if perp_u_ortho_y_norm < 1e-12:
            # x 和 y 平行
            return u.copy()

        perp_u_ortho_y = perp_u_ortho_y / perp_u_ortho_y_norm

        # 在 y 的垂直空间中旋转
        result = (perp_u_parallel_y / np.linalg.norm(perp_u_parallel_y)) * np.dot(perp_u, perp_u_parallel_y) + \
                 (perp_u_ortho_y * np.cos(angle) + np.cross(perp_u_ortho_y, y_flat) * np.sin(angle)) * perp_u_ortho_y_norm

        return result - np.dot(result, y_flat) * y_flat

    def normal(self, vector: np.ndarray) -> np.ndarray:
        """单位球面上任意点的法向量就是该点本身（指向外侧）。"""
        return vector / np.linalg.norm(vector.flatten())

    def random(self) -> np.ndarray:
        """在单位球面上生成随机点（用高斯分布取归一化）。"""
        x = np.random.randn(self._n)
        return x / np.linalg.norm(x)

    def inner_vec(self, x: np.ndarray, u: np.ndarray, v: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.sum(u * v, axis=axis, keepdims=True)

    def norm_vec(self, x: np.ndarray, u: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.linalg.norm(u, axis=axis, keepdims=True)