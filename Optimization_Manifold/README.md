# 🚀 Riemannian Wheelchair: 流形优化算法沙盒

本模块是 `czr-algorithm-sandbox` 的核心组件之一，专注于**黎曼流形上的无约束优化问题**。

**核心哲学：拒绝内蕴几何受苦，拥抱嵌入坐标轮椅。**

本库的所有实现均基于**嵌入几何 (Embedded Geometry)** 视角。我们将流形视为高维欧氏空间的子集，抛弃复杂的局部坐标卡 (Charts) 和克里斯托弗符号 ($\Gamma_{ij}^k$)，全部通过**欧氏导数 (Euclidean Derivatives) + 正交投影 (Orthogonal Projections) + 收回映射 (Retractions)** 来完成流形上的微积分与优化。

---

## 🛠️ 1. 核心优化算法 (Optimizers)

本模块计划手搓实现以下 4 种经典的流形优化算法，覆盖从一阶到二阶的完整生态：

- [ ] **RGD (Riemannian Gradient Descent)**
  - **简介**：黎曼梯度下降。流形优化的 Baseline，最朴素的“算梯度 -> 投影 -> Retraction”三步走。
- [ ] **R-Momentum / RCG (Riemannian Conjugate Gradient)**
  - **简介**：引入流形上的动量机制与共轭梯度。核心挑战是实现 **Vector Transport (向量传输)**，将上一步的动量“平移”到当前切空间。
- [ ] **R-Newton (Riemannian Newton's Method with tCG)**
  - **简介**：黎曼牛顿法。真正的二阶杀器，使用截断共轭梯度法 (tCG) 在切空间内近似求解牛顿方程，**避免显式构造庞大的 Hessian 矩阵**，只需实现 Hessian-Vector Product (HVP)。
- [ ] **RTR (Riemannian Trust Region)**
  - **简介**：黎曼信赖域法。工业界首选，极其鲁棒的二阶方法。自带步长限制与非凸地形（鞍点）逃逸机制，保证全局收敛。

---

## 🌌 2. 代表性流形 (Manifolds)

为了验证算法的有效性，我们选取了 3 个在控制理论、机器人学和机器学习中最具代表性的流形：

- [ ] **Sphere Manifold ($S^{n-1}$)**
  - **约束**：$x^T x = 1$
  - **地位**：流形优化的 "Hello World"。几何结构极其简单，投影矩阵为 $I - xx^T$，极其适合用来 debug 算法基础逻辑。
- [ ] **Special Orthogonal Group ($SO(3)$)**
  - **约束**：$R^T R = I, \det(R) = 1$
  - **地位**：3D 旋转群。机器人姿态控制、刚体动力学和 SLAM 的绝对核心。它的切空间对应反对称矩阵 (Skew-symmetric matrices)，投影和 Retraction 涉及 SVD 分解或矩阵指数。
- [ ] **Stiefel Manifold ($St(p, n)$)**
  - **约束**：$X^T X = I_p$ (正交基矩阵)
  - **地位**：$SO(n)$ 的推广。在特征值计算、PCA 降维和深度学习正交权重初始化中极其常用。

---

## 🎯 3. 测试基准问题 (Benchmark Problems)

每个流形都需要一个经典的优化问题来验证算法是否能成功收敛到谷底。

- [ ] **球面测试：瑞利商最小化 (Rayleigh Quotient Minimization)**
  - **流形**：Sphere $S^{n-1}$
  - **目标函数**：$\min_{x \in S^{n-1}} f(x) = x^T A x$ (其中 $A$ 是对称正定矩阵)
  - **已知最优解**：矩阵 $A$ 最小特征值对应的特征向量。我们将观察 RGD 和 R-Newton 谁能更快收敛到这个特征向量。
- [ ] **$SO(3)$ 测试：Wahba 问题 (Wahba's Problem)**
  - **流形**：$SO(3)$
  - **目标函数**：已知两组 3D 点云 $u_i$ 和 $v_i$，求最优旋转矩阵 $R$ 使得 $\min_{R \in SO(3)} \sum \|v_i - R u_i\|^2$
  - **已知最优解**：可通过 SVD 闭式求解。用来测试流形优化算法在姿态配准中的数值表现。
- [ ] **Stiefel 测试：正交普鲁克问题 (Orthogonal Procrustes)**
  - **流形**：Stiefel $St(p, n)$
  - **目标函数**：$\min_{X \in St} \|AX - B\|_F^2$
  - **物理意义**：寻找一个正交投影矩阵，使得高维数据 $A$ 尽可能贴合目标 $B$。

---

## 📂 4. 目录结构 (Directory Structure)

```text
riemannian-wheelchair/
├── optimizers/           # 优化算法实现
│   ├── rgd.py            # 黎曼梯度下降
│   ├── r_momentum.py     # 黎曼动量法
│   ├── r_newton.py       # 黎曼牛顿法 (含 tCG 子程序)
│   └── rtr.py            # 黎曼信赖域法
├── manifolds/            # 几何结构与算子
│   ├── base.py           # 流形基类 (定义 proj, retract, egrad2rgrad 等接口)
│   ├── sphere.py         # 球面实现
│   ├── so3.py            # 旋转群实现
│   └── stiefel.py        # 斯特弗尔流形实现
├── benchmarks/           # 测试案例脚本
│   ├── test_rayleigh.py  # 球面瑞利商测试
│   └── test_wahba.py     # SO3 姿态配准测试
└── utils/                # 辅助工具 (如向量转矩阵、HVP自动微分包裹器等)