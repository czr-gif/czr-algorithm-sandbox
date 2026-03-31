#!/usr/bin/env python3
"""
讲解 GIF：Lie群、生成元、多模态、切丛
主题：Diffusion Model 赋能机器人制导
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager

# ─── Chinese Font ────────────────────────────────────────────
import glob as _glob
for _f in _glob.glob('/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc'):
    font_manager.fontManager.addfont(_f)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ─── Color Palette ──────────────────────────────────────────
BG     = '#0d1117'
PANEL  = '#161b22'
BORDER = '#30363d'
GRAY   = '#8b949e'
WHITE  = '#e6edf3'
CYAN   = '#58a6ff'
GREEN  = '#3fb950'
ORANGE = '#ff7b72'
PURPLE = '#bc8cff'
YELLOW = '#e3b341'
RED    = '#ff4444'

TOTAL = 240
FPS   = 12
DPI   = 110

# ─── Figure & Axes ──────────────────────────────────────────
fig = plt.figure(figsize=(13, 9), facecolor=BG)
fig.text(0.5, 0.978,
         'Diffusion Model 赋能机器人制导 | Lie群、生成元、多模态、切丛',
         ha='center', va='top', fontsize=12, color=WHITE, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.40, wspace=0.26,
                       left=0.05, right=0.97,
                       top=0.93, bottom=0.04)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

for ax, title, col in [
    (ax1, '① Lie群 与 生成元',        CYAN),
    (ax2, '② 制导向量场 (GVF)',        GREEN),
    (ax3, '③ 多模态：分叉路径的困境',  ORANGE),
    (ax4, '④ Diffusion：概率切流形',   PURPLE),
]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=GRAY, labelsize=7)
    ax.set_title(title, color=col, fontsize=10, pad=5, fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# PANEL 1: SO(2) Lie Group — path as group orbit
# ═══════════════════════════════════════════════════════════════
ax1.set_xlim(-1.9, 1.9)
ax1.set_ylim(-1.9, 1.9)
ax1.set_aspect('equal')
ax1.tick_params(labelbottom=False, labelleft=False)

# Group manifold circle
theta_c = np.linspace(0, 2*np.pi, 300)
ax1.plot(np.cos(theta_c), np.sin(theta_c),
         color=CYAN, lw=2, alpha=0.30, zorder=1)
ax1.text(0, 1.65, r'$SO(2)$ Lie群', color=CYAN,
         ha='center', fontsize=9.5, alpha=0.85)

# Coordinate cross
ax1.axhline(0, color=BORDER, lw=0.8)
ax1.axvline(0, color=BORDER, lw=0.8)

# Identity point (1, 0)
ax1.plot(1.0, 0.0, 'o', color=YELLOW, ms=7, zorder=5)
ax1.text(1.08, -0.22, '单位元 e', color=YELLOW, fontsize=7.5)

# Generator at identity X = (0, 1)  (Lie algebra element)
arr_gen = FancyArrowPatch((1.0, 0.0), (1.0, 0.45),
                           arrowstyle='->', color=YELLOW, lw=2.2,
                           mutation_scale=13, zorder=5, alpha=0.9)
ax1.add_patch(arr_gen)
ax1.text(1.13, 0.24, 'X ∈ Lie代数\n(生成元)',
         color=YELLOW, fontsize=7.5, linespacing=1.4)

# Formula
ax1.text(-1.85, -1.65,
         r'$g(t)=\exp(tX)\cdot g_0$',
         color=WHITE, fontsize=9, alpha=0.9)
ax1.text(-1.85, -1.85,
         '路径 = 群轨迹 (group orbit)',
         color=GRAY, fontsize=7.5)

# Animated: orbit trail + moving point + tangent arrow
orbit_trail, = ax1.plot([], [], '-', color=GREEN, lw=2, alpha=0.4, zorder=2)
point_g,     = ax1.plot([], [], 'o', color=GREEN, ms=10, zorder=6)
arr_tan = FancyArrowPatch((1.0, 0.0), (1.0, 0.40),
                           arrowstyle='->', color=GREEN, lw=2.2,
                           mutation_scale=13, zorder=7)
ax1.add_patch(arr_tan)
tan_label = ax1.text(0, 0, '', color=GREEN, fontsize=7.5, zorder=8)

# ═══════════════════════════════════════════════════════════════
# PANEL 2: Guidance Vector Field
# ═══════════════════════════════════════════════════════════════
ax2.set_xlim(-2.7, 2.7)
ax2.set_ylim(-2.1, 2.1)
ax2.set_aspect('equal')
ax2.tick_params(labelbottom=False, labelleft=False)

A, B = 1.9, 1.1
t_ell = np.linspace(0, 2*np.pi, 400)
ell_x = A * np.cos(t_ell)
ell_y = B * np.sin(t_ell)
ax2.plot(ell_x, ell_y, color=GREEN, lw=2.5, zorder=4)
ax2.text(-2.55, 1.85, '目标路径 Γ', color=GREEN, fontsize=8.5)


def proj_ellipse(px, py, a=A, b=B, n=400):
    t = np.linspace(0, 2*np.pi, n)
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    idx = np.argmin((px - ex)**2 + (py - ey)**2)
    return ex[idx], ey[idx], t[idx]


# Build GVF quiver
gx_g = np.linspace(-2.5, 2.5, 14)
gy_g = np.linspace(-1.9, 1.9, 10)
GX2, GY2 = np.meshgrid(gx_g, gy_g)
U2, V2 = np.zeros_like(GX2), np.zeros_like(GY2)

for i in range(GX2.shape[0]):
    for j in range(GX2.shape[1]):
        px, py = GX2[i, j], GY2[i, j]
        nx, ny, nt = proj_ellipse(px, py)
        tx = -A * np.sin(nt); ty = B * np.cos(nt)
        tn = np.hypot(tx, ty) + 1e-9
        tx /= tn; ty /= tn
        gx_v = nx - px; gy_v = ny - py
        gn = np.hypot(gx_v, gy_v) + 1e-9
        alpha = min(gn * 1.8, 1.8)
        U2[i, j] = tx + alpha * gx_v / gn
        V2[i, j] = ty + alpha * gy_v / gn

mag2 = np.hypot(U2, V2) + 1e-9
ax2.quiver(GX2, GY2, U2 / mag2, V2 / mag2,
           color=GREEN, alpha=0.18, scale=22, width=0.004, zorder=2)

# Tangent arrows along path
for t_v in np.linspace(0, 2*np.pi, 10, endpoint=False):
    px = A * np.cos(t_v); py = B * np.sin(t_v)
    tx = -A * np.sin(t_v); ty = B * np.cos(t_v)
    tn = np.hypot(tx, ty)
    ax2.annotate('', xy=(px + 0.22*tx/tn, py + 0.22*ty/tn), xytext=(px, py),
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5), zorder=5)

ax2.text(-2.55, -1.75, r'$\xi = \tau - k\nabla d$',
         color=GREEN, fontsize=9)
ax2.text(-2.55, -1.95, '切向 + 梯度 → 制导场',
         color=WHITE, fontsize=8)

# Animated robot
robot2,  = ax2.plot([], [], '^', color=ORANGE, ms=12, zorder=8)
trail2,  = ax2.plot([], [], '-', color=ORANGE, lw=1.8, alpha=0.6, zorder=7)


def integrate_gvf(x0, y0, steps, dt=0.07):
    xs, ys = [x0], [y0]
    for _ in range(steps - 1):
        x, y = xs[-1], ys[-1]
        nx, ny, nt = proj_ellipse(x, y)
        tx = -A * np.sin(nt); ty = B * np.cos(nt)
        tn = np.hypot(tx, ty) + 1e-9
        tx /= tn; ty /= tn
        gx_v = nx - x; gy_v = ny - y
        gn = np.hypot(gx_v, gy_v) + 1e-9
        alpha = min(gn * 2.2, 2.2)
        vx = tx + alpha * gx_v / gn
        vy = ty + alpha * gy_v / gn
        vn = np.hypot(vx, vy) + 1e-9
        xs.append(x + dt * vx / vn)
        ys.append(y + dt * vy / vn)
    return np.array(xs), np.array(ys)


r2x, r2y = integrate_gvf(-2.2, 1.6, TOTAL)

# ═══════════════════════════════════════════════════════════════
# PANEL 3: Multi-modal Problem (Y-junction failure)
# ═══════════════════════════════════════════════════════════════
ax3.set_xlim(-2.6, 2.6)
ax3.set_ylim(-2.4, 2.4)
ax3.set_aspect('equal')
ax3.tick_params(labelbottom=False, labelleft=False)

t_br = np.linspace(0, 1, 60)
ax3.plot(np.zeros(60), -2.0 + 2.0 * t_br, '-', color=ORANGE, lw=3, zorder=3)
ax3.plot(-2.2 * t_br,  2.2 * t_br,         '-', color=ORANGE, lw=3, zorder=3)
ax3.plot( 2.2 * t_br,  2.2 * t_br,         '-', color=ORANGE, lw=3, zorder=3)

# Arrows on stem
for y_p in [-1.7, -1.2, -0.65]:
    ax3.annotate('', xy=(0.0, y_p + 0.38), xytext=(0.0, y_p),
                 arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.8), zorder=4)

# Junction marker
ax3.plot(0, 0, 'o', color=RED, ms=13, zorder=9,
         markeredgecolor='white', markeredgewidth=1.5)
ax3.text(-0.35, 0.25, '?', color=RED, fontsize=22, fontweight='bold', zorder=10)

# Contradictory arrows at junction
ax3.annotate('', xy=(-0.75, 0.75), xytext=(-0.05, 0.05),
             arrowprops=dict(arrowstyle='->', color=RED, lw=2.5), zorder=8)
ax3.annotate('', xy=( 0.75, 0.75), xytext=( 0.05, 0.05),
             arrowprops=dict(arrowstyle='->', color=RED, lw=2.5), zorder=8)

# Annotations
ax3.text(-0.8,  0.75, '向左?', color=RED, fontsize=8, ha='center')
ax3.text( 0.8,  0.75, '向右?', color=RED, fontsize=8, ha='center')
ax3.text(-2.5, -2.0,  '生成元在分叉点不连续',       color=RED,  fontsize=8)
ax3.text(-2.5, -2.2,  '传统 GVF 无法定义唯一切向',  color=GRAY, fontsize=8)

# Animated confused robot
robot3, = ax3.plot([], [], '^', color=WHITE, ms=12, zorder=8)
trail3, = ax3.plot([], [], '-', color=WHITE, lw=1.8, alpha=0.5, zorder=7)


def robot3_pos(f):
    f_arrive = int(TOTAL * 0.40)
    if f <= f_arrive:
        t = f / f_arrive
        y = -2.0 + 2.0 * (3*t**2 - 2*t**3)   # smooth step
        return 0.0, y
    else:
        t = (f - f_arrive) / (TOTAL - f_arrive)
        phase = t * 5 * np.pi
        return 0.32 * np.sin(phase), 0.08 * np.cos(phase * 1.4)


r3x = np.array([robot3_pos(f)[0] for f in range(TOTAL)])
r3y = np.array([robot3_pos(f)[1] for f in range(TOTAL)])

# ═══════════════════════════════════════════════════════════════
# PANEL 4: Diffusion Model — Probability Flow
# ═══════════════════════════════════════════════════════════════
ax4.set_xlim(-2.6, 2.6)
ax4.set_ylim(-2.4, 2.4)
ax4.set_aspect('equal')
ax4.tick_params(labelbottom=False, labelleft=False)

ax4.plot(np.zeros(60), -2.0 + 2.0 * t_br, '-', color=PURPLE, lw=2.5, alpha=0.55, zorder=3)
ax4.plot(-2.2 * t_br,  2.2 * t_br,         '-', color=PURPLE, lw=2.5, alpha=0.55, zorder=3)
ax4.plot( 2.2 * t_br,  2.2 * t_br,         '-', color=PURPLE, lw=2.5, alpha=0.55, zorder=3)

# Local upward arrows along stem
for y_p in [-1.7, -1.2, -0.65]:
    ax4.annotate('', xy=(0.0, y_p + 0.38), xytext=(0.0, y_p),
                 arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.8), zorder=4)

# Probability distribution at junction (two branches, equal weight)
ax4.annotate('', xy=(-0.9, 0.9), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=CYAN, lw=3.5, alpha=0.9), zorder=8)
ax4.annotate('', xy=( 0.9, 0.9), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=3.5, alpha=0.9), zorder=8)
ax4.text(-1.35, 1.08, 'p = 0.5', color=CYAN,  fontsize=8.5, ha='center')
ax4.text( 1.35, 1.08, 'p = 0.5', color=GREEN, fontsize=8.5, ha='center')

# Junction glow dot
ax4.plot(0, 0, 'o', color=PURPLE, ms=14, zorder=9, alpha=0.6)
ax4.plot(0, 0, 'o', color=WHITE,  ms=6,  zorder=10)

ax4.text(-2.5, -2.0, 'p(运动方向 | 当前状态) 多模态分布', color=PURPLE, fontsize=7.8)
ax4.text(-2.5, -2.2, 'Diffusion 无需强结构先验，直接学习',  color=GRAY,   fontsize=7.8)

# Two animated trajectory robots
def make_traj(branch, delay=0):
    positions = np.zeros((TOTAL, 2))
    for f in range(TOTAL):
        if f < delay:
            positions[f] = [0.0, -2.0]
            continue
        f_adj = f - delay
        total_adj = TOTAL - delay
        t = f_adj / total_adj
        stem_t = 0.46
        if t <= stem_t:
            frac = t / stem_t
            y = -2.0 + 2.0 * frac
            positions[f] = [0.0, y]
        else:
            frac = min((t - stem_t) / (1 - stem_t), 1.0)
            if branch == 0:
                positions[f] = [-2.2 * frac, 2.2 * frac]
            else:
                positions[f] = [ 2.2 * frac, 2.2 * frac]
    return positions


r4a = make_traj(branch=0, delay=0)
r4b = make_traj(branch=1, delay=22)

robot4a, = ax4.plot([], [], '^', color=CYAN,  ms=12, zorder=8)
robot4b, = ax4.plot([], [], 's', color=GREEN, ms=11, zorder=8)
trail4a, = ax4.plot([], [], '-', color=CYAN,  lw=2,  alpha=0.6, zorder=7)
trail4b, = ax4.plot([], [], '-', color=GREEN, lw=2,  alpha=0.6, zorder=7)

# ═══════════════════════════════════════════════════════════════
# Legend / concept callouts
# ═══════════════════════════════════════════════════════════════
ax4.text( 1.5,  1.9, '← 采样1', color=CYAN,  fontsize=8)
ax4.text( 1.5,  1.65, '← 采样2', color=GREEN, fontsize=8)

ax3.text( 1.6,  1.9, '机器人', color=WHITE, fontsize=8)
ax3.text( 1.6,  1.65, '振荡/卡死', color=RED, fontsize=8)

# ═══════════════════════════════════════════════════════════════
# Animation
# ═══════════════════════════════════════════════════════════════
TRAIL = 55


def animate(frame):
    t = frame / TOTAL
    theta = 2 * np.pi * t

    # ── Panel 1: SO(2) ─────────────────────────────────────
    gx = np.cos(theta)
    gy = np.sin(theta)
    point_g.set_data([gx], [gy])

    # Growing trail arc
    n_trail = max(2, frame + 1)
    trail_t = np.linspace(0, theta, n_trail)
    orbit_trail.set_data(np.cos(trail_t), np.sin(trail_t))

    # Tangent at g(t): derivative = (-sin θ, cos θ)
    tx = -np.sin(theta)
    ty =  np.cos(theta)
    scale = 0.40
    arr_tan.set_positions((gx, gy), (gx + scale*tx, gy + scale*ty))

    tan_label.set_position((gx + scale*tx + 0.07, gy + scale*ty + 0.06))
    tan_label.set_text('切向量\ndg/dt')

    # ── Panel 2: GVF robot ─────────────────────────────────
    idx2 = min(frame, len(r2x) - 1)
    robot2.set_data([r2x[idx2]], [r2y[idx2]])
    ts2 = max(0, idx2 - TRAIL)
    trail2.set_data(r2x[ts2:idx2+1], r2y[ts2:idx2+1])

    # ── Panel 3: Confused robot ─────────────────────────────
    robot3.set_data([r3x[frame]], [r3y[frame]])
    ts3 = max(0, frame - TRAIL)
    trail3.set_data(r3x[ts3:frame+1], r3y[ts3:frame+1])

    # ── Panel 4: Two diverging trajectories ─────────────────
    robot4a.set_data([r4a[frame, 0]], [r4a[frame, 1]])
    robot4b.set_data([r4b[frame, 0]], [r4b[frame, 1]])
    ts4 = max(0, frame - TRAIL)
    trail4a.set_data(r4a[ts4:frame+1, 0], r4a[ts4:frame+1, 1])
    trail4b.set_data(r4b[ts4:frame+1, 0], r4b[ts4:frame+1, 1])


anim = animation.FuncAnimation(
    fig, animate, frames=TOTAL,
    interval=1000 // FPS, blit=False)

output = '3min/lie_group_diffusion.gif'
print(f"Rendering {TOTAL} frames → {output} ...")
anim.save(output, writer='pillow', fps=FPS, dpi=DPI,
          savefig_kwargs={'facecolor': BG})
print(f"Saved: {output}")
plt.close()
