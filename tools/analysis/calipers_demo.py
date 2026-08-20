#!/usr/bin/env python3
"""Visual demo: rotating calipers algorithm with CORRECT bounding rectangles.

The key fix: rectangle corners are intersections of four extremal lines,
not just 'left_point + width*base + height*rot90'.

Also draws:
  - All four caliper sides (supporting lines through extremal points)
  - Full step-by-step enumeration with detailed logs
  - Each step as a separate subplot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull


def rotate90CW(v):
    return np.array([v[1], -v[0]])

def rotate90CCW(v):
    return np.array([-v[1], v[0]])

def first_vec_is_right(v1, v2):
    tmp = rotate90CW(v1)
    return tmp[0]*v2[0] + tmp[1]*v2[1] < 0


def compute_rect_corners(bottom_pt, right_pt, top_pt, left_pt, base_a, base_b):
    """Compute the four corners of the bounding rectangle.

    The four extremal lines:
      left   : base_a*x + base_b*y  = base_a*left_x  + base_b*left_y    = C_left
      right  : base_a*x + base_b*y  = base_a*right_x + base_b*right_y   = C_right
      bottom : -base_b*x + base_a*y = -base_b*bottom_x + base_a*bottom_y = C_bottom
      top    : -base_b*x + base_a*y = -base_b*top_x    + base_a*top_y    = C_top

    Corners are intersections of adjacent lines.
    """
    C_left   = base_a * left_pt[0]   + base_b * left_pt[1]
    C_right  = base_a * right_pt[0]  + base_b * right_pt[1]
    C_bottom = -base_b * bottom_pt[0] + base_a * bottom_pt[1]
    C_top    = -base_b * top_pt[0]    + base_a * top_pt[1]

    # intersection of a*x+b*y = C  and  -b*x+a*y = D  is:
    #   x = a*C - b*D,  y = b*C + a*D  (because (a,b) is unit)
    corners = np.array([
        [base_a*C_left - base_b*C_bottom, base_b*C_left + base_a*C_bottom],   # BL
        [base_a*C_right - base_b*C_bottom, base_b*C_right + base_a*C_bottom], # BR
        [base_a*C_right - base_b*C_top,    base_b*C_right + base_a*C_top],    # TR
        [base_a*C_left - base_b*C_top,     base_b*C_left + base_a*C_top],      # TL
    ])
    return corners, C_left, C_right, C_bottom, C_top


def draw_line(ax, pt, direction, color, label, ls='--', alpha=0.7, lw=1.2):
    """Draw a line through pt in direction 'direction'."""
    t = np.linspace(-20, 20, 100)
    x = pt[0] + direction[0] * t
    y = pt[1] + direction[1] * t
    ax.plot(x, y, color=color, linestyle=ls, alpha=alpha, linewidth=lw, label=label)


def min_area_rect_verbose(pts):
    """Full verbose rotating calipers — returns all steps with caliper data."""
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    n = len(hull_pts)
    edges = [(hull_pts[(i+1)%n] - hull_pts[i]) for i in range(n)]
    edge_len = [np.linalg.norm(e) for e in edges]
    unit_edges = [e / l for e, l in zip(edges, edge_len)]

    bottom = np.argmin(hull_pts[:, 1])
    right  = np.argmax(hull_pts[:, 0])
    top    = np.argmax(hull_pts[:, 1])
    left   = np.argmin(hull_pts[:, 0])
    seq = [bottom, right, top, left]

    base_a, base_b = 1.0, 0.0
    min_area = float('inf')
    best_step = -1
    steps = []

    for k in range(n):
        # Determine caliper side with minimum angle
        main_idx = 0
        rot = [
            unit_edges[seq[0]],
            rotate90CW(unit_edges[seq[1]]),
            -unit_edges[seq[2]],
            rotate90CCW(unit_edges[seq[3]]),
        ]
        for i in range(1, 4):
            if first_vec_is_right(rot[i], rot[main_idx]):
                main_idx = i

        pidx = seq[main_idx]
        lead = unit_edges[pidx]
        if main_idx == 0:
            base_a, base_b = lead[0], lead[1]
        elif main_idx == 1:
            base_a, base_b = lead[1], -lead[0]
        elif main_idx == 2:
            base_a, base_b = -lead[0], -lead[1]
        elif main_idx == 3:
            base_a, base_b = -lead[1], lead[0]

        seq[main_idx] = (seq[main_idx] + 1) % n

        # Rectangle dimensions from four extremal points
        b_pt = hull_pts[seq[0]]
        r_pt = hull_pts[seq[1]]
        t_pt = hull_pts[seq[2]]
        l_pt = hull_pts[seq[3]]

        dx_w = r_pt[0] - l_pt[0]
        dy_w = r_pt[1] - l_pt[1]
        width = abs(dx_w * base_a + dy_w * base_b)

        dx_h = t_pt[0] - b_pt[0]
        dy_h = t_pt[1] - b_pt[1]
        height = abs(-dx_h * base_b + dy_h * base_a)

        area = width * height

        corners, C_left, C_right, C_bottom, C_top = compute_rect_corners(
            b_pt, r_pt, t_pt, l_pt, base_a, base_b
        )

        steps.append({
            'step': k + 1,
            'edge_idx': pidx,
            'edge_start': hull_pts[pidx],
            'base': (base_a, base_b),
            'base_angle_deg': np.degrees(np.arctan2(base_b, base_a)),
            'width': width,
            'height': height,
            'area': area,
            'bottom_pt': b_pt, 'right_pt': r_pt, 'top_pt': t_pt, 'left_pt': l_pt,
            'corners': corners,
            'C_left': C_left, 'C_right': C_right,
            'C_bottom': C_bottom, 'C_top': C_top,
            'main_idx': main_idx,
            'seq': list(seq),
        })

        if area <= min_area:
            min_area = area
            best_step = k

    return steps, min_area, best_step


def visualize_steps(pts, steps, best_step, name):
    n = len(steps)
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    hull_poly = np.vstack([hull_pts, hull_pts[0]])

    # All steps on one figure
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)

    for si, s in enumerate(steps):
        r, c = divmod(si, cols)
        ax = axes[r][c]

        # Convex hull
        ax.plot(hull_poly[:, 0], hull_poly[:, 1], 'k-', linewidth=2)
        ax.scatter(pts[:, 0], pts[:, 1], c='black', s=30, zorder=5)
        ax.scatter(hull_pts[:, 0], hull_pts[:, 1], c='gray', s=15, zorder=4)

        # Caliper sides (4 supporting lines)
        b = s['base']
        perp = (-b[1], b[0])  # perpendicular to base

        # Bottom line: through bottom_pt, along base
        draw_line(ax, s['bottom_pt'], b, 'green', '', lw=1.0)
        # Right line: through right_pt, along perp
        draw_line(ax, s['right_pt'], perp, 'orange', '', lw=1.0)
        # Top line: through top_pt, along base
        draw_line(ax, s['top_pt'], b, 'red', '', lw=1.0)
        # Left line: through left_pt, along perp
        draw_line(ax, s['left_pt'], perp, 'purple', '', lw=1.0)

        # Mark extremal points
        ax.scatter([s['bottom_pt'][0]], [s['bottom_pt'][1]], c='green', s=80, zorder=6, marker='o', label='bottom')
        ax.scatter([s['right_pt'][0]],  [s['right_pt'][1]],  c='orange', s=80, zorder=6, marker='o', label='right')
        ax.scatter([s['top_pt'][0]],    [s['top_pt'][1]],    c='red',    s=80, zorder=6, marker='o', label='top')
        ax.scatter([s['left_pt'][0]],   [s['left_pt'][1]],   c='purple', s=80, zorder=6, marker='o', label='left')

        # Current aligned edge
        edge_pt = s['edge_start']
        base_vec = np.array(s['base'])
        ax.arrow(edge_pt[0], edge_pt[1], base_vec[0]*1.0, base_vec[1]*1.0,
                 head_width=0.15, head_length=0.2, fc='red', ec='red', linewidth=2.5, zorder=7)

        # Bounding rectangle (correct corners from line intersections)
        corners = s['corners']
        rect = np.vstack([corners, corners[0]])
        ax.plot(rect[:, 0], rect[:, 1], 'b--', linewidth=2, label='bounding rect')

        title = f"step {s['step']}/{n}"
        if si == best_step:
            title += " ← MIN"
        ax.set_title(f"{title}\nw={s['width']:.2f} h={s['height']:.2f} area={s['area']:.2f}", fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for si in range(n, rows * cols):
        r, c = divmod(si, cols)
        axes[r][c].axis('off')

    fig.suptitle(
        f"Rotating Calipers — {name}\n"
        f"Red arrow = aligned edge   Green/Orange/Red/Purple lines = caliper sides   Blue dashed = bounding rect",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = f"./test/reports/calipers_full_{name}.png"
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def log_steps(steps, min_area):
    print(f"\n{'step':>4} {'edge':>4} {'base_angle':>10} {'width':>8} {'height':>8} {'area':>8} {'main_idx':>8}")
    print("-" * 60)
    for s in steps:
        marker = " ← MIN" if s['area'] == min_area else ""
        print(f"{s['step']:>4} {s['edge_idx']:>4} {s['base_angle_deg']:>9.1f}° "
              f"{s['width']:>8.3f} {s['height']:>8.3f} {s['area']:>8.3f} {s['main_idx']:>8}{marker}")


# ── Run ──
def make_irregular():
    np.random.seed(3)
    return np.random.uniform(0, 6, (8, 2))

def make_diamond():
    return np.array([[0,0], [0,1], [1,2], [1,1]], dtype=float) * 3

def make_L_shape():
    return np.array([[0,0],[4,0],[4,1],[2,1],[2,4],[1,4],[1,1],[0,1]], dtype=float)

for name, pts in [
    ("irregular", make_irregular()),
    ("diamond", make_diamond()),
    ("L_shape", make_L_shape()),
]:
    print(f"\n{'='*60}")
    print(f"Polygon: {name}  ({len(pts)} points)")
    print(f"Convex hull: {len(ConvexHull(pts).vertices)} vertices")
    print(f"{'='*60}")
    steps, min_area, best_step = min_area_rect_verbose(pts)
    log_steps(steps, min_area)
    visualize_steps(pts, steps, best_step, name)
