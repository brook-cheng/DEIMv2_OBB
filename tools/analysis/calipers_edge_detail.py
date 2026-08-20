#!/usr/bin/env python3
"""Detailed demo: edge ordering, arrow start point, and base direction.

Answers:
  Q1: hull vertices are ALREADY ORDERED by scipy ConvexHull (CCW).
      We index them as 0,1,2,...N-1. The calipers traverse edges
      using their vertex index — edge_i is from vertex[i] to vertex[(i+1)%N].
      The caliper "currently aligned edge" is the one whose index equals
      pidx = seq[main_idx].
  Q2: The red arrow starts at the START vertex of the aligned edge.
      For edge i = pidx, the start is hull_pts[pidx], the end is
      hull_pts[(pidx+1)%N]. The arrow points from the start toward the end.
  Q3: The direction (base_a, base_b) is the UNIT vector of the edge.
      If the aligned edge is from hull_pts[p] to hull_pts[(p+1)%n],
      then (base_a, base_b) = (dx, dy) / |edge| where dx=x_next-x_curr,
      dy=y_next-y_curr. BUT — depending on which caliper side (main_idx
      = 0,1,2,3) triggered the rotation, the base may be rotated by
      90°, 180°, or 270° to match the standard caliper orientation.
      main_idx=0: base = edge_direction (as-is)
      main_idx=1: base = edge rotated +90° (CW)
      main_idx=2: base = edge rotated +180° (flipped)
      main_idx=3: base = edge rotated -90° (CCW)
      This ensures the base always points roughly "rightward" within
      the caliper's coordinate system.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

np.random.seed(3)
pts = np.random.uniform(0, 6, (8, 2))

hull = ConvexHull(pts)
hull_pts = pts[hull.vertices]  # ALREADY CCW from scipy
n = len(hull_pts)

# ── Q1: edge ordering ──
edges_info = []
for i in range(n):
    v0 = hull_pts[i]
    v1 = hull_pts[(i+1)%n]
    dx = v1[0] - v0[0]
    dy = v1[1] - v0[1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))
    edges_info.append({
        'idx': i,
        'start': v0,
        'end': v1,
        'dx': dx,
        'dy': dy,
        'length': length,
        'angle_deg': angle,
    })

print("Q1: Convex hull edge ordering (CCW from scipy ConvexHull.vertices)")
print(f"Hull has {n} vertices, {n} edges")
print(f"{'edge':>4} {'start(x,y)':>18} {'end(x,y)':>18} {'len':>8} {'angle':>8}")
print("-" * 70)
for e in edges_info:
    print(f"{e['idx']:>4} ({e['start'][0]:.3f},{e['start'][1]:.3f})  "
          f"({e['end'][0]:.3f},{e['end'][1]:.3f})  "
          f"{e['length']:>8.3f}  {e['angle_deg']:>7.1f}°")

# ── Q2 & Q3: red arrow start + direction ──
print(f"\nQ2&3: Red arrow start = start vertex of aligned edge.")
print(f"      Arrow direction = unit vector of that edge (possibly rotated 90/180/270).")
print(f"      main_idx=0 → as-is; main_idx=1 → CW90; main_idx=2 → flip180; main_idx=3 → CCW90")

# Draw edge ordering
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: edge indices
ax = axes[0]
for i in range(n):
    v0 = hull_pts[i]
    v1 = hull_pts[(i+1)%n]
    ax.annotate('', xy=v1, xytext=v0, arrowprops=dict(arrowstyle='->', color='red', lw=2))
    mid = (v0 + v1) / 2
    ax.text(mid[0], mid[1], f"e{i}", fontsize=11, color='darkred',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow', alpha=0.8))
    ax.plot([v0[0]], [v1[0]], [v0[1]], [v1[1]], 'ro', markersize=5)
    ax.text(v0[0]-0.15, v0[1]-0.15, f"v{i}", fontsize=9, color='gray')

for i, pt in enumerate(pts):
    ax.scatter(pt[0], pt[1], c='black', s=30, zorder=5)
ax.set_title("Q1: Edge ordering (CCW)\ne0→e1→e2→e3 traversed in order")
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Right: focus on one edge — show how arrow start + direction connect
ax = axes[1]
# pick edge 0 as example
e0 = edges_info[0]
for i, pt in enumerate(pts):
    ax.scatter(pt[0], pt[1], c='black', s=30, zorder=5)

hull_poly = np.vstack([hull_pts, hull_pts[0]])
ax.plot(hull_poly[:,0], hull_poly[:,1], 'k-', linewidth=2)

# Draw edge
v_start = e0['start']
v_end = e0['end']
ax.annotate('', xy=v_end, xytext=v_start,
            arrowprops=dict(arrowstyle='->', color='red', lw=4))
ax.plot(v_start[0], v_end[0], v_start[1], v_end[1], 'ro', markersize=10)

# Label
mid = (v_start + v_end) / 2
ax.text(mid[0], mid[1], f"edge {e0['idx']}\ndirection={e0['angle_deg']:.1f}°",
        fontsize=11, color='darkred', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))

ax.text(v_start[0]-0.3, v_start[1]-0.3, "start (red arrow tail)", fontsize=9, color='red')
ax.text(v_end[0]+0.2, v_end[1]+0.2, "end", fontsize=9, color='gray')

# Show the main_idx rotation table
table_text = (
    "main_idx rotation:\n"
    "0 → base=dir (as-is)\n"
    f"   dir=({e0['dx']/e0['length']:.2f},{e0['dy']/e0['length']:.2f})\n"
    "1 → base=CW90(dir)\n"
    f"   =({e0['dy']/e0['length']:.2f},{-(e0['dx']/e0['length']):.2f})\n"
    "2 → base=flip180(dir)\n"
    "3 → base=CCW90(dir)"
)
ax.text(5.5, 5.5, table_text, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
        va='top')

ax.set_title("Q2&3: Arrow start = edge start vertex\nDirection = unit vector of edge (or rotated)")
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "./test/reports/calipers_edge_ordering.png"
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

# ── Also show full caliper traversal with base direction debug ──
print(f"\nQ3 supplement: show base_a,base_b for each step (from actual calipers run)")
print(f"  Note main_idx SWITCHES which edge gets loaded into base.")

# Quick re-run of calipers with direction annotation
import sys; sys.path.insert(0, '.')
from test.tool_calipers_demo import min_area_rect_verbose
steps, _, _ = min_area_rect_verbose(pts)
print(f"\n{'step':>4} {'edge_idx':>8} {'base_angle':>10} {'main_idx':>8} {'base_vec':>22} {'source desc':>30}")
print("-" * 90)
for s in steps:
    b = s['base']
    midx = s['main_idx']
    names = ['edge_dir(as-is)', 'CW90(edge)', 'flip180(edge)', 'CCW90(edge)']
    print(f"{s['step']:>4} {s['edge_idx']:>8} {s['base_angle_deg']:>9.1f}°  "
          f"{midx:>8}  ({b[0]:+.3f}, {b[1]:+.3f})          "
          f"{names[midx]}")
