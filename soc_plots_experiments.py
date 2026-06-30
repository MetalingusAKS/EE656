

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_blobs, make_moons, make_circles

from soc_core import all_indices, soc_cluster
from soc_methods import imc1_cluster, imc2_cluster, kmeans_cluster

# ── Global style constants ────────────────────────────────────────────────────
COLORS        = ['#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261','#264653','#A8DADC','#6D2E46']
METHOD_COLORS = {'SOC': '#E63946', 'IMC-1': '#457B9D', 'IMC-2': '#2A9D8F', 'K-Means': '#E9C46A'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


def plot_fig5_blobs():

    rng2 = np.random.RandomState(42)
    X2 = np.vstack([rng2.randn(100, 2) * 0.4 + [1, 1],
                    rng2.randn(100, 2) * 0.4 + [3, 3]])

    rng3 = np.random.RandomState(21)
    X3 = np.vstack([rng3.randn(100, 2) * 0.5 + c
                    for c in [(0, 0), (2.5, 0), (1.25, 2)]])

    rng4 = np.random.RandomState(7)
    X4 = np.vstack([rng4.randn(100, 2) * 0.35 + c
                    for c in [(1, 1), (4, 1), (1, 4), (4, 4)]])

    datasets = [
        ("2-Color Dataset\n(k=2, like paper Fig. 1)",  X2, 2),
        ("3-Color Dataset\n(k=3, like paper Table III)", X3, 3),
        ("4-Color Dataset\n(k=4, like paper Table III)", X4, 4),
    ]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#F0F4F8')
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    for row, (name, X, nk) in enumerate(datasets):
        print(f"  Blobs {name.split(chr(10))[0]}")

        soc_idx,  soc_cc,  hist = soc_cluster(X, nk)
        imc1_idx, imc1_cc       = imc1_cluster(X, nk)
        imc2_idx, imc2_cc       = imc2_cluster(X, nk)
        km_idx,   km_cc         = kmeans_cluster(X, nk)

        soc_g,  soc_p,  soc_s,  soc_d  = all_indices(X, soc_idx,  soc_cc)
        i1_g,   i1_p,   i1_s,   i1_d   = all_indices(X, imc1_idx, imc1_cc)
        i2_g,   i2_p,   i2_s,   i2_d   = all_indices(X, imc2_idx, imc2_cc)
        km_g,   km_p,   km_s,   km_d   = all_indices(X, km_idx,   km_cc)
        print(f"    SOC:{soc_g:.4f}  IMC1:{i1_g:.4f}  IMC2:{i2_g:.4f}  KM:{km_g:.4f}")

        inner = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[row], wspace=0.35)
        mdata = [
            ('SOC (Proposed)', soc_idx,  soc_g, soc_p, soc_s, soc_d, True),
            ('IMC-1',         imc1_idx,  i1_g,  i1_p,  i1_s,  i1_d,  False),
            ('IMC-2',         imc2_idx,  i2_g,  i2_p,  i2_s,  i2_d,  False),
            ('K-Means',       km_idx,    km_g,  km_p,  km_s,  km_d,  False),
        ]

        # ── Cluster scatter plots ─────────────────────────────────────────
        for col, (mname, labels, g, p, s, d, is_soc) in enumerate(mdata):
            ax = fig.add_subplot(inner[col])
            for m in np.unique(labels):
                mask = labels == m
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=COLORS[m % len(COLORS)], s=30, alpha=0.8,
                           edgecolors='white', lw=0.4)
            color = '#C1121F' if is_soc else '#333'
            ax.set_title(f'{mname}\nGSI={g:.4f}  PI={p:.4f}',
                         fontsize=9, fontweight='bold', color=color)
            ax.set_xlabel(f'SI={s:.4f}  DI={d:.4f}', fontsize=7, color='#555')
            ax.grid(True, alpha=0.2)
            ax.spines[['top', 'right']].set_visible(False)
            if is_soc:
                for sp in ax.spines.values():
                    sp.set_edgecolor('#E63946'); sp.set_linewidth(2)

        # ── SOC convergence plot ──────────────────────────────────────────
        ax = fig.add_subplot(inner[4])
        iters   = [h['iter'] for h in hist]
        gsi_vals = [h['GSI'] for h in hist]
        best_i  = int(np.argmax(gsi_vals))
        ax.plot(iters, gsi_vals, 'o-', color='#264653', lw=1.8, ms=5)
        ax.scatter([iters[best_i]], [gsi_vals[best_i]], s=150, color='#E63946', zorder=5,
                   label=f'Best GSI={gsi_vals[best_i]:.4f}\n@iter {iters[best_i]}')
        ax.set_title('SOC GSI Convergence\n(Lagrange β-Optimisation)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Iteration'); ax.set_ylabel('GSI')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks(iters)

        fig.text(0.01, 0.97 - row * 0.335, name, fontsize=11, fontweight='bold',
                 color='#1D3557', transform=fig.transFigure)

    fig.suptitle(
        'SOC on Synthetic Datasets — Cluster Assignments (All Methods) + Convergence',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig('fig5_blobs.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved fig5_blobs.png")



def plot_fig6_preferred_shapes():

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('#F0F4F8')
    outer = gridspec.GridSpec(5, 1, figure=fig, hspace=0.6)
    rng = np.random.RandomState(42)

    # ── K-Means ───────────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.3)
    titles = [
        'Spherical Clusters\n(Customer Segmentation)',
        'Compact Equal-Sized\n(Image Compression)',
        'Well-Separated Blobs\n(Market Research)',
    ]
    for i, t in enumerate(titles):
        ax = fig.add_subplot(inner[i])
        X, y = make_blobs(200, centers=3, cluster_std=0.8 + i * 0.3, random_state=i * 10)
        for m in np.unique(y):
            ax.scatter(X[y == m, 0], X[y == m, 1],
                       c=COLORS[m], s=25, alpha=0.7, edgecolors='w', lw=0.3)
        ax.set_title(t, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.01, 0.955, 'K-Means: Preferred Cluster Shapes',
             fontsize=11, fontweight='bold', color='#457B9D', transform=fig.transFigure)

    # ── Fuzzy C-Means ─────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.3)
    fcm_titles = [
        'Overlapping Clusters\n(Medical Diagnosis)',
        'Soft Boundaries\n(Pattern Recognition)',
        'Variable Density\n(Brain Imaging)',
    ]
    for i, t in enumerate(fcm_titles):
        ax = fig.add_subplot(inner[i])
        if i == 0:
            for m, (cx, cy) in enumerate([(2, 2), (3, 3), (4, 2)]):
                pts = rng.multivariate_normal([cx, cy], [[1, .5], [.5, 1]], 100)
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=COLORS[m], s=25, alpha=0.5, edgecolors='w', lw=0.3)
        elif i == 1:
            X, y = make_blobs(300, centers=3, cluster_std=1.5, random_state=42)
            for m in np.unique(y):
                ax.scatter(X[y == m, 0], X[y == m, 1],
                           c=COLORS[m], s=25, alpha=0.45, edgecolors='w', lw=0.3)
        else:
            for m, ((cx, cy), std) in enumerate([((0, 0), .5), ((4, 4), 1.5), ((0, 4), 1.)]):
                pts = rng.multivariate_normal([cx, cy], [[std, 0], [0, std]], 120)
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=COLORS[m], s=25, alpha=0.6, edgecolors='w', lw=0.3)
        ax.set_title(t, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.01, 0.765, 'Fuzzy C-Means: Preferred Cluster Shapes',
             fontsize=11, fontweight='bold', color='#2A9D8F', transform=fig.transFigure)

    # ── Mountain / IMC ────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.3)
    mc_titles = [
        'Multi-Modal Peaks\n(Signal Processing)',
        'Density Gradients\n(Geographic Analysis)',
        'Irregular Peak Regions\n(Climate Modeling)',
    ]
    for i, t in enumerate(mc_titles):
        ax = fig.add_subplot(inner[i])
        if i == 0:
            for m, c in enumerate([(1, 1), (3, 1), (2, 3), (0, 3)]):
                pts = rng.multivariate_normal(list(c), [[.3, 0], [0, .3]], 80)
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=COLORS[m], s=25, alpha=0.7, edgecolors='w', lw=0.3)
        elif i == 1:
            ctr = rng.multivariate_normal([2, 2], [[.2, 0], [0, .2]], 150)
            sur = rng.multivariate_normal([2, 2], [[1.5, 0], [0, 1.5]], 100)
            ax.scatter(sur[:, 0], sur[:, 1], c=COLORS[1], s=15, alpha=0.4)
            ax.scatter(ctr[:, 0], ctr[:, 1], c=COLORS[0], s=25, alpha=0.8)
        else:
            theta = np.linspace(0, 2 * np.pi, 200)
            for m, (r_base, amp, cx, cy, color) in enumerate([
                (2, .5, 2, 2, COLORS[0]),
                (1.5, .3, -2, -1, COLORS[1]),
            ]):
                r = r_base + amp * np.sin(5 * theta)
                ax.scatter(r * np.cos(theta) + cx, r * np.sin(theta) + cy,
                           c=color, s=12, alpha=0.8)
        ax.set_title(t, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.01, 0.575, 'Mountain / IMC: Preferred Cluster Shapes',
             fontsize=11, fontweight='bold', color='#E9C46A', transform=fig.transFigure)

    # ── DBSCAN ────────────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.3)
    db_titles = [
        'Crescent Shapes\n(Fraud Detection)',
        'Nested Circles\n(Social Network Analysis)',
        'Variable Density + Noise\n(Anomaly Detection)',
    ]
    dsets = [
        make_moons(300, noise=0.1, random_state=42),
        make_circles(300, factor=0.5, noise=0.1, random_state=42),
        None,
    ]
    for i, (t, ds) in enumerate(zip(db_titles, dsets)):
        ax = fig.add_subplot(inner[i])
        if ds is not None:
            X2d, y2d = ds
            for m in np.unique(y2d):
                ax.scatter(X2d[y2d == m, 0], X2d[y2d == m, 1],
                           c=COLORS[m], s=25, alpha=0.7, edgecolors='w', lw=0.3)
        else:
            for m, ((cx, cy), std) in enumerate([((0, 0), .3), ((3, 3), .8), ((0, 3), .2)]):
                pts = rng.multivariate_normal([cx, cy], [[std, 0], [0, std]], 100)
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=COLORS[m], s=25, alpha=0.7, edgecolors='w', lw=0.3)
            noise = rng.uniform(-1, 5, (40, 2))
            ax.scatter(noise[:, 0], noise[:, 1], c='gray', s=12, alpha=0.4, label='Noise')
        ax.set_title(t, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.01, 0.385, 'DBSCAN: Preferred Cluster Shapes',
             fontsize=11, fontweight='bold', color='#F4A261', transform=fig.transFigure)

    # ── SOC ───────────────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4], wspace=0.3)
    soc_titles = [
        'Adaptive Threshold\n(Auto-selects cluster density)',
        'Multi-Modal with\nOptimised δ_m per Cluster',
        'Progressive Refinement\n(β_m via Lagrange root η)',
    ]
    for i, t in enumerate(soc_titles):
        ax = fig.add_subplot(inner[i])
        X_s, y_s = make_blobs(250, centers=3 + i, cluster_std=0.6 + i * 0.2,
                               random_state=i * 7)
        soc_idx_s, _, _ = soc_cluster(X_s, 3 + i)
        for m in np.unique(soc_idx_s):
            mask = soc_idx_s == m
            ax.scatter(X_s[mask, 0], X_s[mask, 1],
                       c=COLORS[m % len(COLORS)], s=28, alpha=0.78,
                       edgecolors='white', lw=0.4)
        ax.set_title(t, fontsize=9, fontweight='bold', color='#C1121F')
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#E63946'); sp.set_linewidth(1.5)
    fig.text(0.01, 0.195, 'SOC (Proposed): Preferred Cluster Shapes',
             fontsize=11, fontweight='bold', color='#E63946', transform=fig.transFigure)

    fig.suptitle(
        'Preferred Cluster Shapes for Each Clustering Method\n'
        '(K-Means · Fuzzy C-Means · Mountain/IMC · DBSCAN · SOC)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig('fig6_preferred_shapes.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved fig6_preferred_shapes.png")
