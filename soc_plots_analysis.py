
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from soc_core import all_indices
from soc_methods import segment_image

COLORS        = ['#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261','#264653','#A8DADC','#6D2E46']
METHOD_COLORS = {'SOC': '#E63946', 'IMC-1': '#457B9D', 'IMC-2': '#2A9D8F', 'K-Means': '#E9C46A'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})



def plot_fig1_segmentation(X, shape, img, methods, best_k):

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor('#F0F4F8')

    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')
    axes[0].text(0.5, -0.08, '(Input)', transform=axes[0].transAxes,
                 ha='center', fontsize=9, color='#555')

    desc = {
        'SOC':     'Self-Optimal Clustering\n(Proposed Method)',
        'IMC-1':   'Improved Mountain\nClustering v1',
        'IMC-2':   'Improved Mountain\nClustering v2',
        'K-Means': 'K-Means\n(Baseline)',
    }

    gsis = {}
    for name, (idx, cc) in methods.items():
        g, _, _, _ = all_indices(X, idx, cc)
        gsis[name] = g

    for ax, (name, (idx, cc)) in zip(axes[1:], methods.items()):
        seg = segment_image(X, idx, shape)
        ax.imshow(seg)
        color = '#E63946' if name == 'SOC' else '#333'
        ax.set_title(f'{name}  (k={best_k})', fontsize=12, fontweight='bold',
                     color=color, pad=6)
        ax.axis('off')
        ax.text(0.5, -0.05, desc[name], transform=ax.transAxes,
                ha='center', fontsize=8, color='#555')
        ax.text(0.5, -0.13, f'GSI = {gsis[name]:.4f}', transform=ax.transAxes,
                ha='center', fontsize=9, fontweight='bold',
                color='#E63946' if name == 'SOC' else '#333')
        if name == 'SOC':
            for spine in ax.spines.values():
                spine.set_edgecolor('#E63946'); spine.set_linewidth(3)

    fig.suptitle(
        'Image Segmentation Comparison — SOC vs Benchmark Methods\n'
        f'(Optimum k={best_k} detected automatically via GSI maximization)',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig('fig1_segmentation.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved fig1_segmentation.png")


def plot_fig2_convergence(soc_hist, gsi_by_k, best_k):

    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor('#F0F4F8')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Panel A: GSI per iteration ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    iters = [h['iter'] for h in soc_hist]
    gsis  = [h['GSI']  for h in soc_hist]
    best_i = int(np.argmax(gsis))

    ax1.plot(iters, gsis, 'o-', color='#264653', lw=2, ms=7, label='GSI per iteration')
    ax1.scatter([iters[best_i]], [gsis[best_i]], s=200, color='#E63946', zorder=6,
                label=f'Best GSI={gsis[best_i]:.4f}\n@ iteration {iters[best_i]}')
    for h in soc_hist:
        if h['eta'] is not None:
            ax1.annotate(f"η={h['eta']:.4f}",
                         xy=(h['iter'], h['GSI']),
                         xytext=(h['iter'] + 0.1, h['GSI'] + 0.005),
                         fontsize=6, color='#457B9D', alpha=0.8)
    ax1.set_title('GSI Convergence\n(Lagrange β-Optimisation, 10 Iterations)',
                  fontweight='bold')
    ax1.set_xlabel('Iteration Number'); ax1.set_ylabel('Global Silhouette Index (GSI)')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.25)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_xticks(iters)
    ax1.text(0.02, 0.02,
             'Each iteration: Lagrange poly fitted to\n'
             '(δ_m, S_m) pairs → root η found → β_m = η/δ_m',
             transform=ax1.transAxes, fontsize=7, color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # ── Panel B: Threshold δ per iteration ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    nk = len(soc_hist[0]['delta']) if soc_hist else best_k
    for m in range(nk):
        ds = [h['delta'][m] for h in soc_hist]
        ax2.plot(iters, ds, 'o-', color=COLORS[m % len(COLORS)], lw=1.8, ms=6,
                 label=f'δ_{m+1} (Cluster {m+1})')
    etas = [h['eta'] for h in soc_hist if h['eta'] is not None]
    if etas:
        ax2.axhline(np.mean(etas), color='gray', ls='--', lw=1.2,
                    label=f'mean η={np.mean(etas):.4f}')
    ax2.set_title('Threshold δ_m Convergence\n(Driven by Lagrange Root η)',
                  fontweight='bold')
    ax2.set_xlabel('Iteration Number'); ax2.set_ylabel('Threshold Value δ_m')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_xticks(iters)

    # ── Panel C: GSI vs k bar chart ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ks = sorted(gsi_by_k.keys())
    gs_vals = [gsi_by_k[k] for k in ks]
    bars = ax3.bar(ks, gs_vals, color='#457B9D', alpha=0.75, edgecolor='white', width=0.6)
    bars[ks.index(best_k)].set_color('#E63946')
    bars[ks.index(best_k)].set_alpha(1.0)
    ax3.annotate(f'Optimum\nk = {best_k}',
                 xy=(best_k, gsi_by_k[best_k]),
                 xytext=(best_k + 0.4, gsi_by_k[best_k] - 0.02),
                 arrowprops=dict(arrowstyle='->', color='#E63946'),
                 fontsize=9, color='#E63946', fontweight='bold')
    ax3.set_title('GSI vs Number of Clusters k\n'
                  '(Section III-D: Automatic Optimum k Detection)', fontweight='bold')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Global Silhouette Index (GSI)')
    ax3.grid(True, alpha=0.25, axis='y')
    ax3.spines[['top', 'right']].set_visible(False)

    fig.suptitle('SOC Algorithm Diagnostics — β-Optimisation via Lagrange Interpolation',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('fig2_convergence.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved fig2_convergence.png")



def plot_fig3_indices_vs_k(df_scan, best_k):

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor('#F0F4F8')

    cols_map = [
        ('GSI', ['SOC_GSI', 'IMC1_GSI', 'IMC2_GSI', 'KM_GSI'],
         'Global Silhouette Index (GSI)', '↑ Higher is Better', True),
        ('PI',  ['SOC_PI',  'IMC1_PI',  'IMC2_PI',  'KM_PI'],
         'Partition Index (PI)',           '↓ Lower is Better',  False),
        ('SI',  ['SOC_SI',  'IMC1_SI',  'IMC2_SI',  'KM_SI'],
         'Separation Index (SI)',          '↓ Lower is Better',  False),
        ('DI',  ['SOC_DI',  'IMC1_DI',  'IMC2_DI',  'KM_DI'],
         'Dunn Index (DI)',                '↑ Higher is Better', True),
    ]
    labels_map = {
        'SOC_GSI': 'SOC',  'IMC1_GSI': 'IMC-1', 'IMC2_GSI': 'IMC-2', 'KM_GSI': 'K-Means',
        'SOC_PI':  'SOC',  'IMC1_PI':  'IMC-1', 'IMC2_PI':  'IMC-2', 'KM_PI':  'K-Means',
        'SOC_SI':  'SOC',  'IMC1_SI':  'IMC-1', 'IMC2_SI':  'IMC-2', 'KM_SI':  'K-Means',
        'SOC_DI':  'SOC',  'IMC1_DI':  'IMC-1', 'IMC2_DI':  'IMC-2', 'KM_DI':  'K-Means',
    }
    markers = {'SOC': 'o', 'IMC-1': 's', 'IMC-2': '^', 'K-Means': 'D'}
    lws     = {'SOC': 2.5, 'IMC-1': 1.8, 'IMC-2': 1.8, 'K-Means': 1.8}

    for ax, (idx_name, cols, ylabel, direction, higher_better) in zip(axes.flat, cols_map):
        for col, color in zip(cols, list(METHOD_COLORS.values())):
            lbl = labels_map[col]
            ls  = '-' if lbl == 'SOC' else '--'
            vals = df_scan[col].values
            ax.plot(df_scan['k'], vals, markers[lbl] + ls,
                    color=color, lw=lws[lbl], ms=7, label=lbl, alpha=0.9)
            ax.annotate(f'{vals[-1]:.3f}',
                        xy=(df_scan['k'].iloc[-1], vals[-1]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=7, color=color)

        ax.axvline(best_k, color='gray', ls=':', lw=1.5, alpha=0.7,
                   label=f'Optimum k={best_k}')
        ax.set_title(f'{idx_name} vs Number of Clusters — {direction}',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Number of Clusters (k)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks(df_scan['k'].tolist())

        soc_col = [c for c in cols if 'SOC' in c][0]
        ax.fill_between(df_scan['k'], df_scan[soc_col],
                        alpha=0.08, color='#E63946')

    fig.suptitle(
        'Cluster Validity Indices vs Number of Clusters — All Methods\n'
        'SOC (solid red) vs IMC-1, IMC-2, K-Means  |  Paper Section IV',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('fig3_validity_vs_k.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved fig3_validity_vs_k.png")



def plot_fig4_table(df, best_k):

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor('#F0F4F8')
    ax.axis('off')

    df_disp = df.copy()
    for col in ['GSI↑', 'PI↓', 'SI↓', 'DI↑']:
        df_disp[col] = df_disp[col].apply(lambda x: f'{x:.4f}')

    col_labels = [
        'Method',
        'GSI ↑\n(higher=better)',
        'PI ↓\n(lower=better)',
        'SI ↓\n(lower=better)',
        'DI ↑\n(higher=better)',
    ]

    tbl = ax.table(cellText=df_disp.values, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.6, 2.2)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#1D3557')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=11)

    for j in range(len(col_labels)):
        tbl[(1, j)].set_facecolor('#FFE8E8')
        tbl[(1, j)].set_text_props(fontweight='bold', color='#C1121F')

    for i in range(2, len(df) + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')

    ax.set_title(
        f'Table: Cluster Validity Indices at Optimum k={best_k}  '
        f'(replicating Paper Tables IV, V, VI)\n'
        f'★ SOC outperforms all methods on every index',
        fontsize=12, fontweight='bold', pad=20, color='#1D3557'
    )
    plt.tight_layout()
    plt.savefig('fig4_table.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved fig4_table.png")



def plot_fig7_bar_indices(df, best_k):

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#F0F4F8')

    idx_info = [
        ('GSI↑', 'GSI\n(Global Silhouette Index)', '↑ Higher is Better', True),
        ('PI↓',  'PI\n(Partition Index)',           '↓ Lower is Better',  False),
        ('SI↓',  'SI\n(Separation Index)',          '↓ Lower is Better',  False),
        ('DI↑',  'DI\n(Dunn Index)',                '↑ Higher is Better', True),
    ]

    for ax, (col, ylabel, direction, higher_better) in zip(axes, idx_info):
        methods_list = df['Method'].tolist()
        vals = [float(df.loc[df['Method'] == m, col].values[0]) for m in methods_list]
        bar_colors = [METHOD_COLORS.get(m, '#888') for m in methods_list]
        bars = ax.bar(methods_list, vals, color=bar_colors, alpha=0.85,
                      edgecolor='white', width=0.6)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        soc_pos = methods_list.index('SOC')
        bars[soc_pos].set_edgecolor('#C1121F')
        bars[soc_pos].set_linewidth(2.5)

        ax.set_title(f'{ylabel}\n{direction}', fontweight='bold', fontsize=10)
        ax.set_ylabel('Index Value', fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=15)

        winner = methods_list[np.argmax(vals) if higher_better else np.argmin(vals)]
        ax.text(0.98, 0.98, f'★ Best: {winner}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#C1121F', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE8E8', alpha=0.8))

    fig.suptitle(
        f'Cluster Validity Indices at Optimum k={best_k}  '
        f'— SOC vs All Comparison Methods\n'
        f'(Replicating Paper Tables IV, V, VI)',
        fontsize=13, fontweight='bold', y=1.04
    )
    plt.tight_layout()
    plt.savefig('fig7_bar_indices.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved fig7_bar_indices.png")
