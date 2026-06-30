import warnings
warnings.filterwarnings('ignore')

from soc_methods import run_full_analysis
from soc_plots_analysis import (
    plot_fig1_segmentation,
    plot_fig2_convergence,
    plot_fig3_indices_vs_k,
    plot_fig4_table,
    plot_fig7_bar_indices,
)
from soc_plots_experiments import (
    plot_fig5_blobs,
    plot_fig6_preferred_shapes,
)


def main():
    print("=" * 60)
    print(" SOC — Self-Optimal Clustering (Verma & Roy, 2014)")
    print("=" * 60)

    # ── Run core analysis ─────────────────────────────────────────────────────
    X, shape, img, methods, best_k, gsi_by_k, df, df_scan, soc_hist = run_full_analysis()

    # ── Generate all figures ──────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_fig1_segmentation(X, shape, img, methods, best_k)
    plot_fig2_convergence(soc_hist, gsi_by_k, best_k)
    plot_fig3_indices_vs_k(df_scan, best_k)
    plot_fig4_table(df, best_k)

    print("\nRunning synthetic blob experiments...")
    plot_fig5_blobs()

    print("\nGenerating preferred shape illustrations...")
    plot_fig6_preferred_shapes()

    plot_fig7_bar_indices(df, best_k)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL DONE — 7 figures generated:")
    print("  fig1_segmentation.png     — image segmentation (clearly labelled)")
    print("  fig2_convergence.png      — GSI + delta convergence + GSI vs k")
    print("  fig3_validity_vs_k.png    — all 4 indices vs k for all methods")
    print("  fig4_table.png            — paper-style formatted table")
    print("  fig5_blobs.png            — synthetic datasets (all 4 methods)")
    print("  fig6_preferred_shapes.png — cluster shape illustrations")
    print("  fig7_bar_indices.png      — bar chart all 4 indices at optimum k")
    print("=" * 60)


if __name__ == '__main__':
    main()
