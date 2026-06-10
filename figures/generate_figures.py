"""
Publication-quality SVG figures — complete rewrite.
  Fig 6: Deferral sensitivity heatmap (Table 8)
  Fig 7: Deferral policy grouped bars (Table 9)
  Fig 8: Cost matrix sensitivity dual-axis (Table 10)
  Fig 9: Ablation waterfall (Table 12)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import os

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
})

C_DIMM  = '#2166AC'
C_STEEL = '#E67E22'
C_SECOM = '#1B7837'
C_GRAY  = '#666666'

def save(fig, name):
    for fmt in ['svg', 'png']:
        fig.savefig(os.path.join(OUT, f'{name}.{fmt}'),
                    format=fmt, bbox_inches='tight', pad_inches=0.15,
                    dpi=600 if fmt == 'png' else None)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Heatmap: Deferral Sensitivity (Table 8)
# Two clean panels using imshow, generous cell sizes
# ═════════════════════════════════════════════════════════════════════════════

def make_fig6():
    baseline = {'DIMM': 410.6, 'Steel': 251.8, 'SECOM': 335.9}
    datasets = ['DIMM', 'Steel Plates', 'SECOM']
    ds_keys  = ['DIMM', 'Steel', 'SECOM']

    # Panel A: Review cost (q = 1.0)
    rc_labels = ['$r_c$ = 0.25', '$r_c$ = 0.50', '$r_c$ = 1.00',
                 '$r_c$ = 2.00\n(none deferred)']
    rc_costs = [
        [338.8, 173.5, 271.6],
        [363.9, 198.2, 294.7],
        [401.9, 245.0, 335.9],
        [410.6, 251.8, 335.9],
    ]
    rc_pct = [[(baseline[dk] - row[j]) / baseline[dk] * 100
               for j, dk in enumerate(ds_keys)] for row in rc_costs]

    # Panel B: Expert accuracy (rc = 0.5)
    qa_labels = ['q = 0.80', 'q = 0.90', 'q = 0.95', 'q = 1.00']
    qa_costs = [
        [409.8, 256.5, 328.7],
        [386.3, 225.8, 309.7],
        [374.8, 212.7, 303.0],
        [363.9, 198.2, 294.7],
    ]
    qa_pct = [[(baseline[dk] - row[j]) / baseline[dk] * 100
               for j, dk in enumerate(ds_keys)] for row in qa_costs]

    # Global color range
    all_vals = [v for row in rc_pct + qa_pct for v in row]
    vmin_abs = abs(min(all_vals))
    vmax = max(all_vals) * 1.05
    norm = TwoSlopeNorm(vmin=-max(3, vmin_abs * 1.2), vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('rdbu',
        ['#C62828', '#EF9A9A', '#FFEBEE', '#FFFFFF',
         '#BBDEFB', '#64B5F6', '#1565C0'], N=256)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 4.0),
                                      gridspec_kw={'wspace': 0.45})

    def draw_panel(ax, pct_matrix, cost_matrix, row_labels, title):
        mat = np.array(pct_matrix)
        n_r, n_c = mat.shape

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto',
                        interpolation='nearest')

        # Grid lines
        for i in range(n_r + 1):
            ax.axhline(i - 0.5, color='white', linewidth=2.5)
        for j in range(n_c + 1):
            ax.axvline(j - 0.5, color='white', linewidth=2.5)

        # Cell text
        for i in range(n_r):
            for j in range(n_c):
                val = mat[i, j]
                cost = cost_matrix[i][j]
                # Pick text color for contrast
                if abs(val) > vmax * 0.45:
                    tc = 'white'
                else:
                    tc = '#1a1a1a'

                pct_str = f'{val:+.1f}%' if abs(val) >= 0.05 else '0.0%'
                ax.text(j, i - 0.12, pct_str, ha='center', va='center',
                        fontsize=11, fontweight='bold', color=tc)
                ax.text(j, i + 0.25, f'({cost:.0f})', ha='center', va='center',
                        fontsize=8.5, color=tc, alpha=0.85)

        ax.set_xticks(range(n_c))
        ax.set_xticklabels(datasets, fontsize=10, fontweight='bold')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', length=0, pad=6)

        ax.set_yticks(range(n_r))
        ax.set_yticklabels(row_labels, fontsize=9.5)
        ax.tick_params(axis='y', length=0, pad=8)

        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=28)

        # Remove frame
        for spine in ax.spines.values():
            spine.set_visible(False)

    draw_panel(ax_a, rc_pct, rc_costs, rc_labels,
               '(a) Review Cost Sensitivity\n(q = 1.0, 10% cap)')
    draw_panel(ax_b, qa_pct, qa_costs, qa_labels,
               '(b) Expert Accuracy Sensitivity\n($r_c$ = 0.5, 10% cap)')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.25, 0.02, 0.50, 0.028])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Cost reduction from no-deferral baseline (%)',
                   fontsize=8.5, labelpad=5)
    cbar.ax.tick_params(labelsize=8)

    # Baseline note
    fig.text(0.50, 0.075,
             'No-deferral baselines:  DIMM = 410.6,  Steel = 251.8,  SECOM = 335.9',
             ha='center', fontsize=8, color='#888888', style='italic')

    plt.subplots_adjust(bottom=0.15, top=0.82)
    save(fig, 'fig6_deferral_sensitivity_heatmap')
    print('  Fig 6 done.')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Grouped Bar Chart: Policy Comparison (Table 9)
# ═════════════════════════════════════════════════════════════════════════════

def make_fig7():
    policies = ['No deferral', 'Risk-based', 'Confidence', 'Risk-margin', 'Entropy']
    dimm  = [410.5, 364.4, 375.5, 385.5, 396.8]
    steel = [236.6, 197.8, 211.9, 221.9, 230.7]
    secom = [311.4, 294.7, 308.2, 300.0, 311.4]

    x = np.arange(len(policies))
    w = 0.24

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    bd = ax.bar(x - w, dimm,  w, color=C_DIMM,  edgecolor='white', lw=0.5,
                label='DIMM', zorder=3)
    bs = ax.bar(x,     steel, w, color=C_STEEL, edgecolor='white', lw=0.5,
                label='Steel Plates', zorder=3)
    be = ax.bar(x + w, secom, w, color=C_SECOM, edgecolor='white', lw=0.5,
                label='SECOM', zorder=3)

    # Highlight risk-based bars with gold edge
    for bars in [bd, bs, be]:
        bars[1].set_edgecolor('#DAA520')
        bars[1].set_linewidth(2.0)

    # Value labels only on risk-based (best)
    for bars, vals in [(bd, dimm), (bs, steel), (be, secom)]:
        b = bars[1]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4,
                f'{vals[1]:.0f}', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold')

    # Dotted baselines
    for val, c in [(dimm[0], C_DIMM), (steel[0], C_STEEL), (secom[0], C_SECOM)]:
        ax.axhline(val, color=c, lw=0.6, ls=':', alpha=0.45, zorder=1)

    ax.set_ylabel('Total Operational Cost per 1000')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=8.5)
    ax.set_ylim(0, max(dimm) * 1.18)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='#CCC', framealpha=0.95)
    ax.yaxis.grid(True, alpha=0.25, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    # No annotation — the gold-bordered bars and value labels speak for themselves

    plt.tight_layout()
    save(fig, 'fig7_policy_comparison_bars')
    print('  Fig 7 done.')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Dual-axis: Cost Matrix Sensitivity (Table 10)
# Fix: no text on lines, labels in margin, remove "Monotonic" annotation
# ═════════════════════════════════════════════════════════════════════════════

def make_fig8():
    penalties  = [3, 5, 7, 10]
    cost       = [380.4, 427.8, 452.4, 510.4]
    recall     = [6.1, 14.1, 29.4, 44.4]
    false_hc   = [1.2, 1.9, 3.1, 4.2]
    scenarios  = ['Mild', 'Base', 'Severe', 'Extreme']

    fig, ax1 = plt.subplots(figsize=(5.8, 3.8))
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)

    # ── Lines ──
    ln1 = ax1.plot(penalties, cost, 's-', color=C_DIMM, ms=8,
                   mfc='white', mew=1.8, lw=2.2, label='Total Cost', zorder=4)
    ln2 = ax2.plot(penalties, recall, 'o-', color=C_STEEL, ms=8,
                   mfc='white', mew=1.8, lw=2.2, label='HC Recall (%)', zorder=4)
    ln3 = ax2.plot(penalties, false_hc, '^--', color=C_SECOM, ms=7,
                   mfc='white', mew=1.4, lw=1.5, label='False HC (%)', zorder=4)

    # ── Data labels — placed AWAY from lines, with white outline for clarity ──
    outline = [pe.withStroke(linewidth=3, foreground='white')]

    # Cost labels: always above the cost line
    for i, (p, c) in enumerate(zip(penalties, cost)):
        ax1.text(p, c + 12, f'{c:.0f}', ha='center', va='bottom',
                 fontsize=8.5, fontweight='bold', color=C_DIMM,
                 path_effects=outline)

    # Recall labels: placed to avoid the cost line
    # At x=3,5: recall is LOW, put label below marker
    # At x=7: lines cross — put label far right
    # At x=10: recall is high, put label below marker
    recall_positions = [
        (3, recall[0], 0, -14, 'center'),    # below
        (5, recall[1], 0, -14, 'center'),    # below
        (7, recall[2], 14,  0, 'left'),      # right side to avoid crossing
        (10, recall[3], 0, -14, 'center'),   # below
    ]
    for px, ry, dx, dy, ha in recall_positions:
        ax2.annotate(f'{ry:.0f}%', (px, ry), xytext=(dx, dy),
                     textcoords='offset points', ha=ha, va='center',
                     fontsize=8.5, fontweight='bold', color=C_STEEL,
                     path_effects=outline)

    # ── Axes ──
    ax1.set_xlabel('High-Cost Class (Scrap) Penalty')
    ax1.set_ylabel('Total Cost per 1000', color=C_DIMM)
    ax2.set_ylabel('Rate (%)', color=C_STEEL)
    ax1.tick_params(axis='y', colors=C_DIMM)
    ax2.tick_params(axis='y', colors=C_STEEL)

    ax1.set_xticks(penalties)
    ax1.set_xticklabels([f'{p}\n({s})' for p, s in zip(penalties, scenarios)],
                         fontsize=8.5)

    ax1.set_ylim(340, 550)
    ax2.set_ylim(0, 55)

    # ── Legend ──
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=True, fancybox=False,
               edgecolor='#CCC', framealpha=0.95, fontsize=8)

    ax1.yaxis.grid(True, alpha=0.2, lw=0.4)
    ax1.set_axisbelow(True)

    plt.tight_layout()
    save(fig, 'fig8_cost_sensitivity_lines')
    print('  Fig 8 done.')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Waterfall: Ablation (Table 12)
# ═════════════════════════════════════════════════════════════════════════════

def make_fig9():
    labels = [
        'A0: XGB\n+ Argmax\n(Baseline)',
        'A1: + Isotonic\nCalibration\n+ Bayes-Risk',
        'A2: + Cost-\nSensitive\nTraining',
        'A3: + Cost-\nBalanced\nWeighting',
        'A4: + Ensemble\n(XGB+LGBM)',
        'A5: + Deferral\n10%',
        'A6: + Deferral\n20%',
    ]
    costs = [543.5, 438.1, 431.6, 436.6, 410.5, 364.4, 353.3]
    deltas = [0] + [costs[i] - costs[i-1] for i in range(1, len(costs))]
    vs_a0 = [(costs[0] - c) / costs[0] * 100 for c in costs]

    n = len(costs)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    x = np.arange(n)
    bw = 0.56

    for i in range(n):
        if i == 0:
            color = '#4A7DB5'
            ax.bar(x[i], costs[0], bw, bottom=0, color=color,
                   edgecolor='white', lw=0.8, zorder=3, alpha=0.88)
            ax.text(x[i], costs[0] + 5, f'{costs[0]:.1f}', ha='center',
                    va='bottom', fontsize=8.5, fontweight='bold', color='#333')
        else:
            d = deltas[i]
            color = '#2CA02C' if d < 0 else '#D62728'
            bottom = min(costs[i], costs[i-1])
            height = abs(d)
            ax.bar(x[i], height, bw, bottom=bottom, color=color,
                   edgecolor='white', lw=0.8, zorder=3, alpha=0.88)

            # Delta label inside bar if tall enough, else outside
            mid = bottom + height / 2
            if height > 18:
                ax.text(x[i], mid, f'{d:+.1f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
            else:
                # Outside above for increases, outside left for small decreases
                if d > 0:
                    ax.text(x[i], costs[i-1] + 3, f'{d:+.1f}', ha='center',
                            va='bottom', fontsize=7.5, fontweight='bold', color=color)
                else:
                    ax.text(x[i], mid, f'{d:+.1f}', ha='center', va='center',
                            fontsize=7.5, fontweight='bold', color='white')

            # Resulting cost below the remaining stack
            ax.text(x[i], costs[i] - 8, f'{costs[i]:.1f}', ha='center',
                    va='top', fontsize=8.5, fontweight='bold', color='#333')

            # % vs A0
            if vs_a0[i] > 0:
                ax.text(x[i], costs[i] - 22,
                        f'−{vs_a0[i]:.1f}% vs A0',
                        ha='center', va='top', fontsize=6.5,
                        color=C_GRAY, style='italic')

        # Connector
        if i < n - 1:
            ax.plot([x[i] + bw/2, x[i+1] - bw/2],
                    [costs[i], costs[i]],
                    color='#aaa', lw=0.6, ls='--', zorder=2)

    # Phase brackets
    ax.annotate('', xy=(-0.1, 268), xytext=(4.1, 268),
                arrowprops=dict(arrowstyle='|-|', color='#888', lw=0.8))
    ax.text(2.0, 258, 'Fully automated framework', ha='center',
            fontsize=7.5, color='#888', style='italic')
    ax.annotate('', xy=(4.6, 268), xytext=(6.4, 268),
                arrowprops=dict(arrowstyle='|-|', color='#888', lw=0.8))
    ax.text(5.5, 258, '+ Human\ndeferral', ha='center',
            fontsize=7.5, color='#888', style='italic')

    ax.set_ylabel('Total Operational Cost per 1000')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha='center')
    ax.set_ylim(240, 580)
    ax.yaxis.grid(True, alpha=0.2, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    legend_el = [
        Patch(fc='#4A7DB5', ec='white', label='Baseline'),
        Patch(fc='#2CA02C', ec='white', label='Cost reduction'),
        Patch(fc='#D62728', ec='white', label='Cost increase'),
    ]
    ax.legend(handles=legend_el, loc='upper right', frameon=True,
              fancybox=False, edgecolor='#CCC', framealpha=0.95, fontsize=7.5)

    plt.tight_layout()
    save(fig, 'fig9_ablation_waterfall')
    print('  Fig 9 done.')


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Creating figures...')
    make_fig6()
    make_fig7()
    make_fig8()
    make_fig9()
    print('All done. Output:', OUT)
