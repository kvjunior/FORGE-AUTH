import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Configure matplotlib for IEEE Transactions style
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman']
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
rcParams['axes.linewidth'] = 0.5
rcParams['grid.linewidth'] = 0.3
rcParams['lines.linewidth'] = 1.0

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
fig.subplots_adjust(hspace=0.35, wspace=0.30, left=0.08, right=0.98, top=0.95, bottom=0.08)

# Color scheme for professional appearance
color_primary = '#1f77b4'
color_secondary = '#ff7f0e'
color_tertiary = '#2ca02c'
color_quaternary = '#d62728'
color_grid = '#e0e0e0'
color_text = '#333333'

# Panel A: Privacy Budget Evolution
ax_a = axes[0, 0]
auth_attempts = np.arange(0, 51)
epsilon_per_auth = 0.019
epsilon_cumulative = auth_attempts * epsilon_per_auth
epsilon_threshold = 0.93

ax_a.plot(auth_attempts, epsilon_cumulative, color=color_primary, linewidth=1.5, label='Cumulative $\\epsilon$')
ax_a.axhline(y=epsilon_threshold, color=color_quaternary, linestyle='--', linewidth=1.0, label='Privacy threshold')
ax_a.fill_between(auth_attempts, 0, epsilon_cumulative, where=(epsilon_cumulative <= epsilon_threshold),
                   color=color_primary, alpha=0.15)

ax_a.set_xlabel('Authentication Attempts', fontsize=9)
ax_a.set_ylabel('Privacy Budget ($\\epsilon$)', fontsize=9)
ax_a.set_xlim(0, 50)
ax_a.set_ylim(0, 1.0)
ax_a.grid(True, alpha=0.3, linewidth=0.3)

# Fixed legend handling for compatibility
legend_a = ax_a.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
legend_a.get_frame().set_linewidth(0.5)

ax_a.text(0.02, 0.95, '(A)', transform=ax_a.transAxes, fontsize=10, fontweight='bold')
ax_a.text(25, 0.5, f'$\\epsilon = {epsilon_per_auth}$ per auth', fontsize=8, ha='center')

# Panel B: Attack Success Rates
ax_b = axes[0, 1]
attacks = ['Membership\nInference', 'Model\nInversion', 'Attribute\nInference', 'Behavioral\nSpoofing']
success_rates = [51.3, 8.2, 54.7, 4.1]
colors_bars = [color_primary, color_secondary, color_tertiary, color_quaternary]

bars = ax_b.bar(range(len(attacks)), success_rates, color=colors_bars, edgecolor='black', linewidth=0.5)
ax_b.axhline(y=50, color='gray', linestyle=':', linewidth=0.8, alpha=0.7, label='Random baseline')

for i, (bar, rate) in enumerate(zip(bars, success_rates)):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
              f'{rate}%', ha='center', va='bottom', fontsize=8)

ax_b.set_xlabel('Attack Type', fontsize=9)
ax_b.set_ylabel('Success Rate (%)', fontsize=9)
ax_b.set_ylim(0, 70)
ax_b.set_xticks(range(len(attacks)))
ax_b.set_xticklabels(attacks, fontsize=7)
ax_b.grid(True, axis='y', alpha=0.3, linewidth=0.3)

# Fixed legend handling
legend_b = ax_b.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')
legend_b.get_frame().set_linewidth(0.5)

ax_b.text(0.02, 0.95, '(B)', transform=ax_b.transAxes, fontsize=10, fontweight='bold')

# Panel C: Information Leakage
ax_c = axes[1, 0]
metrics = ['Bits per\nSession', 'Attribute\nDisclosure', 'Behavioral\nEntropy']
values = [1.24, 6.0, 3.8]
units = ['bits', '%', 'bits']
colors_metrics = [color_primary, color_secondary, color_tertiary]

x_pos = np.arange(len(metrics))
bars_c = ax_c.bar(x_pos, values, color=colors_metrics, edgecolor='black', linewidth=0.5)

for i, (bar, value, unit) in enumerate(zip(bars_c, values, units)):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
              f'{value} {unit}', ha='center', va='bottom', fontsize=8)

ax_c.set_xlabel('Information Leakage Metric', fontsize=9)
ax_c.set_ylabel('Value', fontsize=9)
ax_c.set_ylim(0, 7)
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(metrics, fontsize=7)
ax_c.grid(True, axis='y', alpha=0.3, linewidth=0.3)
ax_c.text(0.02, 0.95, '(C)', transform=ax_c.transAxes, fontsize=10, fontweight='bold')

# Add breakdown for bits per session
ax_c.text(0, 0.5, 'Timing: 0.73\nProof size: 0.51', fontsize=6, ha='center', 
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.5))

# Panel D: Privacy-Utility Tradeoff Curve
ax_d = axes[1, 1]
epsilon_range = np.logspace(-2, 0.5, 100)
delta = 1e-5

# Utility function (sigmoid-like curve)
def utility_function(eps):
    return 1 - 1 / (1 + 10 * eps**0.8)

utility = utility_function(epsilon_range)

ax_d.semilogx(epsilon_range, utility, color=color_primary, linewidth=1.5)
ax_d.scatter([0.93], [utility_function(0.93)], color=color_quaternary, s=50, zorder=5,
             edgecolor='black', linewidth=0.5, label='Operating point')

# Add shaded regions
ax_d.fill_between(epsilon_range[epsilon_range < 0.93], 0, utility[epsilon_range < 0.93],
                  color=color_tertiary, alpha=0.1, label='Strong privacy')
ax_d.fill_between(epsilon_range[epsilon_range >= 0.93], 0, utility[epsilon_range >= 0.93],
                  color=color_quaternary, alpha=0.1, label='Weak privacy')

# Add annotation for operating point
ax_d.annotate(f'($\\epsilon = 0.93$, $\\delta = 10^{{-5}}$)',
              xy=(0.93, utility_function(0.93)), xytext=(0.3, 0.75),
              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                            linewidth=0.5, color='black'),
              fontsize=8, ha='center')

ax_d.set_xlabel('Privacy Budget ($\\epsilon$)', fontsize=9)
ax_d.set_ylabel('Authentication Utility', fontsize=9)
ax_d.set_xlim(0.01, 3)
ax_d.set_ylim(0, 1.05)
ax_d.grid(True, alpha=0.3, linewidth=0.3, which='both')

# Fixed legend handling
legend_d = ax_d.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
legend_d.get_frame().set_linewidth(0.5)

ax_d.text(0.02, 0.95, '(D)', transform=ax_d.transAxes, fontsize=10, fontweight='bold')

# Set minor ticks for better granularity
for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.tick_params(axis='both', which='major', length=3, width=0.5)
    ax.tick_params(axis='both', which='minor', length=2, width=0.3)
    
# Add subtle box around each subplot
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

# Save figure
plt.savefig('privacy_analysis.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('privacy_analysis.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()

print("Figure generated successfully as 'privacy_analysis.pdf' and 'privacy_analysis.png'")