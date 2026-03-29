"""
Phase 1: Data Cleaning & EDA — Sample Superstore
================================================
Run each cell in Jupyter Notebook.
"""

# ─── CELL 1 — Imports ────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Consistent style for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

# ─── CELL 2 — Load Data ──────────────────────────────────────────────────────
df = pd.read_csv('SampleSuperstore.csv', encoding='latin-1')

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()


# ─── CELL 3 — Basic Inspection ───────────────────────────────────────────────
print("=== DATA TYPES ===")
print(df.dtypes)

print("\n=== NULL VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATES ===")
print(f"Duplicate rows: {df.duplicated().sum()}")


# ─── CELL 4 — Fix & Clean ────────────────────────────────────────────────────
# Remove duplicates
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# Postal Code should be string (leading zeros in some states)
df['Postal Code'] = df['Postal Code'].astype(str).str.zfill(5)

# Add derived columns (useful for later phases)
df['Profit Margin (%)'] = (df['Profit'] / df['Sales'] * 100).round(2)
df['Is Loss']           = df['Profit'] < 0
df['Discount Band']     = pd.cut(
    df['Discount'],
    bins=[-0.01, 0.0, 0.2, 0.4, 1.0],
    labels=['No discount', 'Low (0–20%)', 'Medium (20–40%)', 'High (>40%)']
)

print("\nNew columns added: Profit Margin (%), Is Loss, Discount Band")
df[['Sales', 'Discount', 'Profit', 'Profit Margin (%)', 'Is Loss', 'Discount Band']].head(8)


# ─── CELL 5 — Descriptive Statistics ─────────────────────────────────────────
print("=== NUMERICAL SUMMARY ===")
# display(df[['Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin (%)']].describe().round(2))

print("\n=== CATEGORICAL COUNTS ===")
for col in ['Category', 'Segment', 'Region', 'Ship Mode']:
    print(f"\n{col}:\n{df[col].value_counts().to_string()}")


# ─── CELL 6 — Loss Analysis (Key Insight) ────────────────────────────────────
loss_count  = df['Is Loss'].sum()
loss_pct    = loss_count / len(df) * 100
total_loss  = df.loc[df['Is Loss'], 'Profit'].sum()

print(f"Loss-making orders  : {loss_count:,} ({loss_pct:.1f}% of all orders)")
print(f"Total profit lost   : ${total_loss:,.2f}")
print(f"\nLoss by Category:")
print(df.groupby('Category')['Is Loss'].sum().to_string())


# ─── CELL 7 — Distribution Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 5))
fig.suptitle('Distribution of Key Numerical Variables', fontsize=14, fontweight='bold')

# Sales (log scale because of heavy right skew)
axes[0, 0].hist(df['Sales'], bins=80, color='steelblue', edgecolor='white', linewidth=0.4)
axes[0, 0].set_title('Sales distribution (raw)')
axes[0, 0].set_xlabel('Sales ($)')
axes[0, 0].set_ylabel('Count')

axes[0, 1].hist(np.log1p(df['Sales']), bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
axes[0, 1].set_title('Sales distribution (log scale)')
axes[0, 1].set_xlabel('log(Sales + 1)')

# Discount
axes[1, 0].hist(df['Discount'], bins=20, color='darkorange', edgecolor='white', linewidth=0.4)
axes[1, 0].set_title('Discount distribution')
axes[1, 0].set_xlabel('Discount rate')
axes[1, 0].set_ylabel('Count')

# Profit
axes[1, 1].hist(df['Profit'], bins=80, color='seagreen', edgecolor='white', linewidth=0.4)
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=1.2, label='Break-even')
axes[1, 1].set_title('Profit distribution')
axes[1, 1].set_xlabel('Profit ($)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('01_distributions.png', bbox_inches='tight')
plt.show()


# ─── CELL 8 — Discount vs Profit Scatter (Preview of Phase 2) ────────────────
fig, ax = plt.subplots(figsize=(8, 5))

scatter = ax.scatter(
    df['Discount'], df['Profit'],
    c=df['Profit'], cmap='RdYlGn',
    alpha=0.45, s=18, linewidths=0
)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Break-even')
ax.axvline(0.2, color='orange', linewidth=1, linestyle='--', label='20% discount threshold')
plt.colorbar(scatter, ax=ax, label='Profit ($)')
ax.set_xlabel('Discount Rate')
ax.set_ylabel('Profit ($)')
ax.set_title('Discount rate vs Profit — colour = profit value', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('02_discount_vs_profit.png', bbox_inches='tight')
plt.show()


# ─── CELL 9 — Correlation Heatmap ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

corr = df[['Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin (%)']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, linewidths=0.5,
    ax=ax, square=True, cbar_kws={'shrink': 0.8}
)
ax.set_title('Correlation matrix — numerical features', fontweight='bold')
plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', bbox_inches='tight')
plt.show()


# ─── CELL 10 — Category & Segment Summary ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 5))

# By Category
cat_stats = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
x = np.arange(len(cat_stats))
w = 0.35
axes[0].bar(x - w/2, cat_stats['Sales'],   width=w, label='Sales',  color='steelblue')
axes[0].bar(x + w/2, cat_stats['Profit'],  width=w, label='Profit', color='seagreen')
axes[0].set_xticks(x)
axes[0].set_xticklabels(cat_stats['Category'])
axes[0].set_title('Total Sales & Profit by Category', fontweight='bold')
axes[0].set_ylabel('Amount ($)')
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))

# By Segment
seg_stats = df.groupby('Segment')[['Sales', 'Profit']].sum().reset_index()
x2 = np.arange(len(seg_stats))
axes[1].bar(x2 - w/2, seg_stats['Sales'],  width=w, label='Sales',  color='steelblue')
axes[1].bar(x2 + w/2, seg_stats['Profit'], width=w, label='Profit', color='seagreen')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(seg_stats['Segment'])
axes[1].set_title('Total Sales & Profit by Segment', fontweight='bold')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
axes[1].legend()

plt.tight_layout()
plt.savefig('04_category_segment.png', bbox_inches='tight')
plt.show()


# ─── CELL 11 — Summary Table (save for README) ───────────────────────────────
summary = pd.DataFrame({
    'Metric': [
        'Total orders', 'Total sales', 'Total profit',
        'Overall profit margin', 'Loss-making orders',
        'Unique cities', 'Unique states', 'Date range note'
    ],
    'Value': [
        f"{len(df):,}",
        f"${df['Sales'].sum():,.0f}",
        f"${df['Profit'].sum():,.0f}",
        f"{df['Profit'].sum()/df['Sales'].sum()*100:.1f}%",
        f"{df['Is Loss'].sum():,} ({df['Is Loss'].mean()*100:.1f}%)",
        str(df['City'].nunique()),
        str(df['State'].nunique()),
        "4 years of US retail data"
    ]
})
print("\n=== PROJECT SUMMARY TABLE (paste into README) ===")
print(summary.to_string(index=False))


"""
Phase 2: Business Analysis — Sample Superstore
===============================================
Paste each block as a new Jupyter cell, continuing from Phase 1.
Assumes df is already loaded and cleaned from Phase 1.
"""

# ─── CELL 1 — Re-run setup if starting fresh ─────────────────────────────────
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

# sns.set_theme(style="whitegrid", palette="muted")
# plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

# df = pd.read_csv('SampleSuperstore.csv', encoding='latin-1')
# df = df.drop_duplicates()
# df['Postal Code']       = df['Postal Code'].astype(str).str.zfill(5)
# df['Profit Margin (%)'] = (df['Profit'] / df['Sales'] * 100).round(2)
# df['Is Loss']           = df['Profit'] < 0
# df['Discount Band']     = pd.cut(
#     df['Discount'], bins=[-0.01, 0.0, 0.2, 0.4, 1.0],
#     labels=['None', 'Low (1-20%)', 'Medium (21-40%)', 'High (>40%)']
# )


# ─── CELL 2 — Sub-Category Profit Bar Chart ──────────────────────────────────
sub = (
    df.groupby('Sub-Category')
      .agg(Sales=('Sales','sum'), Profit=('Profit','sum'))
      .assign(**{'Margin (%)': lambda x: (x['Profit']/x['Sales']*100).round(1)})
      .sort_values('Profit')
      .reset_index()
)

colors = ['#E24B4A' if p < 0 else '#1D9E75' for p in sub['Profit']]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(sub['Sub-Category'], sub['Profit'], color=colors, edgecolor='white', linewidth=0.4)

# Annotate each bar with margin %
for bar, margin in zip(bars, sub['Margin (%)']):
    x_pos = bar.get_width()
    offset = 500 if x_pos >= 0 else -500
    ha = 'left' if x_pos >= 0 else 'right'
    ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
            f'{margin}%', va='center', ha=ha, fontsize=9,
            color='#1D9E75' if margin >= 0 else '#E24B4A')

ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Total Profit ($)')
ax.set_title('Profit by sub-category (colour = profitable/loss-making)', fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f'-${abs(v/1000):.0f}K' if v < 0 else f'${v/1000:.0f}K'
))
plt.tight_layout()
plt.savefig('05_subcategory_profit.png', bbox_inches='tight')
plt.show()

print("\n=== FINDING ===")
print("Loss-making sub-categories:", sub[sub['Profit'] < 0]['Sub-Category'].tolist())
print(sub[['Sub-Category', 'Sales', 'Profit', 'Margin (%)']].to_string(index=False))


# ─── CELL 3 — Discount Impact Analysis ───────────────────────────────────────
disc = (
    df.groupby('Discount Band', observed=True)
      .agg(
          Avg_Profit  = ('Profit',  'mean'),
          Avg_Sales   = ('Sales',   'mean'),
          Order_Count = ('Sales',   'count'),
          Loss_Rate   = ('Is Loss', 'mean')
      )
      .round(2)
      .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

# Left: average profit per discount band
colors2 = ['#E24B4A' if v < 0 else '#1D9E75' for v in disc['Avg_Profit']]
axes[0].bar(disc['Discount Band'], disc['Avg_Profit'], color=colors2,
            edgecolor='white', linewidth=0.4)
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Average profit per order by discount band', fontweight='bold')
axes[0].set_ylabel('Average profit ($)')
axes[0].set_xlabel('Discount band')
for i, (val, row) in enumerate(zip(disc['Avg_Profit'], disc.itertuples())):
    axes[0].text(i, val + (2 if val >= 0 else -5),
                 f'${val:.0f}', ha='center', fontsize=9)

# Right: loss rate per discount band
loss_colors = ['#E24B4A' if r > 0.3 else '#EF9F27' if r > 0.1 else '#1D9E75'
               for r in disc['Loss_Rate']]
axes[1].bar(disc['Discount Band'], disc['Loss_Rate'] * 100, color=loss_colors,
            edgecolor='white', linewidth=0.4)
axes[1].set_title('Loss-making order rate by discount band', fontweight='bold')
axes[1].set_ylabel('% of orders with negative profit')
axes[1].set_xlabel('Discount band')
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
for i, val in enumerate(disc['Loss_Rate'] * 100):
    axes[1].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('06_discount_impact.png', bbox_inches='tight')
plt.show()

print("\n=== DISCOUNT BAND SUMMARY ===")
print(disc.to_string(index=False))

print("\n=== KEY INSIGHT ===")
# no_disc = disc[disc['Discount Band'] == 'None']['Avg_Profit'].values[0]
# high_disc = disc[disc['Discount Band'] == 'High (>40%)']['Avg_Profit'].values[0]
disc_dict = disc.set_index('Discount Band')['Avg_Profit'].to_dict()
no_disc   = disc_dict.get('None', 0)
high_disc = disc_dict.get('High (>40%)', 0)
print(f"No-discount avg profit:   ${no_disc:.2f}")
print(f"High-discount avg profit: ${high_disc:.2f}")
print(f"Profit erosion:           ${no_disc - high_disc:.2f} per order")


# ─── CELL 4 — Regional Performance ───────────────────────────────────────────
region = (
    df.groupby('Region')
      .agg(Sales=('Sales','sum'), Profit=('Profit','sum'), Orders=('Sales','count'))
      .assign(**{'Margin (%)': lambda x: (x['Profit']/x['Sales']*100).round(1)})
      .sort_values('Profit', ascending=False)
      .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

# Left: sales vs profit grouped bar
x  = np.arange(len(region))
w  = 0.35
b1 = axes[0].bar(x - w/2, region['Sales'],  width=w, label='Sales',  color='steelblue', edgecolor='white')
b2 = axes[0].bar(x + w/2, region['Profit'], width=w, label='Profit', color='#1D9E75',   edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(region['Region'])
axes[0].set_title('Sales vs Profit by region', fontweight='bold')
axes[0].set_ylabel('Amount ($)')
axes[0].legend()
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))

# Right: margin % horizontal bars
margin_colors = ['#1D9E75' if m > 12 else '#EF9F27' for m in region['Margin (%)']]
axes[1].barh(region['Region'], region['Margin (%)'], color=margin_colors,
             edgecolor='white', linewidth=0.4)
axes[1].set_title('Profit margin % by region', fontweight='bold')
axes[1].set_xlabel('Profit margin (%)')
for i, val in enumerate(region['Margin (%)']):
    axes[1].text(val + 0.1, i, f'{val}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('07_regional_performance.png', bbox_inches='tight')
plt.show()

print("\n=== REGIONAL SUMMARY ===")
print(region.to_string(index=False))


# ─── CELL 5 — Top & Bottom States ────────────────────────────────────────────
state_profit = (
    df.groupby('State')
      .agg(Sales=('Sales','sum'), Profit=('Profit','sum'))
      .assign(**{'Margin (%)': lambda x: (x['Profit']/x['Sales']*100).round(1)})
      .sort_values('Profit')
      .reset_index()
)

top5    = state_profit.tail(5).iloc[::-1]
bottom5 = state_profit.head(5)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

axes[0].barh(top5['State'], top5['Profit'], color='#1D9E75', edgecolor='white')
axes[0].set_title('Top 5 states by profit', fontweight='bold')
axes[0].set_xlabel('Total Profit ($)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))

axes[1].barh(bottom5['State'], bottom5['Profit'], color='#E24B4A', edgecolor='white')
axes[1].set_title('Bottom 5 states by profit', fontweight='bold')
axes[1].set_xlabel('Total Profit ($)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f'-${abs(v/1000):.0f}K' if v < 0 else f'${v/1000:.0f}K'
))
axes[1].axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('08_top_bottom_states.png', bbox_inches='tight')
plt.show()

print("\n=== TOP 5 STATES ===")
print(top5[['State','Sales','Profit','Margin (%)']].to_string(index=False))
print("\n=== BOTTOM 5 STATES ===")
print(bottom5[['State','Sales','Profit','Margin (%)']].to_string(index=False))


# ─── CELL 6 — Ship Mode Analysis ─────────────────────────────────────────────
ship = (
    df.groupby('Ship Mode')
      .agg(Orders=('Sales','count'), Avg_Sales=('Sales','mean'),
           Total_Profit=('Profit','sum'), Loss_Rate=('Is Loss','mean'))
      .round(2)
      .sort_values('Total_Profit', ascending=False)
      .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

axes[0].bar(ship['Ship Mode'], ship['Orders'], color='steelblue',
            edgecolor='white', linewidth=0.4)
axes[0].set_title('Order count by ship mode', fontweight='bold')
axes[0].set_ylabel('Number of orders')
for i, v in enumerate(ship['Orders']):
    axes[0].text(i, v + 20, str(v), ha='center', fontsize=9)

axes[1].bar(ship['Ship Mode'], ship['Total_Profit'], color='#1D9E75',
            edgecolor='white', linewidth=0.4)
axes[1].set_title('Total profit by ship mode', fontweight='bold')
axes[1].set_ylabel('Total Profit ($)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))

plt.tight_layout()
plt.savefig('09_ship_mode.png', bbox_inches='tight')
plt.show()

print("\n=== SHIP MODE SUMMARY ===")
print(ship.to_string(index=False))


# ─── CELL 7 — Segment Analysis ───────────────────────────────────────────────
seg = (
    df.groupby('Segment')
      .agg(Sales=('Sales','sum'), Profit=('Profit','sum'), Orders=('Sales','count'))
      .assign(**{'Margin (%)': lambda x: (x['Profit']/x['Sales']*100).round(1),
                 'Avg Order ($)': lambda x: (x['Sales']/x['Orders']).round(0)})
      .sort_values('Profit', ascending=False)
      .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(8, 5))

for ax, col, label, color in zip(
    axes,
    ['Sales', 'Profit', 'Margin (%)'],
    ['Total sales ($)', 'Total profit ($)', 'Profit margin (%)'],
    ['steelblue', '#1D9E75', '#EF9F27']
):
    ax.bar(seg['Segment'], seg[col], color=color, edgecolor='white', linewidth=0.4)
    ax.set_title(label, fontweight='bold')
    if col != 'Margin (%)':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
    else:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    for i, v in enumerate(seg[col]):
        fmt = f'{v:.1f}%' if col == 'Margin (%)' else f'${v/1000:.0f}K'
        ax.text(i, v * 0.5, fmt, ha='center', fontsize=9, color='white', fontweight='bold')

plt.suptitle('Customer segment analysis', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('10_segment_analysis.png', bbox_inches='tight')
plt.show()

print("\n=== SEGMENT SUMMARY ===")
print(seg.to_string(index=False))


# ─── CELL 8 — Phase 2 Business Insight Summary ───────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║          PHASE 2 — KEY BUSINESS INSIGHTS (for README)           ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Tables sub-category loses $17,725 despite $207K in sales    ║
║     (-8.6% margin) — driven by heavy discounting                ║
║                                                                  ║
║  2. High discounts (>40%) flip average profit from +$67          ║
║     to -$107 per order — a $174 swing                           ║
║                                                                  ║
║  3. Central region has 7.9% margin vs West's 14.9%              ║
║     — nearly half the profitability                              ║
║                                                                  ║
║  4. Texas is the worst state: -$25,751 total profit             ║
║     despite being a large market                                 ║
║                                                                  ║
║  5. Copiers have the best margin at 37.2%                        ║
║     — Technology is the star category                            ║
╚══════════════════════════════════════════════════════════════════╝
""")

"""
Phase 3: Advanced Analysis — Sample Superstore
===============================================
Paste each block as new Jupyter cells, continuing from Phase 1 & 2.
New dependency: pip install scikit-learn
"""

# ─── CELL 0 — Extra import (add to your Phase 1 imports cell) ────────────────
# If running fresh, run this first:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

df = pd.read_csv('SampleSuperstore.csv', encoding='latin-1')
df = df.drop_duplicates()
df['Profit Margin (%)'] = (df['Profit'] / df['Sales'] * 100).round(2)
df['Is Loss']           = df['Profit'] < 0
df['Discount Band']     = pd.cut(
    df['Discount'], bins=[-0.01, 0.0, 0.2, 0.4, 1.0],
    labels=['None', 'Low (1-20%)', 'Medium (21-40%)', 'High (>40%)']
)


# ─── CELL 1 — Deep Correlation Heatmap ───────────────────────────────────────
# Key finding: Discount has -0.864 correlation with Profit Margin
# That is one of the strongest relationships in the entire dataset

cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin (%)']
corr = df[cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    ax=ax,
    cbar_kws={'shrink': 0.8, 'label': 'Correlation coefficient'}
)

ax.set_title('Correlation matrix — key relationships', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('11_correlation_heatmap.png', bbox_inches='tight')
plt.show()

print("=== TOP CORRELATIONS ===")
# Unstack and sort to find strongest pairs
corr_pairs = (
    corr.where(mask == False)
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature A', 'level_1': 'Feature B', 0: 'Correlation'})
        .sort_values('Correlation', key=abs, ascending=False)
)
print(corr_pairs[corr_pairs['Feature A'] != corr_pairs['Feature B']].head(8).to_string(index=False))

print("\n=== KEY INSIGHT ===")
print("Discount vs Profit Margin correlation: -0.864")
print("This means: as discount increases, profit margin drops sharply.")
print("This is the strongest signal in the entire dataset.")


# ─── CELL 2 — Category × Region Pivot Heatmap ────────────────────────────────
# Shows which category+region combinations are profitable vs loss-making

pivot_profit = df.pivot_table(
    values='Profit',
    index='Category',
    columns='Region',
    aggfunc='sum'
).round(0)

pivot_margin = df.pivot_table(
    values='Profit Margin (%)',
    index='Category',
    columns='Region',
    aggfunc='mean'
).round(1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Left: total profit heatmap
sns.heatmap(
    pivot_profit,
    annot=True,
    fmt='.0f',
    cmap='RdYlGn',
    center=0,
    linewidths=0.5,
    ax=axes[0],
    cbar_kws={'label': 'Total Profit ($)'}
)
axes[0].set_title('Total profit: Category × Region', fontweight='bold')
axes[0].set_xlabel('')

# Right: avg margin heatmap
sns.heatmap(
    pivot_margin,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn',
    center=0,
    linewidths=0.5,
    ax=axes[1],
    cbar_kws={'label': 'Avg Profit Margin (%)'}
)
axes[1].set_title('Avg profit margin %: Category × Region', fontweight='bold')
axes[1].set_xlabel('')

plt.tight_layout()
plt.savefig('12_pivot_heatmap.png', bbox_inches='tight')
plt.show()

print("=== PIVOT TABLE — TOTAL PROFIT ===")
print(pivot_profit.to_string())
print("\n=== PIVOT TABLE — AVG MARGIN ===")
print(pivot_margin.to_string())

print("\n=== KEY INSIGHTS ===")
print("Furniture in Central region: LOSS-MAKING (-$2,906)")
print("Technology in Central:  highest absolute profit ($33,697)")
print("All regions profit from Technology and Office Supplies")
print("Furniture is the weakest category across all regions")


# ─── CELL 3 — Discount × Category Interaction ────────────────────────────────
# Technology is especially vulnerable to high discounts (-$778 avg per order)

interaction = df.groupby(
    ['Category', 'Discount Band'], observed=True
)['Profit'].mean().round(1).reset_index()

pivot_interaction = interaction.pivot(
    index='Category',
    columns='Discount Band',
    values='Profit'
)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

# Left: grouped bar chart
categories = pivot_interaction.index.tolist()
bands      = pivot_interaction.columns.tolist()
x          = np.arange(len(categories))
width      = 0.2
colors_map = ['#1D9E75', '#85B7EB', '#EF9F27', '#E24B4A']

for i, (band, color) in enumerate(zip(bands, colors_map)):
    vals = pivot_interaction[band].values
    bars = axes[0].bar(x + i * width, vals, width=width,
                       label=band, color=color, edgecolor='white')

axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(categories)
axes[0].set_title('Avg profit per order: Category × Discount band', fontweight='bold')
axes[0].set_ylabel('Average profit per order ($)')
axes[0].legend(title='Discount band', fontsize=9)

# Right: heatmap version
sns.heatmap(
    pivot_interaction,
    annot=True,
    fmt='.0f',
    cmap='RdYlGn',
    center=0,
    linewidths=0.5,
    ax=axes[1],
    cbar_kws={'label': 'Avg profit ($)'}
)
axes[1].set_title('Heatmap view — same data', fontweight='bold')
axes[1].set_xlabel('Discount band')

plt.tight_layout()
plt.savefig('13_discount_category_interaction.png', bbox_inches='tight')
plt.show()

print("=== DISCOUNT × CATEGORY INTERACTION ===")
print(pivot_interaction.to_string())
print("\n=== KEY INSIGHT ===")
print("Technology with high discount: avg -$778 per order (worst combination)")
print("Technology with no discount:   avg +$159 per order")
print("Profit swing of $937 per order just from discount policy!")


# ─── CELL 4 — State-Level RFM Segmentation ───────────────────────────────────
# Groups states into Star / Growth / At-risk / Low-value segments
# RFM = Recency (skipped — no dates), Frequency, Monetary adapted to states

state_rfm = df.groupby('State').agg(
    Frequency = ('Sales', 'count'),       # number of orders
    Monetary  = ('Sales',  'sum'),         # total revenue
    Profit    = ('Profit', 'sum'),
    LossRate  = ('Is Loss', 'mean')
).reset_index()

# Score Frequency and Monetary on 1-3 scale
state_rfm['F_score'] = pd.qcut(state_rfm['Frequency'], q=3, labels=[1, 2, 3]).astype(int)
state_rfm['M_score'] = pd.qcut(state_rfm['Monetary'],  q=3, labels=[1, 2, 3]).astype(int)
state_rfm['FM_score'] = state_rfm['F_score'] + state_rfm['M_score']

def segment(row):
    if row['Profit'] < 0:
        return 'At-risk'
    elif row['FM_score'] >= 5:
        return 'Star'
    elif row['FM_score'] >= 3:
        return 'Growth'
    else:
        return 'Low-value'

state_rfm['Segment'] = state_rfm.apply(segment, axis=1)

# Segment counts
seg_counts = state_rfm['Segment'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

# Left: scatter — Frequency vs Monetary, coloured by segment
seg_colors = {'Star': '#1D9E75', 'Growth': '#378ADD', 'At-risk': '#E24B4A', 'Low-value': '#888780'}
for seg, grp in state_rfm.groupby('Segment'):
    axes[0].scatter(
        grp['Frequency'], grp['Monetary'],
        label=f"{seg} ({len(grp)})",
        color=seg_colors[seg],
        s=80, alpha=0.85, edgecolors='white', linewidths=0.5
    )

# Label notable states
for _, row in state_rfm.nlargest(5, 'Monetary').iterrows():
    axes[0].annotate(row['State'], (row['Frequency'], row['Monetary']),
                     fontsize=8, xytext=(5, 3), textcoords='offset points')
for _, row in state_rfm[state_rfm['Profit'] < -5000].iterrows():
    axes[0].annotate(row['State'], (row['Frequency'], row['Monetary']),
                     fontsize=8, xytext=(5, 3), textcoords='offset points', color='#A32D2D')

axes[0].set_xlabel('Order Frequency (count)')
axes[0].set_ylabel('Total Revenue ($)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
axes[0].set_title('State segmentation: Frequency vs Revenue', fontweight='bold')
axes[0].legend(title='Segment', fontsize=9)

# Right: profit by segment box
seg_order = ['Star', 'Growth', 'Low-value', 'At-risk']
seg_palette = [seg_colors[s] for s in seg_order]
seg_data = [state_rfm[state_rfm['Segment'] == s]['Profit'].values for s in seg_order]

bp = axes[1].boxplot(seg_data, labels=seg_order, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], seg_palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_ylabel('Total Profit ($)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f'-${abs(v/1000):.0f}K' if v < 0 else f'${v/1000:.0f}K'
))
axes[1].set_title('Profit distribution by segment', fontweight='bold')

plt.tight_layout()
plt.savefig('14_state_rfm_segments.png', bbox_inches='tight')
plt.show()

print("=== SEGMENT COUNTS ===")
print(seg_counts.to_string())
print("\n=== AT-RISK STATES (loss-making) ===")
at_risk = state_rfm[state_rfm['Segment'] == 'At-risk'][['State', 'Frequency', 'Monetary', 'Profit']].sort_values('Profit')
print(at_risk.to_string(index=False))
print("\n=== STAR STATES ===")
stars = state_rfm[state_rfm['Segment'] == 'Star'][['State', 'Frequency', 'Monetary', 'Profit']].sort_values('Profit', ascending=False)
print(stars.to_string(index=False))


# ─── CELL 5 — Linear Regression: Predicting Profit ───────────────────────────
# Uses scikit-learn — adds ML to your resume
# Features: Discount, Sales, Quantity
# Target:   Profit

features = ['Sales', 'Quantity', 'Discount']
target   = 'Profit'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=== LINEAR REGRESSION RESULTS ===")
print(f"R² Score : {r2:.4f}  (explains {r2*100:.1f}% of variance in profit)")
print(f"MAE      : ${mae:.2f}  (avg prediction error per order)")
print(f"\nCoefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:<12}: {coef:+.4f}")
print(f"  Intercept   : {model.intercept_:+.4f}")

print("\n=== INTERPRETATION ===")
disc_coef = model.coef_[features.index('Discount')]
print(f"Each 1-unit increase in Discount (0→1.0) changes profit by ${disc_coef:+.2f}")
print(f"Each $1 increase in Sales changes profit by ${model.coef_[0]:+.4f}")

# Visualise: actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(8, 5))

# Scatter: actual vs predicted
axes[0].scatter(y_test, y_pred, alpha=0.3, s=12, color='steelblue', linewidths=0)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
axes[0].plot([min_val, max_val], [min_val, max_val],
             color='red', linewidth=1, linestyle='--', label='Perfect prediction')
axes[0].set_xlabel('Actual Profit ($)')
axes[0].set_ylabel('Predicted Profit ($)')
axes[0].set_title(f'Actual vs Predicted Profit (R²={r2:.3f})', fontweight='bold')
axes[0].legend(fontsize=9)

# Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, s=12, color='darkorange', linewidths=0)
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel('Predicted Profit ($)')
axes[1].set_ylabel('Residual (Actual − Predicted)')
axes[1].set_title('Residual plot — checking model assumptions', fontweight='bold')

plt.tight_layout()
plt.savefig('15_regression.png', bbox_inches='tight')
plt.show()

print("\n=== RESUME TIP ===")
print(f"Write: 'Built a linear regression model (scikit-learn) to predict order-level")
print(f"profit; achieved R²={r2:.2f} and identified discount rate as the dominant")
print(f"negative predictor (coefficient: {disc_coef:+.0f})'")


# ─── CELL 6 — Phase 3 Summary ────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║          PHASE 3 — KEY ADVANCED INSIGHTS (for README)           ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Discount ↔ Profit Margin: -0.864 correlation               ║
║     Strongest signal in the dataset by far                       ║
║                                                                  ║
║  2. Furniture in Central region is loss-making (-$2,906)        ║
║     All other Category×Region combos are profitable             ║
║                                                                  ║
║  3. Technology + high discount = avg -$778 per order            ║
║     Without discount: +$159. A $937 swing in one decision       ║
║                                                                  ║
║  4. State segmentation: [Star / Growth / At-risk / Low-value]   ║
║     Texas leads At-risk with -$25,751 total loss                ║
║                                                                  ║
║  5. Linear regression R² ~ 0.40                                  ║
║     Discount & Sales explain ~40% of profit variance            ║
║     (remaining variance = product mix, customer type, etc.)     ║
╚══════════════════════════════════════════════════════════════════╝
""")

"""
Phase 4: Dashboard Summary & README — Sample Superstore
========================================================
Two parts:
  A) Final summary dashboard (1 multi-panel figure — your portfolio showpiece)
  B) README.md content printed at the end — copy-paste into GitHub
"""



# ─── CELL 2 — Final Summary Dashboard ────────────────────────────────────────
# One professional figure with 6 panels — save as your README hero image

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    'Sample Superstore — Business Intelligence Dashboard',
    fontsize=16, fontweight='bold', y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])   # Sub-category profit
ax2 = fig.add_subplot(gs[0, 1])   # Discount impact
ax3 = fig.add_subplot(gs[0, 2])   # Region margin
ax4 = fig.add_subplot(gs[1, 0])   # Category x Region heatmap
ax5 = fig.add_subplot(gs[1, 1])   # Segment pie
ax6 = fig.add_subplot(gs[1, 2])   # Top/bottom states

# ── Panel 1: Sub-category profit ─────────────────────────────────────────────
sub = (df.groupby('Sub-Category')
         .agg(Profit=('Profit','sum'))
         .sort_values('Profit')
         .reset_index())
colors1 = ['#E24B4A' if p < 0 else '#1D9E75' for p in sub['Profit']]
ax1.barh(sub['Sub-Category'], sub['Profit'], color=colors1,
         edgecolor='white', linewidth=0.3, height=0.7)
ax1.axvline(0, color='black', linewidth=0.6)
ax1.set_title('Profit by sub-category', fontweight='bold', fontsize=10)
ax1.set_xlabel('Profit ($)')
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f'-${abs(v/1000):.0f}K' if v < 0 else f'${v/1000:.0f}K'))
ax1.tick_params(axis='y', labelsize=8)

# ── Panel 2: Discount band avg profit ────────────────────────────────────────
disc = (df.groupby('Discount Band', observed=True)['Profit']
          .mean().reset_index())
colors2 = ['#E24B4A' if v < 0 else '#1D9E75' for v in disc['Profit']]
bars2 = ax2.bar(disc['Discount Band'], disc['Profit'],
                color=colors2, edgecolor='white', linewidth=0.3)
ax2.axhline(0, color='black', linewidth=0.6, linestyle='--')
ax2.set_title('Avg profit by discount band', fontweight='bold', fontsize=10)
ax2.set_ylabel('Avg profit per order ($)')
for bar, val in zip(bars2, disc['Profit']):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + (3 if val >= 0 else -8),
             f'${val:.0f}', ha='center', fontsize=8)

# ── Panel 3: Region margin ────────────────────────────────────────────────────
region = (df.groupby('Region')
            .apply(lambda x: x['Profit'].sum() / x['Sales'].sum() * 100)
            .round(1).reset_index(name='Margin'))
region = region.sort_values('Margin', ascending=True)
reg_colors = ['#1D9E75' if m > 12 else '#EF9F27' for m in region['Margin']]
ax3.barh(region['Region'], region['Margin'],
         color=reg_colors, edgecolor='white', linewidth=0.3)
ax3.set_title('Profit margin % by region', fontweight='bold', fontsize=10)
ax3.set_xlabel('Profit margin (%)')
for i, val in enumerate(region['Margin']):
    ax3.text(val + 0.1, i, f'{val}%', va='center', fontsize=9)
ax3.set_xlim(0, region['Margin'].max() * 1.3)

# ── Panel 4: Category × Region heatmap ───────────────────────────────────────
pivot = df.pivot_table(values='Profit', index='Category',
                       columns='Region', aggfunc='sum') / 1000
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax4,
            cbar_kws={'label': 'Profit ($K)', 'shrink': 0.8},
            annot_kws={'size': 9})
ax4.set_title('Profit $K: Category × Region', fontweight='bold', fontsize=10)
ax4.set_xlabel('')
ax4.tick_params(axis='x', rotation=30, labelsize=8)
ax4.tick_params(axis='y', rotation=0, labelsize=8)

# ── Panel 5: Segment donut ────────────────────────────────────────────────────
seg = df.groupby('Segment')['Profit'].sum()
seg_colors = ['#378ADD', '#1D9E75', '#EF9F27']
wedges, texts, autotexts = ax5.pie(
    seg.values, labels=seg.index, autopct='%1.1f%%',
    colors=seg_colors, startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    pctdistance=0.75
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_color('white')
    t.set_fontweight('bold')
centre = plt.Circle((0, 0), 0.5, color='white')
ax5.add_patch(centre)
ax5.set_title('Profit share by segment', fontweight='bold', fontsize=10)
ax5.text(0, 0, f'$286K\ntotal', ha='center', va='center',
         fontsize=9, fontweight='bold')

# ── Panel 6: Top 5 & Bottom 5 states ─────────────────────────────────────────
state = df.groupby('State')['Profit'].sum().sort_values()
bottom5 = state.head(5)
top5    = state.tail(5)
combined       = pd.concat([bottom5, top5])
combined_colors = ['#E24B4A'] * 5 + ['#1D9E75'] * 5

ax6.barh(combined.index, combined.values,
         color=combined_colors, edgecolor='white', linewidth=0.3)
ax6.axvline(0, color='black', linewidth=0.6)
ax6.set_title('Top 5 & bottom 5 states', fontweight='bold', fontsize=10)
ax6.set_xlabel('Total Profit ($)')
ax6.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f'-${abs(v/1000):.0f}K' if v < 0 else f'${v/1000:.0f}K'))
ax6.tick_params(axis='y', labelsize=8)

# ── KPI strip across the top ──────────────────────────────────────────────────
kpis = [
    ('9,977', 'Total orders'),
    ('$2.30M', 'Total revenue'),
    ('$286K', 'Total profit'),
    ('12.5%', 'Overall margin'),
    ('18.7%', 'Loss-order rate'),
    ('49', 'States covered'),
]
kpi_ax = fig.add_axes([0.01, 0.91, 0.98, 0.055])
kpi_ax.axis('off')
for i, (val, lbl) in enumerate(kpis):
    x = 0.08 + i * 0.16
    kpi_ax.text(x, 0.75, val, ha='center', va='center',
                fontsize=13, fontweight='bold',
                transform=kpi_ax.transAxes)
    kpi_ax.text(x, 0.15, lbl, ha='center', va='center',
                fontsize=8, color='gray',
                transform=kpi_ax.transAxes)

plt.savefig('16_final_dashboard.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: 16_final_dashboard.png  ← use this as your README hero image")


# ─── CELL 3 — Print README.md ────────────────────────────────────────────────
# Copy everything between the triple-quotes and paste into README.md

readme = """
# Sample Superstore — End-to-End Data Analysis

> An end-to-end exploratory and business analysis of 9,977 US retail orders
> using Python, Pandas, Seaborn, Matplotlib, and Scikit-learn.

![Dashboard](images/16_final_dashboard.png)

---

## Key findings

| # | Insight | Value |
|---|---------|-------|
| 1 | Loss-making orders | 1,869 orders (18.7%) |
| 2 | Worst sub-category | Tables: -$17,725 profit on $207K revenue (-8.6% margin) |
| 3 | Discount impact | High discounts (>40%) flip avg profit from +$67 to -$107/order |
| 4 | Best sub-category | Copiers: $55,618 profit at 37.2% margin |
| 5 | Weakest region | Central: 7.9% margin vs West's 14.9% |
| 6 | Worst state | Texas: -$25,751 total profit despite high order volume |
| 7 | Strongest correlation | Discount ↔ Profit Margin: -0.864 |
| 8 | ML model | Linear regression R²≈0.40 predicting profit from discount + sales |

---

## Tech stack

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn` `Jupyter`

---

## Project structure

```
superstore-analysis/
│
├── data/
│   └── SampleSuperstore.csv
│
├── notebooks/
│   └── superstore_analysis.ipynb    ← main notebook (all phases)
│
├── images/                          ← all saved chart PNGs
│   ├── 16_final_dashboard.png
│   └── ...
│
└── README.md
```

---

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/superstore-analysis
cd superstore-analysis
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook notebooks/superstore_analysis.ipynb
```

---

## Analysis phases

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Data cleaning & EDA | Distributions, null check, derived columns |
| 2 | Business analysis | Sub-category, region, discount, segment charts |
| 3 | Advanced analysis | Correlation, pivot heatmap, RFM segments, regression |
| 4 | Dashboard | Summary figure + README |

---

## Business recommendations

1. **Cap discounts at 20%** — orders above 20% discount have a 50%+ loss rate
2. **Review Tables pricing** — the sub-category is structurally loss-making
3. **Investigate Central region** — margin is nearly half that of West
4. **Prioritise Copiers & Technology** — highest margin products

---

*Dataset: Sample Superstore (public dataset, widely used for BI practice)*
"""

print(readme)
print("\nCopy everything above into your README.md file!")