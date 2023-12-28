import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# Generate some sample data
x = [0.16, 1.4, 2.8, 6.9, 12]
# wiki_ll = [.505, .509, .515, .515, .517]
# wiki_lira = [.512, .515, .521, .520, .521]

# pubmed_central_ll = [.502, .502, .504, .509, .512]
# pubmed_central_lira = [.510, .512, .513, .518, .520]

# github_ll = [.519, .521, .527, .521, .524]
# github_lira = [.513, .511, .516, .512, .513]

# dm_math_ll = [.488, .490, .489, .489, .490]
# dm_math_lira = [.491, .492, .492, .492, .494]

wiki_ll = [0.01100, 0.01300, 0.01600, 0.01100, 0.01500]
wiki_lira = [0.00400, 0.01700, 0.00700, 0.01400, 0.01700]

pubmed_central_ll = [0.01200, 0.01900, 0.01800, 0.01700, 0.01700]
pubmed_central_lira = [0.01600, 0.00700, 0.01600, 0.01300, 0.01800]

github_ll = [0.01300, 0.01200, 0.01400, 0.01500, 0.01900]
github_lira = [0.00500, 0.00700, 0.01200, 0.01000, 0.01200]

dm_math_ll = [0.01300, 0.01200, 0.01000, 0.01300, 0.01300]
dm_math_lira = [0.00200, 0.00600, 0.00500, 0.00600, 0.00600]

# ArXiv
# LL .514 .520 .523 .527 .532
# LiRA .527 .531 .533 .537 .543


# HackerNews
# LL .492 .503 .509 .513 .518
# LiRA .497 .506 .518 .521 .529

# Pile-CC
# LL .492 .496 .497 .504 .511
# LiRA .504 .509 .509 .515 .519


# Set Seaborn style
sns.set(style="whitegrid")

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data with Seaborn
sns.lineplot(x=x, y=wiki_ll, color="blue", marker='o', linestyle="solid", label='Wikipedia')
sns.lineplot(x=x, y=wiki_lira, color="blue", marker='o', linestyle="dashed")
sns.lineplot(x=x, y=pubmed_central_ll, marker='o', linestyle="solid", color="green", label='Pubmed Central')
sns.lineplot(x=x, y=pubmed_central_lira, marker='o', linestyle="dashed", color="green")
sns.lineplot(x=x, y=github_ll, marker='o', linestyle="solid", color="orange", label='Github')
sns.lineplot(x=x, y=github_lira, marker='o', linestyle="dashed", color="orange")
sns.lineplot(x=x, y=dm_math_ll, marker='o', linestyle="solid", color="purple", label='DM Math')
sns.lineplot(x=x, y=dm_math_lira, marker='o', linestyle="dashed", color="purple")
# ax.axhline(0.5)
# ax.text(11.2,.501,'Random', fontsize='x-small')
# Add a title and labels
# ax.set_title('Research Paper Figure')
x_ax = ax.set_xlabel('Model Size (B)')
y_ax = ax.set_ylabel('TPR@1%FPR')
lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left')


# Save the figure
fig.savefig('model_size_tpr@lowfpr.png', bbox_extra_artists=(lgd,x_ax, y_ax), bbox_inches='tight')