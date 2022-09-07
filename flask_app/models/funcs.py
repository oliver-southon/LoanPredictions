import numpy as np
import seaborn as sns

import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def example_run():
    sns.set_theme(style="whitegrid")
    penguins = sns.load_dataset("penguins")

    g = sns.catplot(
        data=penguins,
        kind='bar',
        x="species", y="body_mass_g", hue="sex",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Body mass (g)")
    g.legend.set_title("")

    img = io.BytesIO()

    g.savefig(img, format='png')
    plt.close()
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
