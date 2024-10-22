import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itt
import re

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, rgb2hex
from pathlib import Path
import torch

text_configuration = {
    'usetex': True,
    'axes_labels_size': 12,
    'ticks_labels_size': 10,
    'legend_font_size': 10
}

plt.rc('text', usetex=True)
plt.rc('axes', titlesize=text_configuration['axes_labels_size'])
plt.rc('axes', labelsize=text_configuration['axes_labels_size'])
plt.rc('xtick', labelsize=text_configuration['ticks_labels_size'])
plt.rc('ytick', labelsize=text_configuration['ticks_labels_size'])
plt.rc('legend', fontsize=text_configuration['legend_font_size'])
plt.rc('axes', linewidth=0.2)

def formatter(text):
    '''Extract number from `text = "$\\mathdefault{2}$"` and format to 10^2.'''
    #Regular expression pattern to extract the number
    pattern = r'\d+'
    # Using re.findall() to extract numbers from the string
    number = re.findall(pattern, text)
    new_text = r"$10^{}$".format(number[0])
    return new_text

def next_color_hex(cmap_name, index=0, n_traces=3):
    """
    Generates the next color in the colormap sequence and returns its hexadecimal representation.

    Parameters:
        cmap_name (str): Name of the colormap to use.
        index (int, optional): Starting index for the color. Defaults to 0.
        n_traces (int, optional): Number of colors to generate from the colormap. Defaults to 3.

    Returns:
        str: Hexadecimal color code.
    """
    cmap = sns.color_palette(cmap_name, n_traces)
    color = cmap[index]
    index = (index + 1) % len(cmap)
    return rgb2hex(color)

class CornerPlot:
    """
    A class to create and manage a customizable corner plot using Matplotlib.
    The corner plot is a grid of scatter plots and histograms to visualize
    pairwise relationships and marginal distributions of a set of parameters.

    Parameters:
        colormap (str): Name of the colormap to use for traces.
        parameters (list): List of parameters to be plotted.
        labels (list, optional): List of labels for the parameters. Defaults to None, which uses default parameter names.
        n_ticks_per_axs (int, optional): Number of ticks per axis. Defaults to 4.
        figsize (tuple, optional): Size of the figure. Defaults to (6, 5).
    """
    def __init__(
        self, colormap, parameters, labels=None,
        n_ticks_per_axs=4, figsize=(6,5)
        ):
        """
        Constructor to initialize the CornerPlot object.

        Parameters:
            colormap (str): Name of the colormap to use for traces.
            parameters (list): List of parameters to be plotted.
            labels (list, optional): List of labels for the parameters. Defaults to None.
            n_ticks_per_axs (int, optional): Number of ticks per axis. Defaults to 4.
            figsize (tuple, optional): Size of the figure. Defaults to (6, 5).
        """

        self.color_counter = 0
        self.fig = None
        self.cmap = colormap
        self.n_ticks = n_ticks_per_axs
        self.legend_elements = []
        self.d = len(parameters)
        self.fig, self.axs = plt.subplots(
            self.d, self.d, figsize=figsize,
            )
        self.labels = labels
        if self.labels == None:
            self.labels = [r'$\theta_{}$'.format(i) for i in range(self.d)]
        self.initialise_fig()
        self.colors = []

    def add_trace(self, data, trace_name, alpha=0.2):
        """
        Adds a new trace to the corner plot.

        Parameters:
            data (pandas.DataFrame): DataFrame containing the data for the trace.
            trace_name (str): Name of the trace.
            alpha (float, optional): Transparency level of the trace. Defaults to 0.2.
        """
        column_ranges = np.ptp(data, axis=0)
        color = next_color_hex(
                    self.cmap, self.color_counter
                    )
        self.colors.append(color)

        self.legend_elements.append(
            Line2D(
                [0], [0], marker='o',  color=color,
               label=trace_name,linestyle='None')
            )
        data_ = data.to_numpy()
        # Plot scatter plots on the off-diagonal

        for i in range(self.d):
            for j in range(i+1, self.d):
                ax = self.axs[j, i]
                #xlim, ylim = ax.get_xlim(), ax.get_ylim()
                ax.scatter(
                    data_[:, i], data_[:, j],
                    s=1, alpha=alpha, c=color
                )

               # Adding custom text box for as helper
                custom_text = r"({},{})".format(i,j)  # Your custom text
                ax.text(
                    1.3, 1.3, custom_text, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.1, boxstyle='round')
                    )
                ax.set_rasterized(True)

        # Add marginal histograms
        for i in range(self.d):
            ax = self.axs[i, i]
            if i != self.d-1 :
                ax.set_xticklabels([])

            ax.hist(data_[:, i], alpha=0.2, color=color, histtype='stepfilled')
            ax.hist(data_[:, i], alpha=0.5, color=color, histtype='step')
            #ax.set_xscale('log')

            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(r'$\mathrm{Counts}$')

            custom_text = r"({},{})".format(i,j)  # Your custom text
            ax.text(
                1.3, 1.3, custom_text, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.1, boxstyle='round')
                )
        self.color_counter += 1

    def update_ticks(self):
        """
        Updates the tick labels for the plots.
        """
        for i in range(self.d):
            ax = self.axs[i,0]
            ax.set_yticklabels(ax.get_xticks(), rotation = 50)
        for j in range(self.d):
            ax = self.axs[-1,j]
            ax.set_xticklabels(ax.get_xticks(), rotation = 50)

    def clean(self):
        """
        Cleans helper elements to prepare the plot for saving.
        """        # Clean helper texts boxes
        for i,j in itt.product(range(self.d), range(self.d)):
            ax = self.axs[i,j]
            if len(ax.texts) != 0:

                #current_texts = [ax.texts[i].get_text() for i in range(len(ax.texts))]
                _ = [ax.texts[i].set_text('') for i in range(len(ax.texts))]

    def initialise_fig(self):
        """
        Initializes the figure by setting labels and configuring ticks.
        """        
        # Set labels
        for i in range(self.d):
            self.axs[-1,i].set_xlabel('{}'.format(self.labels[i]))
            for j in range(i+1, self.d):
                self.axs[j, 0].set_ylabel('{}'.format(self.labels[j]))

        # Ticks configuration
        for i in range(self.d):
            for j in range(self.d):
                ax = self.axs[i, j]
                ax.tick_params(axis='both', direction='in', labelrotation = 45)
                ax.grid(linestyle='--', linewidth=0.5 )
                ax.locator_params("both", nbins = self.n_ticks)

                if i == j:
                    ax.locator_params("both", nbins = self.n_ticks)
                    ax.tick_params(axis='y', direction='in', labelrotation = 0)
                    continue

                # Hide x labels and tick labels for top plots and y ticks for right plots.
                ax.label_outer()

        # Remove empty subplots
        for i in range(self.d):
            for j in range(i+1, self.d):
                self.axs[i, j].remove()

    def modify_single_ax_x(self, i, j):
        """
        Modifies the x-axis labels of a single subplot.

        Parameters:
            i (int): Row index of the subplot.
            j (int): Column index of the subplot.
        """
        ax = self.axs[j,i]
        print(ax)
        current_ticks = ax.get_xticklabels()
        current_labels = [
            formatter(
                current_ticks[i].get_text()
                ) for i in range(len(current_ticks))]
        ax.set_xticklabels(current_labels)

    def modify_single_ax_y(self, i, j):
        """
        Modifies the y-axis labels of a single subplot.

        Parameters:
            i (int): Row index of the subplot.
            j (int): Column index of the subplot.
        """
        ax = self.axs[j,i]
        current_ticks = ax.get_yticklabels()
        current_labels = [
            formatter(
                current_ticks[i].get_text()
                ) for i in range(len(current_ticks))]
        ax.set_yticklabels(current_labels)

    def update_legend(self):
        """
        Updates the legend of the figure.
        """
        self.fig.tight_layout()
        plt.subplots_adjust(wspace=0.12, hspace=0.15)
        self.fig.legend(
            handles=self.legend_elements, loc=(0.75, .8))

    def save(self, path):
        """
        Saves the figure to the specified path.

        Parameters:
            path (str): Path to save the figure.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=300)



def generate_mesh_predictions(model, ranges, num_points=50):
    """
    Generate model predictions on a grid sampled from specified ranges across any number of dimensions.
    
    Parameters:
    - model: The trained PyTorch model.
    - ranges: A list of tuples [(min1, max1), (min2, max2), ...] defining the range for each dimension.
    - num_points: Number of points to sample per dimension for the grid (default: 50).
    
    Returns:
    - grid: The grid of points across all dimensions (n_points, n_dimensions).
    - predictions: The model predictions for each point on the grid.
    """
    n_dimensions = len(ranges)  # Number of input dimensions
    bounds = []
    
    # Create linearly spaced points for each dimension based on the given ranges
    for dim_range in ranges:
        min_val, max_val = dim_range
        bounds.append(np.linspace(min_val, max_val, num_points))
    
    # Create a grid of points by taking the Cartesian product of the bounds across dimensions
    grid = np.array(np.meshgrid(*bounds)).T.reshape(-1, n_dimensions)
    
    # Convert the grid to a PyTorch tensor
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(grid_tensor).numpy()  # Assuming the model output is convertible to numpy
    
    return grid, predictions