import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itt
import re

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, rgb2hex
import matplotlib.cm as cm

from pathlib import Path
import torch
text_configuration = {
    'usetex': False,
    'axes_labels_size': 12,
    'ticks_labels_size': 10,
    'legend_font_size': 10
}

plt.rc('text', usetex=text_configuration['usetex'])
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

def add_colorbar(fig, color_dimension, label, color_map):
    z = color_dimension # your custom values for the color bar
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))  # normalize based on z values
    cmap = cm.get_map(color_map)  # choose any matplotlib colormap you like
    # Create the color bar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # ScalarMappable needs an array, but we won't use it

    # Manually add a color bar axis to the right of the figure
    cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(label)

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
        self, parameters, labels=None,
        n_ticks_per_axs=4, figsize=(6,5), colormap='viridis'
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

    def add_trace(self, data, trace_name=None, alpha=0.8, color_dimension=None, color_label='', color_bar_cmap=None):
        """
        Adds a new trace to the corner plot.

        Parameters:
            data (pandas.DataFrame): DataFrame containing the data for the trace.
            trace_name (str): Name of the trace.
            alpha (float, optional): Transparency level of the trace. Defaults to 0.8.
            color_dimension (str, optional): Column name in the DataFrame used for color encoding. Defaults to None.
            color_label (str, optional): Label for the color dimension in the plot. Defaults to ''.
            color_bar_cmap (str, optional): Colormap to use for the color bar. Defaults to None.
        """
        column_ranges = np.ptp(data, axis=0)

        single_color = next_color_hex(
                    self.cmap, self.color_counter
                    )
        self.colors.append(single_color)

        self.legend_elements.append(
            Line2D(
                [0], [0], marker='o',  color=single_color,
               label=trace_name,linestyle='None')
            )
        data_ = data.to_numpy()

        if color_dimension is not None:
            color = color_dimension
        else:
            color = single_color

        for i in range(self.d):
            for j in range(i+1, self.d):
                ax = self.axs[j, i]
                #xlim, ylim = ax.get_xlim(), ax.get_ylim()
                ax.scatter(
                    data_[:, i], data_[:, j],
                    s=1, alpha=alpha, c=color, cmap=self.cmap
                )

               # Adding custom text box for as helper
                custom_text = r"({},{})".format(i,j)  # Your custom text
                ax.text(
                    1.3, 1.3, custom_text, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.1, boxstyle='round')
                    )
                ax.set_rasterized(True)

        # Add colobar if needed
        if color_dimension is not None:
            cmap = color_bar_cmap if color_bar_cmap is not None else self.cmap
            add_colorbar(self.fig, color_dimension, color_label, color_map=cmap)
        # Add marginal histograms
        for i in range(self.d):
            ax = self.axs[i, i]
            if i != self.d-1 :
                ax.set_xticklabels([])

            ax.hist(data_[:, i], alpha=0.2, color=single_color, histtype='stepfilled')
            ax.hist(data_[:, i], alpha=0.5, color=single_color, histtype='step')
            #ax.set_xscale('log')

            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            #ax.set_ylabel(r'$\mathrm{Counts}$')

            custom_text = r"({},{})".format(i,i)  # Your custom text
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


def generate_meshgrid(dim_ranges, steps_per_dim):
    """
    Generate a meshgrid for an arbitrary number of dimensions and return it in shape (N, m),
    where N is the total number of grid points, and m is the number of dimensions.

    Parameters:
        dim_ranges (list of tuples): Each tuple contains (start, end) for each dimension.
        steps_per_dim (list of int): Number of steps for each dimension.

    Returns:
        grid_points (Tensor): A tensor of shape (N, m) containing the grid points.
    """

    # Generate a list of linspace tensors for each dimension
    linspaces = [torch.linspace(start, end, steps) for (start, end), steps in zip(dim_ranges, steps_per_dim)]

    # Generate the meshgrid for all dimensions
    meshgrid = torch.meshgrid(*linspaces, indexing='ij')
    meshgrid_numpy =  [mesh.numpy() for mesh in meshgrid]
    # Flatten each dimension's mesh and stack them together
    flattened_grids = [grid.flatten() for grid in meshgrid]

    # Concatenate all flattened grids into a tensor of shape (N, m)
    grid_points = torch.stack(flattened_grids, dim=-1)

    return grid_points, meshgrid_numpy

def reshape_model_output(model_output, steps_per_dim):
    """
    Reshape the model output from (N, 1) back into the original grid shape.

    Parameters:
        model_output (Tensor): The output of the model with shape (N, 1).
        steps_per_dim (list of int): The number of steps for each dimension, used to reshape the output.

    Returns:
        reshaped_output (Tensor): The reshaped model output with shape corresponding to the grid dimensions.
    """
    # Ensure model_output is (N, 1)
    if len(model_output.shape) == 1:
        model_output = model_output.reshape(-1,1)
    #assert model_output.shape[1] == 1, "Model output must have shape (N, 1)"

    # Compute the total number of grid points (N) from the steps_per_dim
    N = torch.prod(torch.tensor(steps_per_dim))
    assert model_output.shape[0] == N, "Model output size does not match the total number of grid points"

    # Reshape the model output using the steps_per_dim
    reshaped_output = model_output.view(*steps_per_dim)

    return reshaped_output
