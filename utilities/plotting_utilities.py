"""
    Set of simple functions to help with plotting more attractive matplotlib
    plots
"""

# Module imports
import matplotlib.pyplot as plt
import numbers
import math
import numpy as np


def plot_object_grid(plot_object, plot_functions, **kwargs):
    """
    Function that takes in a plotting class and a number of methods to plot
    and then returns a grid with all the relevant figures plotted

    :param plot_object: Class with a number of plotting functions bound to it
    :param plot_functions: The plotting functions that are bound to the class
    :param kwargs: Keyword Arguments to pass to the figure instantiation
    :return:
    """

    # Check that the object has the necessary functions
    if isinstance(plot_functions, list):
        good_functions = {function: {} for function in plot_functions
                          if hasattr(plot_object, function)}
    else:
        good_functions = {function: params for function, params in plot_functions.items()
                          if hasattr(plot_object, function)}

    # Get the grid size
    grid_x, grid_y = get_grid_parameters(len(good_functions))

    plt_fig = plt.figure(**kwargs)

    # Iterate over plotting functions and call them
    for grid_pos, (function, params) in enumerate(good_functions.items()):
        call_function = getattr(plot_object, function)
        plt_fig.add_subplot(grid_x, grid_y, grid_pos + 1)
        call_function(**params)


def get_grid_parameters(n_plots):
    """

    Function to convert a number of plots to a best size 2D grid for plotting

    :param n_plots: Number of plots
    :return: tuple showing how many plots on the horizontal and vertical axes of the grid
    """

    for n_horizontal in range(int(math.sqrt(n_plots)), 0, -1):
        n_vertical = int(n_plots / n_horizontal)
        if n_plots % n_horizontal == 0:
            return n_horizontal, n_vertical


def plot_table(pandas_df, number_format="1.5f", title=None):
    """

    Function to plot a pretty table from a pandas DataFrame

    :param pandas_df: Input dataframe for the table
    :param number_format: Format in which we want to display numbers
    :param title: Title for the plot containing the table
    :return: Matplotlib plotting object
    """

    # Get Row and Column Headers
    column_headers = pandas_df.columns.tolist()
    row_headers = pandas_df.index.tolist()

    # Define Colors
    r_colors = plt.cm.BuPu(np.full(len(row_headers), 0.5))
    c_colors = plt.cm.BuPu(np.full(len(column_headers), 0.5))

    # Format the numerical data
    cell_text = []
    for index, row in pandas_df.iterrows():
        row_text = []
        for value in row:
            if isinstance(value, numbers.Number):
                row_text.append(f"{value:{number_format}}")
            else:
                row_text.append(str(value))
        cell_text.append(row_text)

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=row_headers,
                          rowColours=r_colors,
                          rowLoc='right',
                          colColours=c_colors,
                          colLabels=column_headers,
                          loc='center')

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    # Add a title
    if title:
        ax.title.set_text(title)

    return the_table
