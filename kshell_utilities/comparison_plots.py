from typing import Union, Callable, Tuple, Iterable
from itertools import cycle, islice
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .kshell_utilities import ReadKshellOutput
from .general_utilities import (
    level_plot, level_density, gamma_strength_function_average
)


class ComparisonPlots:
    """
    Plot levels, level density and gamma strength function for easy comparison
    between multiple kshell outputs.
    """
    def __init__(self, *kshell_outputs):
        """
        Initialize instance with the given kshell outputs and a default color
        palette.

        Parameters
        ----------
        kshell_outputs : list[ReadKshellOutput]
            list of instances of the ReadKshellOutput class to be plotted 
            together.
        """
        self.kshell_outputs = kshell_outputs
        self.color_palette = sns.color_palette(
            palette = "deep",
            n_colors = len(self.kshell_outputs))

    def set_color_palette(
            self, 
            color_palette: Union[str, list[str], list[tuple]]):
        """
        Set the color palette to use for the plots.

        Parameters
        ----------
        color_palette : Union[str, list[str], list[tuple]]
            name of the color palette to use or a list of colors.
        """
        required_number_of_colors = len(self.kshell_outputs)
        if isinstance(color_palette, str):
            self.color_palette = sns.color_palette(
                palette = color_palette,
                n_colors = required_number_of_colors)
        elif len(color_palette) >= required_number_of_colors:
            self.color_palette = color_palette
        else:
            warning_message = (
                "The supplied color palette is too short. "
                "Some colors will be repeated.")
            print(warning_message)
            self.color_palette = list(islice(
                cycle(color_palette), 
                required_number_of_colors))

    def plot_levels(
        self,
        include_n_levels: int = 1_000,
        filter_spins: Union[None, list] = None,
        ax: Union[None, plt.Axes] = None):
        """
        Draw level plots for all kshell outputs. 

        See level_plot in general_utilities.py for details on the parameters.
        """
        ax_input = False if (ax is None) else True
        if not ax_input:
            fig, ax = plt.subplots()

        for color, kshell_output in zip(self.color_palette, 
                                        self.kshell_outputs):
            level_plot(
                levels = kshell_output.levels,
                include_n_levels = include_n_levels,
                filter_spins = filter_spins,
                ax = ax,
                color = color)

        if not ax_input:
            plt.show()

