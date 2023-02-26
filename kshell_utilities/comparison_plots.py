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
        self._kshell_outputs = kshell_outputs
        self._color_palette = sns.color_palette(
            palette = "deep",
            n_colors = len(self._kshell_outputs))

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
        required_number_of_colors = len(self._kshell_outputs)
        if isinstance(color_palette, str):
            self._color_palette = sns.color_palette(
                palette = color_palette,
                n_colors = required_number_of_colors)
        elif len(color_palette) >= required_number_of_colors:
            self._color_palette = color_palette
        else:
            warning_message = (
                "The supplied color palette is too short. "
                "Some colors will be repeated.")
            print(warning_message)
            self._color_palette = list(islice(
                cycle(color_palette), 
                required_number_of_colors))

    def plot_levels(
        self,
        ax: Union[None, plt.Axes] = None,
        include_n_levels: int = 1_000,
        filter_spins: Union[None, list] = None
        ):
        """
        Draw level plots for all kshell outputs. 

        Parameters
        ----------
        ax : Union[None, plt.Axes]
            matplotlib Axes on which to plot. If None, plt.Figure and plt.Axes 
            is generated in this function.

        See level_plot in general_utilities.py for details on the other
        parameters.
        """
        ax_input, fig, ax = self._get_fig_and_ax(ax)

        for color, kshell_output in zip(self._color_palette, 
                                        self._kshell_outputs):
            level_plot(
                levels = kshell_output.levels,
                include_n_levels = include_n_levels,
                filter_spins = filter_spins,
                ax = ax,
                color = color)

        if not ax_input:
            plt.show()

    def plot_level_densities(
        self,
        ax: Union[None, plt.Axes] = None,
        bin_width: Union[int, float] = 0.2,
        include_n_levels: Union[None, int] = None,
        filter_spins: Union[None, int, list] = None,
        filter_parity: Union[None, str, int] = None,
        E_min: Union[float, int] = 0,
        E_max: Union[float, int] = np.inf
        ):
        """
        Draw level density plots for all kshell outputs.

        Parameters
        ----------
        ax : Union[None, plt.Axes]
            matplotlib Axes on which to plot. If None, plt.Figure and plt.Axes 
            is generated in this function.

        See level_density in general_utilities.py for details on the other
        parameters.
        """
        ax_input, fig, ax = self._get_fig_and_ax(ax)

        ax.set_ylabel(r"NLD [MeV$^{-1}$]")
        ax.set_xlabel("E [MeV]")
        ax.legend([f"{bin_width=} MeV"])

        for color, kshell_output in zip(self._color_palette,
                                        self._kshell_outputs):
            bins, density = level_density(
                levels = kshell_output.levels,
                bin_width = bin_width,
                include_n_levels = include_n_levels,
                filter_spins = filter_spins,
                filter_parity = filter_parity,
                E_min = E_min,
                E_max = E_max,
                return_counts = False,
                plot = False,
                save_plot = False
                )

            ax.step(bins, density, color=color)

        if not ax_input:
            plt.show()

    def plot_gamma_strength_functions(
        self,
        ax: Union[None, plt.Axes] = None,
        bin_width: Union[float, int] = 0.2,
        Ex_min: Union[float, int] = 5,
        Ex_max: Union[float, int] = 50,
        multipole_type: str = "M1",
        prefactor_E1: Union[None, float] = None,
        prefactor_M1: Union[None, float] = None,
        prefactor_E2: Union[None, float] = None,
        initial_or_final: str = "initial",
        partial_or_total: str = "partial",
        include_only_nonzero_in_average: bool = True,
        include_n_levels: Union[None, int] = None,
        filter_spins: Union[None, list] = None,
        filter_parities: str = "both",
        return_n_transitions: bool = False,
        ):
        """
        Draw gamma strength function plots for all kshell outputs.

        Parameters
        ----------
        ax : Union[None, plt.Axes]
            matplotlib Axes on which to plot. If None, plt.Figure and plt.Axes 
            is generated in this function.

        See level_density in general_utilities.py for details on the other
        parameters.
         """
        ax_input, fig, ax = self._get_fig_and_ax(ax)

        for color, kshell_output in zip(self._color_palette,
                                        self._kshell_outputs):
            transitions_dict = {
                "M1": kshell_output.transitions_BM1,
                "E2": kshell_output.transitions_BE2,
                "E1": kshell_output.transitions_BE1
            }

            bins, gsf = gamma_strength_function_average(
                levels = kshell_output.levels,
                transitions = transitions_dict[multipole_type],
                bin_width = bin_width,
                Ex_min = Ex_min,
                Ex_max = Ex_max,
                multipole_type = multipole_type,
                prefactor_E1 = prefactor_E1,
                prefactor_M1 = prefactor_M1,
                prefactor_E2 = prefactor_E2,
                initial_or_final = initial_or_final,
                partial_or_total = partial_or_total,
                include_only_nonzero_in_average = include_only_nonzero_in_average,
                include_n_levels = include_n_levels,
                filter_spins = filter_spins,
                filter_parities = filter_parities,
                return_n_transitions = return_n_transitions
                )
            ax.plot(bins, gsf, color=color)

        if not ax_input:
            plt.show()

    @staticmethod
    def _get_fig_and_ax(
        ax: Union[None, plt.Axes]
        ) -> Tuple[bool, plt.Figure, plt.Axes]:
        """
        Return a matplotlib Figure and Axes on which to plot, and whether these
        were generated in this function or existed previously.

        Parameter:
        ----------
        ax : Union[None, plt.Axes]
            matplotlib Axes on which to plot. If None, plt.Figure and plt.Axes
            is generated by this function.

        Returns:
        --------
        ax_input : bool
            whether a matplotlib Axes was passed.
        fig : plt.Figure
            matplotlib Figure on which to plot.
        ax : plt.Axes
            matplotlib Axes on which to plot. If a matplotlib Axes object was
            passed, it is returned unchanged.
        """
        ax_input = False if (ax is None) else True
        if not ax_input:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        return (ax_input, fig, ax)
