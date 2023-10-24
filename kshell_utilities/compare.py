from __future__ import annotations
from typing import Union, Tuple
from itertools import cycle, islice
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .kshell_utilities import ReadKshellOutput
from .general_utilities import (
    level_plot, level_density, gamma_strength_function_average
)

class Compare:
    """
    Plot levels, level density and gamma strength function for easy
    comparison between multiple kshell outputs.
    """
    def __init__(self,
            kshell_outputs: list[ReadKshellOutput],
            legend_labels: Union[None, list[str]] = None,
        ):
        """
        Initialize instance with the given kshell outputs and a default
        color palette.

        Parameters
        ----------
        kshell_outputs : list[ReadKshellOutput]
            list of instances of the `ReadKshellOutput` class to be
            plotted together.

        legend_labels : Union[None, list[str]]
            A list of labels for the legends of the plots. The number of
            labels must equal the number of elements in
            `kshell_outputs`.
        """
        type_error_msg = 'kshell_outputs must be a list of ReadKshellOutput'
        type_error_msg += ' instances (the return value of ksutil.loadtxt).'
        type_error_msg += ' Eg: ksutil.Compare(kshell_outputs=[V50, V51]).'
        if not isinstance(kshell_outputs, (list, tuple)):
            raise TypeError(type_error_msg)
        
        else:
            if not all(isinstance(i, ReadKshellOutput) for i in kshell_outputs):
                raise TypeError(type_error_msg)
            
        if isinstance(legend_labels, list):
            if len(legend_labels) != len(kshell_outputs):
                msg = (
                    "The number of labels must equal the number of"
                    " ReadKshellOutput instances!"
                )
                raise RuntimeError(msg)

        self._legend_labels = legend_labels
        self._kshell_outputs = kshell_outputs
        self._color_palette = sns.color_palette(
            palette = "tab10",
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
        filter_spins: Union[None, list] = None,
        filter_parity: Union[None, str] = None,
        use_relative_energy: bool = False,
        ):
        """
        Draw level plots for all kshell outputs. 

        Parameters
        ----------
        ax : Union[None, plt.Axes]
            matplotlib Axes on which to plot. If None, plt.Figure and
            plt.Axes is generated in this function.

        use_relative_energy : bool
            Use relative energy (with respect to the ground state) for
            the y-axis. Default is `False`. I think one should be
            careful with using this option, as it can be misleading
            when comparing different nuclei.

        See level_plot in general_utilities.py for details on the other
        parameters.
        """
        ax_input, fig, ax = self._get_fig_and_ax(ax)
        xticks = {}

        is_half_integer: bool = any(self._kshell_outputs[0].levels[:, 1]%2 == 1)
        for kshell_output in self._kshell_outputs:
            if is_half_integer != any(kshell_output.levels[:, 1]%2 == 1):
                """
                If both integer and half integer angular momentum
                exsists in two or more kshell data sets, then the line
                width must be halved to make room for both.
                """
                line_width = 0.2
                x_offset_scale = 0.5
                break

        else:
            """
            In this case, the kshell outputs contain either integer or
            half integer angular momenta, not both.
            """
            line_width = 0.4
            x_offset_scale = 1

        if self._legend_labels is None:
            labels: list = [i.nucleus for i in self._kshell_outputs]
        else:
            labels: list = self._legend_labels

        for color, kshell_output, label in zip(
                self._color_palette, self._kshell_outputs, labels
            ):

            if use_relative_energy:
                levels_tmp = kshell_output.levels.copy()
                levels_tmp[:, 0] -= kshell_output.ground_state_energy
            else:
                levels_tmp = kshell_output.levels

            level_plot(
                levels = levels_tmp,
                include_n_levels = include_n_levels,
                filter_spins = filter_spins,
                filter_parity = filter_parity,
                ax = ax,
                color = color,
                line_width = line_width,
                x_offset_scale = x_offset_scale,
            )

            for tick_position, tick_label in zip(
                    ax.get_xticks(),
                    [l.get_text() for l in ax.get_xticklabels()]
                    ):

                xticks[tick_position] = tick_label

            ax.plot([], [], label=label, color=color)

        ax.set_xticks(ticks=list(xticks.keys()), labels=list(xticks.values()))

        ax.legend(loc="lower right")

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
