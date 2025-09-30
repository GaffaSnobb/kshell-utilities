from __future__ import annotations
import sys, time, warnings
from fractions import Fraction
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import chi2

from .parameters import (
    flags, elements, latex_plot, DPI, MATPLOTLIB_SAVEFIG_FORMAT, GRID_ALPHA
)
from .other_tools import savefig
from ._log import logger

def isotope(name: str, A: int):
    protons = elements[name]
    neutrons = A - protons
    return protons, neutrons

def create_spin_parity_list(
    spins: npt.NDArray,
    parities: npt.NDArray
    ) -> list:
    """
    Pair up input spins and parities in a list of lists.

    Parameters
    ----------
    spins : npt.NDArray
        Array of spins for each energy level.

    parities : npt.NDArray
        Array of corresponding parities for each energy level.

    Returns
    -------
    spins_parities : list
        A nested list of spins and parities [[spin, parity], ...] sorted
        with respect to the spin. N is the number of unique spins in
        'spins'.

    Examples
    --------
    Example list:
    ``` python
    [[1, +1], [3, +1], [5, +1], [7, +1], [9, +1], [11, +1], [13, +1]]
    ```
    """
    spin_parity_list = []
    for i in range(len(spins)):
        if (tmp := [int(spins[i]), int(parities[i])]) in spin_parity_list:
            continue
        spin_parity_list.append(tmp)

    return spin_parity_list

def div0(numerator, denominator):
    """
    Suppress ZeroDivisionError, set x/0 to 0, and set inf, -inf and nan
    to 0. Author Jørgen Midtbø.

    Examples
    --------
    >>> div0([1, 1, 1], [1, 2, 0])
    array([1. , 0.5, 0. ])
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.true_divide(numerator, denominator)
        res[~np.isfinite(res)] = 0    # -inf inf NaN
    return res

def gamma_strength_function_average(
    levels: npt.NDArray,
    transitions: npt.NDArray,
    bin_width: float | int,
    Ex_min: float | int,
    Ex_max: float | int,
    multipole_type: str,
    Ex_final_min: float | int = 0,
    Ex_final_max: float | int = np.inf,
    include_n_levels: float | int = np.inf,
    filter_spins: None | list = None,
    filter_parities: str = "both",
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Calculate the gamma strength function averaged over total angular
    momenta, parities, and initial excitation energies.
    
    Author: Jørgen Midtbø.
    Modified by: GaffaSnobb.
    
    TODO: Make res.transitions_BXL.ji, res.transitions_BXL.pii, etc.
    class attributes (properties).

    Parameters
    ----------
    levels : npt.NDArray
        Array containing energy, spin, and parity for each excited
        state. [[E, 2*spin, parity, idx], ...]. idx counts how many
        times a state of that given spin and parity has occurred. The
        first 0+ state will have an idx of 1, the second 0+ will have an
        idx of 2, etc.

    transitions : npt.NDArray
        Array containing transition data for the specified
        multipolarity.
        OLD:
        Mx8 array containing [2*spin_final, parity_initial, Ex_final,
        2*spin_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)]

        OLD NEW:
        [2*spin_initial, parity_initial, Ex_initial, 2*spin_final,
        parity_final, Ex_final, E_gamma, B(.., i->f), B(.., f<-i)]

        NEW:
        [2*spin_initial, parity_initial, idx_initial, Ex_initial,
        2*spin_final, parity_final, idx_final, Ex_final, E_gamma,
        B(.., i->f), B(.., f<-i)]

    bin_width : float | int
        The width of the energy bins. A bin width of 0.2 contains 20
        states of uniform spacing of 0.01.

    Ex_min : float | int
        Lower limit for initial level excitation energy, usually in MeV.

    Ex_max : float | int
        Upper limit for initial level excitation energy, usually in MeV.

    Ex_final_min : float | int
        Lower limit for final level excitation energy, usually in MeV.

    Ex_final_max : float | int
        Upper limit for final level excitation energy, usually in MeV.

    multipole_type : str
        Choose whether to calculate for 'E1', 'M1' or 'E2'.

    include_n_levels : float | int
        The number of levels per spin to include. Example:
        include_n_levels = 100 will include only the 100 lowest laying
        levels for each spin.

    filter_spins : None | list
        Which spins to include in the GSF. If None, all spins are
        included. TODO: Make int valid input too.

    filter_parities : str
        Which parities to include in the GSF. 'both', '+', '-' are
        allowed.

    Returns
    -------
    bins : npt.NDArray
        The bins corresponding to gSF_ExJpiavg (x values for plot).
        
    gSF_ExJpiavg : npt.NDArray
        The gamma strength function.

    n_transitions_array : npt.NDArray
        Count the number of transitions, as a function of gamma energy,
        involved in the GSF calculation. For calculating Porter-Thomas
        fluctuations in the GSF by

            r(E_gamma) = sqrt(2/n(E_gamma))

        where n is the number of transitions for each gamma energy, used
        to calculate the GSF. See for example DOI:
        10.1103/PhysRevC.98.054303 for details.
    """
    skip_counter = {    # Debug.
        "Transit: Energy range (less)": 0,
        "Transit: Energy range (greater)": 0,
        "Transit: Energy final range (less)": 0,
        "Transit: Energy final range (greater)": 0,
        "Transit: Number of levels": 0,
        "Transit: Parity": 0,
        "Transit: Ei <= Ef": 0,
        "Level density: Energy range": 0,
        "Level density: Number of levels": 0,
        "Level density: Parity": 0
    }
    total_gsf_time = time.perf_counter()
    
    if filter_parities == "both":
        filter_parities = [-1, +1]
    
    elif filter_parities == "-":
        filter_parities = [-1]
    
    elif filter_parities == "+":
        filter_parities = [+1]

    else:
        msg = f"filter_parities must be '+', '-' or 'both'! Got {filter_parities}"
        raise ValueError(msg)

    if (Ex_min < 0) or (Ex_max < 0):
        msg = "Ex_min and Ex_max cannot be negative!"
        raise ValueError(msg)

    if Ex_max < Ex_min:
        msg = "Ex_max cannot be smaller than Ex_min!"
        raise ValueError(msg)
    
    if (Ex_final_min < 0) or (Ex_final_max < 0):
        msg = "Ex_final_min and Ex_final_max cannot be negative!"
        raise ValueError(msg)

    if Ex_final_max < Ex_final_min:
        msg = "Ex_final_max cannot be smaller than Ex_final_min!"
        raise ValueError(msg)

    prefactors: dict[str, float] = {   # Factor for converting from B(XL) to GSF
        "M1": 11.5473e-9, # [1/(mu_N**2*MeV**2)].
        # "E1": 1.047e-6,
        "E1": 3.4888977e-7,
        "E2": 0.80632e-12,   # PhysRevC.90.064321
    }
    prefactor = prefactors[multipole_type]

    # Extract data to a more readable form:
    n_transitions = len(transitions[:, 0])
    n_levels = len(levels[:, 0])
    E_ground_state = levels[0, 0] # Read out the absolute ground state energy so we can get relative energies later.

    try:
        Ex, spins, parities, level_counter = np.copy(levels[:, 0]), levels[:, 1], levels[:, 2], levels[:, 3]
    
    except IndexError as err:
        msg = f"{err.__str__()}\n"
        msg += "Error probably due to old tmp files. Use loadtxt parameter"
        msg += " load_and_save_to_file = 'overwrite' (once) to re-read data from the"
        msg += " summary file and generate new tmp files."
        raise Exception(msg) from err

    Ex_initial = np.copy(transitions[:, 3])   # To avoid altering the raw data.
    Ex_final = np.copy(transitions[:, 7])

    if Ex_initial[0] < 0:
        """
        Adjust energies relative to the ground state energy if they have
        not been adjusted already. The ground state energy is usually
        minus a few tens of MeV and above, so checking absolute value
        above 10 MeV is probably safe. Cant check for equality to zero
        since the initial state will never be zero.
        NOTE: Just check if the value is negative instead?
        2023-08-31: I have now changed it to check if the value is
        negative. Fingers crossed for no negative side effects!
        """
        Ex_initial -= E_ground_state
        Ex_final -= E_ground_state

    if Ex[0] != 0:
        """
        Adjust energies relative to the ground state energy if they have
        not been adjusted already.
        """
        Ex -= E_ground_state

    if (Ex_actual_max := np.max(Ex)) < Ex_max:
        msg = "Requested max excitation energy is greater than the largest"
        msg += " excitation energy in the data file."
        msg += f" Changing Ex_max from {Ex_max} to {Ex_actual_max}."
        Ex_max = Ex_actual_max
        print(msg)

    """
    Find index of first and last bin (lower bin edge) where we put
    counts. It's important to not include the other Ex bins in the
    averaging later, because they contain zeros which will pull the
    average down.
    
    Bin alternatives:
    bin_array = np.linspace(0, bin_width*n_bins, n_bins + 1) # Array of lower bin edge energy values
    bin_array_middle = (bin_array[0: -1] + bin_array[1:])/2 # Array of middle bin values
    """
    Ex_min_idx = int(Ex_min/bin_width)
    Ex_max_idx = int(Ex_max/bin_width)
    n_bins = int(np.ceil(Ex_max/bin_width)) # Make sure the number of bins cover the whole Ex region.

    """
    B_pixel_sum[Ex_final_idx, E_gamma_idx, spin_parity_idx] contains the
    summed reduced transition probabilities for all transitions
    contained within the Ex_final_idx bin, E_gamma_idx bin, and
    spin_parity_idx bin. B_pixel_counts counts the number of transitions
    within the same bins.
    """
    spin_parity_list = create_spin_parity_list(spins, parities) # To create a unique index for every [spin, parity] pair.
    n_unique_spin_parity_pairs = len(spin_parity_list)
    B_pixel_sum = np.zeros((n_bins, n_bins, n_unique_spin_parity_pairs))     # Summed B(..) values for each pixel.
    B_pixel_count = np.zeros((n_bins, n_bins, n_unique_spin_parity_pairs))   # The number of transitions.
    rho_ExJpi = np.zeros((n_bins, n_unique_spin_parity_pairs))  # (Ex, Jpi) matrix to store level density
    gSF = np.zeros((n_bins, n_bins, n_unique_spin_parity_pairs))    
    n_transitions_array = np.zeros(n_bins, dtype=int)  # Count the number of transitions per gamma energy bin.
    included_transitions = []
    transit_gsf_time = time.perf_counter()

    for transition_idx in range(n_transitions):
        """
        Iterate over all transitions in the transitions matrix and add
        up all reduced transition probabilities and the number of
        transitions in the correct bins.
        """
        if Ex_initial[transition_idx] <= Ex_final[transition_idx]:
            """
            Probably safest to skip these?
            """
            skip_counter["Transit: Ei <= Ef"] += 1
            continue
        
        if Ex_initial[transition_idx] < Ex_min:
            """
            Check if transition is within min limit, skip if not.
            """
            skip_counter["Transit: Energy range (less)"] += 1   # Debug.
            continue

        if Ex_initial[transition_idx] >= Ex_max:
            skip_counter["Transit: Energy range (greater)"] += 1
            continue

        if Ex_final[transition_idx] < Ex_final_min:
            skip_counter["Transit: Energy final range (less)"] += 1
            continue

        if Ex_final[transition_idx] >= Ex_final_max:
            skip_counter["Transit: Energy final range (greater)"] += 1
            continue

        idx_initial = transitions[transition_idx, 2]
        idx_final = transitions[transition_idx, 6]

        if (idx_initial > include_n_levels) or (idx_final > include_n_levels):
            """
            Include only 'include_n_levels' number of levels. Defaults
            to np.inf (include all).
            """
            skip_counter["Transit: Number of levels"] += 1
            continue

        spin_initial = int(transitions[transition_idx, 0])  # int cast might not be necessary here.

        if filter_spins is not None:
            if spin_initial/2 not in filter_spins:
                """
                Skip transitions to or from levels of total angular momentum
                not in the filter list.
                """
                try:
                    skip_counter[f"Transit: j init: {spin_initial}"] += 1
                except KeyError:
                    skip_counter[f"Transit: j init: {spin_initial}"] = 1
                continue

        parity_initial = transitions[transition_idx, 1]
        parity_final = transitions[transition_idx, 5]

        if multipole_type in ["M1", "E2"]:
            assert parity_initial == parity_final

        elif multipole_type == "E1":
            assert parity_initial != parity_final

        else:
            msg = "Should not be able to get to this point!"
            raise RuntimeError(msg)

        if (parity_initial not in filter_parities):# or (parity_final not in filter_parities):
            """
            Skip initial or final parities which are not in the filter
            list. NOTE: Might be wrong to filter on the final parity.
            """
            skip_counter["Transit: Parity"] += 1
            continue

        """
        NOTE: Should I check `transitions[transition_idx, 9]` (the B decay
        value) and skip transitions which are zero?
        """
        # Get bin index for E_gamma and Ex. Indices are defined with respect to the lower bin edge.
        included_transitions.append(transitions[transition_idx])
        E_gamma_idx = int(transitions[transition_idx, 8]/bin_width)
        Ex_initial_idx = int(Ex_initial[transition_idx]/bin_width)
        
        n_transitions_array[E_gamma_idx] += 1    # Count the number of transitions involved in this GSF (Porter-Thomas fluctuations).
        spin_parity_idx = spin_parity_list.index([spin_initial, parity_initial])

        B_pixel_sum[Ex_initial_idx, E_gamma_idx, spin_parity_idx] += \
            transitions[transition_idx, 9]
        B_pixel_count[Ex_initial_idx, E_gamma_idx, spin_parity_idx] += 1

    transit_gsf_time = time.perf_counter() - transit_gsf_time
    level_density_gsf_time = time.perf_counter()
    
    for levels_idx in range(n_levels):
        """
        Calculate the level density for each (Ex, spin_parity) pixel.
        """
        if Ex[levels_idx] >= Ex_max:
            """
            Skip if level is outside range. Only upper limit since
            decays to levels below the lower limit are allowed.
            """
            skip_counter["Level density: Energy range"] += 1
            continue
        
        if level_counter[levels_idx] > include_n_levels:
            """
            Include only 'include_n_levels' number of levels. Defaults
            to np.inf (include all).
            """
            skip_counter["Level density: Number of levels"] += 1
            continue

        if filter_spins is not None:
            if (spin_tmp := levels[levels_idx, 1]/2) not in filter_spins:
                """
                Skip levels of total angular momentum not in the filter
                list.
                """
                try:
                    skip_counter[f"Level density: j: {spin_tmp}"] += 1
                except KeyError:
                    skip_counter[f"Level density: j: {spin_tmp}"] = 1
                continue

        Ex_idx = int(Ex[levels_idx]/bin_width)

        spin_parity_idx = \
            spin_parity_list.index([spins[levels_idx], parities[levels_idx]])
        
        rho_ExJpi[Ex_idx, spin_parity_idx] += 1

    level_density_gsf_time = time.perf_counter() - level_density_gsf_time

    rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1.
    gsf_time = time.perf_counter()
    for spin_parity_idx in range(n_unique_spin_parity_pairs):
        """
        Calculate gamma strength functions for each [Ex, E_gamma,
        spin_parity] individually using the partial level density for
        each [Ex, spin_parity].
        """
        for Ex_idx in range(n_bins):
            gSF[Ex_idx, :, spin_parity_idx] = \
                prefactor*rho_ExJpi[Ex_idx, spin_parity_idx]*div0(
                    numerator = B_pixel_sum[Ex_idx, :, spin_parity_idx],
                    denominator = B_pixel_count[Ex_idx, :, spin_parity_idx]
                )

    gsf_time = time.perf_counter() - gsf_time
    avg_gsf_time = time.perf_counter()

    """
    Update 20171009 (Midtbø): Took proper care to only average over
    the non-zero f(Eg,Ex,J,parity_initial) pixels.

    NOTE: Probably not necessary to set an upper limit on gSF
    due to the input adjustment of Ex_max.
    """
    gSF_currentExrange = gSF[Ex_min_idx:Ex_max_idx + 1, :, :]
    gSF_ExJpiavg = div0(
        numerator = gSF_currentExrange.sum(axis = (0, 2)),
        denominator = (gSF_currentExrange != 0).sum(axis = (0, 2))
    )
    
    avg_gsf_time = time.perf_counter() - avg_gsf_time

    bins = np.linspace(0, Ex_max, n_bins + 1)
    bins = (bins[:-1] + bins[1:])/2   # Middle point of the bins. NOTE: Why did I choose this...?
    bins = bins[:len(gSF_ExJpiavg)]
    included_transitions = np.array(included_transitions)

    total_gsf_time = time.perf_counter() - total_gsf_time
    
    if flags["debug"]:
        transit_total_skips = \
            sum([skip_counter[key] for key in skip_counter if key.startswith("Transit")])
        level_density_total_skips = \
            sum([skip_counter[key] for key in skip_counter if key.startswith("Level density")])
        n_transitions_included = n_transitions - transit_total_skips
        n_levels_included = n_levels - level_density_total_skips
        print("--------------------------------")
        print(f"{transit_gsf_time = } s")
        print(f"{level_density_gsf_time = } s")
        print(f"{gsf_time = } s")
        print(f"{avg_gsf_time = } s")
        print(f"{total_gsf_time = } s")
        print(f"{multipole_type = }")
        for elem in skip_counter:
            print(f"Skips: {elem}: {skip_counter[elem]}")
        print(f"{transit_total_skips = }")
        print(f"{n_transitions = }")
        print(f"{n_transitions_included = }")
        print(f"{level_density_total_skips = }")
        print(f"{n_levels = }")
        print(f"{n_levels_included = }")
        print("--------------------------------")

    return bins, gSF_ExJpiavg, n_transitions_array, included_transitions

def level_plot(
    levels: npt.NDArray,
    include_n_levels: int = 1_000,
    filter_spins: None | list = None,
    filter_parity: None | str = None,
    ax: None | plt.Axes = None,
    color: None | str | tuple[str, str] = None,
    line_width: float = 0.4,
    x_offset_scale: float = 1.0,
    alpha: float = 0.5,
    line_thickness: None | float = None,
    ):
    """
    Generate a level plot for a single isotope. Total angular momentum
    on the x axis, energy on the y axis.

    Parameters
    ----------
    levels : npt.NDArray
        NxM array of [[energy, j, parity, index], ...]. This is the
        instance attribute 'levels' of ReadKshellOutput. N is the number
        of levels, M is the number of parameters.
    
    include_n_levels : int
        The maximum amount of levels to plot for each total angular
        momenta. Default set to a large number to indicate ≈ no limit.

    filter_spins : None | list
        Which total angular momenta to include in the plot. If None, all
        total angular momenta are plotted.

    filter_parity : None | str
        A filter for parity. If None (default) then the parity of the
        ground state will be used. `+` is positive, `-` is negative,
        while `both` gives both parities.

    ax : None | plt.Axes
        matplotlib Axes to plot on. If None, plt.Figure and plt.Axes is
        generated in this function.

    color : None | str
        Color to use for the levels. If None, the next color in the
        matplotlib color_cycle iterator is used.
    
    line_width : float
        The width of the level lines. Not really supposed to be changed
        by the user. Set to 0.2 for comparison plots when both integer
        and half integer angular momenta are included, 0.4 else.

    x_offset_scale : float
        To scale the x offset for the hlines. This is used to fit
        columns for both integer and half integer angular momenta, as
        well as both parities.

    alpha : float
        alpha value for the hlines in the level scheme plot.
    """
    ax_input = False if (ax is None) else True
    energies = levels[:, 0]

    # if levels[0, 0] != 0:
    #     """
    #     Adjust energies relative to the ground state energy.
    #     """
    #     energies = levels[:, 0] - levels[0, 0]
    # else:
    #     energies = levels[:, 0]

    spins = levels[:, 1]/2  # levels[:, 1] is 2*spin.
    parities = levels[:, 2]

    allowed_filter_parity = [None, "+", "-", "both"]
    if filter_parity not in allowed_filter_parity:
        msg = f"Allowed parity filters are: {allowed_filter_parity}."
        raise ValueError(msg)

    if filter_parity is None:
        """
        Default to the ground state parity.
        """
        parity_integer: int = [levels[0, 2]]
        parity_symbol: str = "+" if (levels[0, 2] == 1) else "-"
        x_offset = 0    # No offset needed for single parity plot.

    elif filter_parity == "+":
        parity_symbol: str = filter_parity
        parity_integer: list = [1]
        x_offset = 0

    elif filter_parity == "-":
        parity_symbol: str = filter_parity
        parity_integer: list = [-1]
        x_offset = 0

    elif filter_parity == "both":
        line_width /= 2 # Make room for both parities.
        parity_symbol: str = r"-+"
        parity_integer: list = [-1, 1]
        x_offset = 1/4*x_offset_scale  # Offset for plots containing both parities.
    
    if filter_spins is not None:
        spin_scope = np.unique(filter_spins)    # x values for the plot.
    else:
        spin_scope = np.unique(spins)
    
    counts = {} # Dict to keep tabs on how many levels of each angular momentum have been plotted.

    if not ax_input:
        fig, ax = plt.subplots()

    if color is None:
        color = next(ax._get_lines.color_cycle)

    if not isinstance(color, (list, tuple)):
        color = (color, color)


    for i in range(len(energies)):
        if filter_spins is not None:
            if spins[i] not in filter_spins:
                """
                Skip spins which are not in the filter.
                """
                continue

        if parities[i] not in parity_integer:
            continue

        key: str = f"{spins[i]} + {parities[i]}"
        try:
            counts[key] += 1
        except KeyError:
            counts[key] = 1
        
        if counts[key] > include_n_levels:
            """
            Include only the first `include_n_levels` amount of levels
            for any of the spins.
            """
            continue

        ax.hlines(
            y = energies[i],
            xmin = spins[i] - line_width + x_offset*parities[i]*0.9,
            xmax = spins[i] + line_width + x_offset*parities[i]*0.9,
            color = color[int((parities[i] + 1)/2)],    # Convert -1, 1 -> 0, 1 and use as index.
            alpha = alpha,
            linewidth = line_thickness,
        )

    ax.set_xticks(spin_scope)
    ax.set_xticklabels([f"{Fraction(i)}" + r"$^{" + f"{parity_symbol}" + r"}$" for i in spin_scope])
    ax.set_xlabel(r"$j^{\pi}$", labelpad=-5)
    ax.set_ylabel(r"$E$ [MeV]")
    ax.grid(visible=True, alpha=GRID_ALPHA)

    if not ax_input:
        savefig(fig=fig, fname=f"level-plot.{MATPLOTLIB_SAVEFIG_FORMAT}", dpi=DPI)
        plt.show()

def level_density(
    levels: npt.NDArray,
    bin_width: int | float,
    include_n_levels: None | int = None,
    filter_spins: None | int | list = None,
    filter_parity: None | str | int = None,
    E_min: float | int = 0,
    E_max: float | int = np.inf,
    return_counts: bool = False,
    plot: bool = False,
    save_plot: bool = False,
    ax: None | plt.Axes = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Calculate the level density for a given bin size.

    Parameters
    ----------
    levels : npt.NDArray | list
        Nx4 array of [[E, 2*spin, parity, idx], ...] or 1D array / list
        of only energies.

    bin_width : int | float
        Energy interval of which to calculate the density.

    include_n_levels : None | int
        The number of states per spin to include. Example:
        include_n_levels = 100 will include only the 100 lowest laying
        states for each spin.

    filter_spins : None | int | list
        Keep only the levels which have angular momenta in the filter.
        If None, all angular momenta are kept. Input must be the actual
        angular momenta values and not 2*j.

    filter_parity : None | str | int
        Keep only levels of parity 'filter_parity'. +1, -1, '+', '-'
        allowed inputs.

    E_min : None | float | int
        Minimum energy to include in the calculation.

    E_max : None | float | int
        Maximum energy to include in the calculation. If input E_max is
        larger than the largest E in the data set, E_max is set to that
        value.

    return_counts : bool
        Return the counts per bin instead of the density
        (density = counts/bin_width).

    plot : bool
        For toggling plotting on / off.

    save_plot : bool    
        Toogle saving of plot on / off.

    Returns
    -------
    bins : npt.NDArray
        The corresponding bins (x value for plotting).

    density : npt.NDArray
        The level density.

    Raises
    ------
    ValueError:
        If any filter is given when energy_levels is a list of only
        energy levels.

    TypeError:
        If input parameters are of the wrong type.
    """
    if ax is not None:
        """
        If the user gives an ax, then this function should not create a
        new ax, nor call plt.show, nor save the plot.        
        """
        if plot or save_plot:
            msg = "'plot' and 'save_plot' is ignored when an ax is supplied."
            logger.warning(msg)
        
        plot = False
        save_plot = False

    if not isinstance(levels, np.ndarray):
        levels = np.array(levels)

    if not isinstance(filter_spins, (int, float, list, type(None), np.ndarray)):
        msg = f"'filter_spins' must be of type: int, float, list, None. Got {type(filter_spins)}."
        raise TypeError(msg)

    if not isinstance(include_n_levels, (int, type(None))):
        msg = f"'include_n_levels' must be of type: int, None. Got {type(include_n_levels)}."
        raise TypeError(msg)

    if not isinstance(filter_parity, (type(None), int, str)):
        msg = f"'filter_parity' must be of type: None, int, str. Got {type(filter_parity)}."
        raise TypeError(msg)

    if not isinstance(E_min, (int, float)):
        msg = f"'E_min' must be of type: int, float. Got {type(E_min)}."
        raise TypeError(msg)

    if not isinstance(E_max, (int, float)):
        msg = f"'E_max' must be of type: int, float. Got {type(E_max)}."
        raise TypeError(msg)

    if isinstance(filter_parity, str):
        valid_filter_parity = ["+", "-"]
        if filter_parity not in valid_filter_parity:
            msg = f"Valid parity filters are: {valid_filter_parity}."
            raise ValueError(msg)
        
        filter_parity = 1 if (filter_parity == "+") else -1

    if isinstance(filter_spins, (int, float)):
        filter_spins = [filter_spins]

    if (levels.ndim == 1) and (filter_spins is not None):
        msg = "Spin filter cannot be applied to a list of only energy levels!"
        raise ValueError(msg)

    if (levels.ndim == 1) and (include_n_levels is not None):
        msg = "Cannot choose the number of levels per spin if 'levels' is only a list of energies!"
        raise ValueError(msg)
    
    if (levels.ndim == 1) and (filter_parity is not None):
        msg = "Parity filter cannot be applied to a list of only energy levels!"
        raise ValueError(msg)

    energy_levels = np.copy(levels) # Just in case.
    if energy_levels.ndim == 1:
        """
        'levels' only contains energy values.
        """
        if energy_levels[0] < 0:
            msg = "Please scale energies relative to the ground state"
            msg += " energy before calculating the NLD!"
            raise ValueError(msg)

    elif energy_levels.ndim == 2:
        """
        'levels' is a multidimensional array on the form
        [[E, 2*spin, parity, idx], ...]. Subtract ground state energy
        from all energies.
        """
        energy_levels[:, 0] -= energy_levels[0, 0]
        
        if include_n_levels is not None:
            """
            Include ony 'include_n_levels' of levels per angular
            momentum and parity pair.
            """
            indices = energy_levels[:, 3]  # Counter for the number of levels per spin.
            energy_levels = energy_levels[indices <= include_n_levels]

        if filter_spins is not None:
            """
            filter_spins is a list of angular momenta. Inside this if
            statement we know that 'levels' is a multidimensional array due
            to the check inside the previous except.
            levels: 
            """
            filter_spins = [2*j for j in filter_spins]  # energy_levels has 2*j to avoid fractions.

            mask_list = []
            for j in filter_spins:
                """
                Create a [bool1, bool2, ...] mask for each j.
                """
                mask_list.append(energy_levels[:, 1] == j)

            energy_levels = energy_levels[np.logical_or.reduce(mask_list)]  # Contains only levels of j in the filter.

        if filter_parity is not None:
            energy_levels = energy_levels[energy_levels[:, 2] == filter_parity]
        
        energy_levels = energy_levels[:, 0]

    if energy_levels.size == 0:
        msg = f"No energy levels for {filter_parity = }, {filter_spins = }"
        logger.error(msg)
        raise RuntimeError(msg)

    E_max = min(E_max, energy_levels[-1] + 0.1) # E_max cant be larger than the largest energy in the data set. Add small number to include the final level(s) in the counting.
    if E_max <= E_min:
        """
        This behaviour is OK for angular momentum distribution heatmaps
        where the NLDs for different angular momenta are compared using
        the same bin range.

        In this situation, the density at E_min is of course zero
        because there are no levels of this energy for the given angular
        momentum.
        """
        bins = np.array([E_min])
        density = np.array([0])
        return bins, density

    bins = np.arange(E_min, E_max + bin_width, bin_width)
    # bins[-1] = E_max    # arange will mess up the final bin if it does not match the bin width. 2025-09-30: Don't know if I believe back-in-the-days myself here...

    n_bins = len(bins)
    counts = np.zeros(n_bins - 1)

    for i in range(n_bins - 1):
        mask_1 = energy_levels >= bins[i]
        mask_2 = energy_levels < bins[i+1]
        mask_3 = np.logical_and(mask_1, mask_2)
        counts[i] = sum(mask_3)
        # counts[i] = np.sum(bins[i] <= energy_levels[energy_levels < bins[i + 1]])
    
    density = (counts/bin_width)
    # bins = bins[1:]
    bins = bins[:-1]    # Maybe just a matter of preference...?

    if plot or (ax is not None):
        if plot: fig, ax = plt.subplots()
        
        if return_counts:
            ax.step(bins, counts, color="black")
            ax.set_ylabel(r"Counts")
        else:
            ax.step(bins, density, color="black")
            ax.set_ylabel(r"NLD [MeV$^{-1}$]")
        ax.set_xlabel("E [MeV]")
        # ax.legend([f"{bin_width=} MeV"])
        
        if plot: ax.grid()
        
        if save_plot:
            savefig(fig=fig, fname=f"nld.{MATPLOTLIB_SAVEFIG_FORMAT}", dpi=DPI)
        
        if plot: plt.show()

    if return_counts:
        return bins, density, counts
    else:
        return bins, density

def porter_thomas(
    transitions: npt.NDArray,
    Ei: int | float | list,
    BXL_bin_width: int | float,
    j_list: list | None = None,
    Ei_bin_width: int | float = 0.1,
    E_gamma_min: float = 0.0,
    E_gamma_max: float = np.inf,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Calculate the distribution of B(XL)/mean(B(XL)) values.

    Parameters
    ----------
    transitions : npt.NDArray
        Array containing transition data for the specified
        multipolarity.

        [2*spin_initial, parity_initial, idx_initial, Ex_initial,
        2*spin_final, parity_final, idx_final, Ex_final, E_gamma,
        B(XL, i->f), B(XL, f<-i)]

    Ei : int, float, list
        The initial excitation energy of the transitions where the
        distribution will be calculated. If Ei is only a number, then a
        bin of size 'Ei_bin_width' around Ei will be used. If Ei is a
        list, tuple, or array with both a lower and an upper limit, then
        all excitation energies in that interval will be used.

    BXL_bin_width : int, float
        The bin size of the BXL values for the distribution (not the Ei
        bin size!).

    Ei_bin_width : int, float
        The size of the initial energy bin if 'Ei' is only one number.
        Will not be used if 'Ei' is both a lower and an upper limit.

    Returns
    -------
    BXL_bins : npt.NDArray
        The BXL bins (x values).

    BXL_counts : npt.NDArray
        The number of counts in each BXL_bins bin (y values).
    """
    E_gamma_mask = np.logical_and(
        transitions[:, 8] >= E_gamma_min,
        transitions[:, 8] < E_gamma_max,
    )
    pt_prepare_data_time = time.perf_counter()
    if isinstance(Ei, (list, tuple, np.ndarray)):
        """
        If Ei defines a lower and an upper limit.
        """
        Ei_mask = np.logical_and(
            transitions[:, 3] >= Ei[0],
            transitions[:, 3] < Ei[-1]
        )
        BXL = transitions[np.logical_and(Ei_mask, E_gamma_mask)]
        assert BXL.size, "No values in BXL!"
    else:
        BXL = transitions[np.abs(transitions[:, 3] - Ei) < Ei_bin_width] # Consider only levels around Ei.

    if j_list is not None:
        """
        Create a mask of j values for the transitions array. Allow only
        entries with initial angular momenta in j_list.
        """
        if not isinstance(j_list, list):
            msg = f"j_list must be of type list! Got {type(j_list)}."
            raise TypeError(msg)

        j_list = [2*j for j in j_list]  # Angular momenta are stored as 2*j to avoid fractions.

        mask_list = []
        for j in j_list:
            """
            Create a [bool1, bool2, ...] mask for each j.
            """
            mask_list.append(BXL[:, 0] == j)

        BXL = BXL[np.logical_or.reduce(mask_list)]  # Contains only transitions of j in the filter.

    # BXL = np.copy(BXL[:, 9]) # The 9th col. is the reduced decay transition probabilities.
    n_BXL_before = len(BXL)
    idxi_masks = []
    pii_masks = []
    ji_masks = []
    BXL_tmp = []

    initial_indices = np.unique(BXL[:, 2]).astype(int)
    initial_parities = np.unique(BXL[:, 1]).astype(int)
    initial_j = np.unique(BXL[:, 0])

    for idxi in initial_indices:
        idxi_masks.append(BXL[:, 2] == idxi)

    for pii in initial_parities:
        pii_masks.append(BXL[:, 1] == pii)

    for ji in initial_j:
        ji_masks.append(BXL[:, 0] == ji)

    n_B_skips = 0
    for pii in pii_masks:
        for idxi in idxi_masks:
            for ji in ji_masks:
                mask = np.logical_and(ji, np.logical_and(pii, idxi))
                tmp = BXL[mask][:, 9]   # 9 is B decay.
                n_B_skips += sum(tmp == 0)  # Count the number of zero values.
                tmp = tmp[tmp != 0] # Remove all zero values.
                if not tmp.size:
                    """
                    Some combinations of masks might not match any
                    levels.
                    """
                    continue
                
                BXL_tmp.extend(tmp/tmp.mean())  # Normalise to the mean value per bin.
                # BXL_tmp.extend(tmp)

    BXL = np.asarray(BXL_tmp)
    BXL.sort()
    # BXL = BXL/np.mean(BXL)    # Normalise to the total mean value.
    n_BXL_after = len(BXL)
    assert np.all(BXL > 0)  # Sanity check.

    if (n_BXL_before - n_B_skips) != n_BXL_after:
        msg = "The number of BXL values has changed too much during the Porter-Thomas analysis!"
        msg += f" This should not happen! n_BXL_after should be: {n_BXL_before - n_B_skips}, got: {n_BXL_after}."
        msg += f"\n{n_BXL_before = }"
        msg += f"\n{n_BXL_after = }"
        msg += f"\n{n_B_skips = }"
        raise RuntimeError(msg)

    BXL_bins = np.arange(0, BXL[-1] + BXL_bin_width, BXL_bin_width)
    n_BXL_bins = len(BXL_bins)
    BXL_counts = np.zeros(n_BXL_bins)
    pt_prepare_data_time = time.perf_counter() - pt_prepare_data_time
    
    pt_count_time = time.perf_counter()
    for i in range(n_BXL_bins - 1):
        """
        Calculate the number of transitions with BXL values between
        BXL_bins[i] and BXL_bins[i + 1].
        """
        BXL_counts[i] = np.sum(BXL_bins[i] <= BXL[BXL < BXL_bins[i + 1]])
    
    pt_count_time = time.perf_counter() - pt_count_time

    if flags["debug"]:
        print("--------------------------------")
        print(f"Porter-Thomas: Prepare data time: {pt_prepare_data_time:.3f} s")
        print(f"Porter-Thomas: Count time: {pt_count_time:.3f} s")
        print(f"{sum(BXL_counts) = }")
        print(f"{n_B_skips = }")
        print(f"{Ei = }")
        print(f"{Ei_bin_width = }")
        print(f"{j_list = }")
        print(f"{E_gamma_min = }")
        print(f"{E_gamma_max = }")
        print(f"{np.mean(BXL) = }")
        print(f"{np.var(BXL) = }")

        # flattened = []
        # for b, c in zip(BXL_bins, BXL_counts):
        #     flattened += [b]*int(c)

        # print(f"{np.var(flattened) = }")
        # print(f"{np.mean(flattened) = }")

    return BXL_bins, BXL_counts

def nuclear_shell_model(
    show_spectroscopic_notation: bool = True,
    show_interactions: bool = True,
    show_cores: bool = True,
    show_16o_core: bool = True,
    show_40ca_core: bool = True,
    show_56ni_core: bool = True,
    show_usd: bool = True,
    show_gxpf: bool = True,
    show_jun45: bool = True,
    show_sdpfsdgmu: bool = True,
    show_sdpfmu: bool = True,
):
    """
    Generate a diagram of the nuclear shell model shell structure.
    """
    latex_plot()
    fig, ax = plt.subplots(figsize=(6.4, 8))
    ax.axis(False)
    fontsize = 15
    x_offset = 0.6
    x_text_offset = x_offset - 0.5

    first_layer_labels = [
        r"$0s$", r"$0p$", r"$0d$", r"$1s$", r"$0f$", r"$1p$", r"$0g$",
        r"$1d$", r"$2s$"
    ]
    first_layer_y = [1, 2.4, 4.2, 4.45, 6.3, 6.8, 9, 10.0, 10.5]
    second_layer_labels = [
        r"$0s_{1/2}$", r"$0p_{3/2}$", r"$0p_{1/2}$", r"$0d_{5/2}$",
        r"$1s_{1/2}$", r"$0d_{3/2}$", r"$0f_{7/2}$", r"$1p_{3/2}$",
        r"$0f_{5/2}$", r"$1p_{1/2}$", r"$0g_{9/2}$", r"$1d_{5/2}$",
        r"$0g_{7/2}$", r"$2s_{1/2}$", r"$1d_{3/2}$"
    ]
    second_layer_y = [
        first_layer_y[0], first_layer_y[1] - 0.15, first_layer_y[1] + 0.15,
        first_layer_y[2] - 0.3, first_layer_y[3], first_layer_y[2] + 0.51,
        first_layer_y[4] - 0.6, first_layer_y[5] - 0.10, first_layer_y[4] + 0.7,
        first_layer_y[5] + 0.5, first_layer_y[6] - 1.0, first_layer_y[7] - 0.4,
        first_layer_y[6] + 0.9, first_layer_y[7] + 0.8, first_layer_y[8]
    ]
    dash_layer = [
        [2 + x_offset, first_layer_y[0], 2.5 + x_offset, second_layer_y[0]],
        [2 + x_offset, first_layer_y[1], 2.5 + x_offset, second_layer_y[1]],
        [2 + x_offset, first_layer_y[1], 2.5 + x_offset, second_layer_y[2]],
        [2 + x_offset, first_layer_y[2], 2.5 + x_offset, second_layer_y[3]],
        [2 + x_offset, first_layer_y[2], 2.5 + x_offset, second_layer_y[5]],
        [2 + x_offset, first_layer_y[3], 2.5 + x_offset, second_layer_y[4]],
        [2 + x_offset, first_layer_y[4], 2.5 + x_offset, second_layer_y[6]],
        [2 + x_offset, first_layer_y[4], 2.5 + x_offset, second_layer_y[8]],
        [2 + x_offset, first_layer_y[5], 2.5 + x_offset, second_layer_y[7]],
        [2 + x_offset, first_layer_y[5], 2.5 + x_offset, second_layer_y[9]],
        [2 + x_offset, first_layer_y[6], 2.5 + x_offset, second_layer_y[10]],
        [2 + x_offset, first_layer_y[7], 2.5 + x_offset, second_layer_y[11]],
        [2 + x_offset, first_layer_y[6], 2.5 + x_offset, second_layer_y[12]],
        [2 + x_offset, first_layer_y[7], 2.5 + x_offset, second_layer_y[13]],
        [2 + x_offset, first_layer_y[8], 2.5 + x_offset, second_layer_y[14]],
    ]
    core_layer_labels = []
    core_layer_y = []
    if show_16o_core:
        core_layer_labels.append(r"$^{16}$O")
        core_layer_y.append(second_layer_y[2] + 0.5)

    if show_40ca_core:
        core_layer_labels.append(r"$^{40}$Ca")
        core_layer_y.append(second_layer_y[5] + 0.5)

    if show_56ni_core:
        core_layer_labels.append(r"$^{56}$Ni")
        core_layer_y.append(second_layer_y[6] + 0.5)

    occupations = [
        2, 4, 2, 6, 2, 4, 8, 4, 6, 2, 10, 6, 8, 2, 4
    ]
    occupations_y = second_layer_y
    cum_occupations = [
        2, 8, 20, 28, 50
    ]
    cum_occupations_y = [
        second_layer_y[0], second_layer_y[2], second_layer_y[5],
        second_layer_y[6], second_layer_y[10]
    ]
    ax.hlines(  # To force the width of the figure.
        y = 1,
        xmin = 3.5 + x_offset,
        xmax = 4.5 + x_offset,
        color = "white"
    )
    ax.hlines(  # To force the width of the figure.
        y = 1,
        xmin = 1,
        xmax = 2,
        color = "white"
    )
    for y, label in zip(first_layer_y, first_layer_labels):
        ax.hlines(
            y = y,
            xmin = 1 + x_offset,
            xmax = 2 + x_offset,
            color = "black",
        )
        fig.text(
            x = 0.12 + x_text_offset,
            y = y/13.95 + 0.067,
            s = label,
            fontsize = fontsize
        )

    for y, label in zip(second_layer_y, second_layer_labels):
        ax.hlines(
            y = y,
            xmin = 2.5 + x_offset,
            xmax = 3.5 + x_offset,
            color = "black",
        )
        fig.text(
            x = 0.6 + x_text_offset,
            y = y/14.2 + 0.067,
            s = label,
            fontsize = fontsize
        )

    for x1, y1, x2, y2 in dash_layer:
        ax.plot([x1, x2], [y1, y2], linestyle="dashed", color="black")

    for occupation, y in zip(occupations, occupations_y):
        fig.text(
            x = 0.69 + x_text_offset,
            y = y/14.2 + 0.067,
            s = occupation,
            fontsize = fontsize - 1
        )

    for occupation, y in zip(cum_occupations, cum_occupations_y):
        fig.text(
            x = 0.73 + x_text_offset,
            y = y/14.2 + 0.067,
            s = occupation,
            fontsize = fontsize - 1
        )

    if show_cores:
        for y, label in zip(core_layer_y, core_layer_labels):
            fig.text(
                x = 0.77 + x_text_offset,
                y = y/14 + 0.067,
                s = label,
                fontsize = fontsize - 1
            )
            fig.text(
                x = 0.73 + x_text_offset,
                # y = y/14 + 0.064,
                y = y/14 + 0.054,
                s = "---------",
                fontsize = fontsize - 1
            )
    if show_interactions:
        # USD
        x1 = 1.35
        x2 = 1.25
        y1 = 4.9
        y2 = 3.83
        if show_usd:
            ax.vlines(
                x = x2,
                ymin = y2,
                ymax = y1,
                color = "darkorange",
            )
            fig.text(
                x = 0.15,
                y = 0.37,
                s = "USD",
                fontsize = 12,
                rotation = "vertical",
                color = "darkorange"
            )
        # GXPF
        y3 = 7.5
        y4 = 5.6
        if show_gxpf:
            ax.vlines(
                x = x2,
                ymin = y4,
                ymax = y3,
                color = "firebrick",
            )
            fig.text(
                x = 0.15,
                y = 0.52,
                s = "GXPF",
                fontsize = 12,
                rotation = "vertical",
                color = "firebrick"
            )
        # SDPF-MU
        x4 = x2 - 0.04
        if show_sdpfmu:
            ax.vlines(
                x = x4,
                ymin = y2,
                ymax = y3,
                color = "royalblue",
            )
            fig.text(
                x = 0.14,
                y = 0.42,
                s = "SDPF-MU",
                fontsize = 12,
                rotation = "vertical",
                color = "royalblue"
            )
        #JUN45
        y7 = 6.5
        y8 = 8.2
        x6 = x4 - 0.04
        if show_jun45:
            ax.vlines(
                x = x6,
                ymin = y8,
                ymax = y7,
                color = "green",
            )
            fig.text(
                x = 0.14,
                y = 0.59,
                s = "JUN45",
                fontsize = 12,
                rotation = "vertical",
                color = "green"
            )
        # SDPF-SDG
        if show_sdpfsdgmu:
            ax.vlines(
                x = x6 - 0.04,
                ymin = y2,
                ymax = 11,
                color = "mediumorchid",
            )
            fig.text(
                x = 0.15,
                y = 0.66,
                s = "SDPFSDG-MU",
                fontsize = 12,
                rotation = "vertical",
                color = "mediumorchid"
            )
    # Spectroscopic notation
    if show_spectroscopic_notation:
        fig.text(
            x = 0.45,
            y = 0.93,
            s = r"$s \;\; p \;\; d \;\; f \;\; g \;\; h$",
            fontsize = fontsize - 1,
            color = "black",
        )
        fig.text(
            x = 0.45,
            y = 0.918,
            s = "------------------",
            fontsize = fontsize - 1,
            color = "black",
        )
        fig.text(
            x = 0.415,
            y = 0.90,
            s = r"$l: 0 \;\; 1 \;\; 2 \;\; 3 \;\; 4 \;\; 5$",
            fontsize = fontsize - 1,
            color = "black",
        )
        fig.text(
            x = 0.405,
            y = 0.88,
            s = r"$\pi:+  -  +  - \, + \, -$",
            fontsize = fontsize - 1,
            color = "black",
        )

    fig.savefig(fname=f"nuclear_shell_model.{MATPLOTLIB_SAVEFIG_FORMAT}", dpi=DPI)
    plt.show()
