import sys, time, warnings
from typing import Union, Tuple, Optional
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit
from .parameters import flags

def create_spin_parity_list(
    spins: np.ndarray,
    parities: np.ndarray
    ) -> list:
    """
    Pair up input spins and parities in a list of lists.

    Parameters
    ----------
    spins : np.ndarray
        Array of spins for each energy level.

    parities : np.ndarray
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
    levels: np.ndarray,
    transitions: np.ndarray,
    bin_width: Union[float, int],
    Ex_min: Union[float, int],
    Ex_max: Union[float, int],
    multipole_type: str,
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
    # plot: bool = False,
    # save_plot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Calculate the gamma strength function averaged over total angular
    momenta, parities, and initial excitation energies.
    
    Author: Jørgen Midtbø.
    Modified by: GaffaSnobb.
    
    TODO: Figure out the pre-factors.
    TODO: Use numpy.logical_or to filter levels and transitions to avoid
    TODO: Make res.transitions_BXL.ji, res.transitions_BXL.pii, etc.
    class attributes (properties).
    using many if statements in the loops.

    Parameters
    ----------
    levels : np.ndarray
        Array containing energy, spin, and parity for each excited
        state. [[E, 2*spin, parity, idx], ...]. idx counts how many
        times a state of that given spin and parity has occurred. The
        first 0+ state will have an idx of 1, the second 0+ will have an
        idx of 2, etc.

    transitions : np.ndarray
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

    bin_width : Union[float, int]
        The width of the energy bins. A bin width of 0.2 contains 20
        states of uniform spacing of 0.01.

    Ex_min : Union[float, int]
        Lower limit for initial level excitation energy, usually in MeV.

    Ex_max : Union[float, int]
        Upper limit for initial level excitation energy, usually in MeV.

    multipole_type : str
        Choose whether to calculate for 'E1', 'M1' or 'E2'. NOTE:
        Currently only M1 and E1 is implemented.

    prefactor_E1 : Union[None, float]
        E1 pre-factor from the definition of the GSF. Defaults to a
        standard value if None.

    prefactor_M1 : Union[None, float]
        M1 pre-factor from the definition of the GSF. Defaults to a
        standard value if None.

    prefactor_E2 : Union[None, float]
        E2 pre-factor from the definition of the GSF. Defaults to a
        standard value if None.

    initial_or_final : str
        Choose whether to use the energy of the initial or final state
        for the transition calculations. NOTE: This may be removed in
        a future release since the correct alternative is to use the
        initial energy.

    partial_or_total : str
        Choose whether to use the partial level density
        rho(E_i, J_i, pi_i) or the total level density rho(E_i) for
        calculating the gamma strength function. Note that the partial
        level density, the default value, is probably the correct
        alternative. Using the total level density will introduce an
        arbitrary scaling depending on how many (J, pi) combinations
        were included in the calculations.

        This argument is included for easy comparison between the two
        densities. See the appendix of PhysRevC.98.064321 for details.

    include_only_nonzero_in_average : bool
        If True (default) only non-zero values are included in the final
        averaging of the gamma strength function. The correct
        alternative is to use only the non-zero values, so setting this
        parameter to False should be done with care.

    include_n_levels : Union[None, int]
        The number of states per spin to include. Example:
        include_n_levels = 100 will include only the 100 lowest laying
        states for each spin.

    filter_spins : Union[None, list]
        Which spins to include in the GSF. If None, all spins are
        included. TODO: Make int valid input too.

    filter_parities : str
        Which parities to include in the GSF. 'both', '+', '-' are
        allowed.

    return_n_transitions : bool
        Count the number of transitions, as a function of gamma energy,
        involved in the GSF calculation and return this number as a
        third return value. For calculating Porter-Thomas fluctuations
        in the GSF by

            r(E_gamma) = sqrt(2/n(E_gamma))

        where n is the number of transitions for each gamma energy, used
        to calculate the GSF. The value n is called n_transitions_array
        in the code. See for example DOI: 10.1103/PhysRevC.98.054303 for
        details.

    plot : bool
        Toogle plotting on / off.

    save_plot : bool
        Toogle saving of plot (as .png with dpi=300) on / off.

    Variables
    ---------
    Ex : np.ndarray 
        The excitation energy of all levels.

    Ex_initial : np.ndarray
        The excitation energy of the initial state of a transition.

    spins : np.ndarray
        The spins of all levels.

    parities : np.ndarray
        The parities of all levels.

    Returns
    -------
    bins : np.ndarray
        The bins corresponding to gSF_ExJpiavg (x values for plot).
        
    gSF_ExJpiavg : np.ndarray
        The gamma strength function.
    """
    skip_counter = {    # Debug.
        "Transit: Energy range": 0,
        "Transit: Number of levels": 0,
        "Transit: Parity": 0,
        "Level density: Energy range": 0,
        "Level density: Number of levels": 0,
        "Level density: Parity": 0
    }
    total_gsf_time = time.perf_counter()

    allowed_filter_parities = ["+", "-", "both"]
    if filter_parities not in allowed_filter_parities:
        msg = f"filter_parities must be {allowed_filter_parities}"
        raise TypeError(msg)
    if filter_parities == "both":
        filter_parities = [-1, +1]
    elif filter_parities == "-":
        filter_parities = [-1]
    elif filter_parities == "+":
        filter_parities = [+1]

    if include_n_levels is None:
        include_n_levels = np.inf   # Include all states.

    if (Ex_min < 0) or (Ex_max < 0):
        msg = "Ex_min and Ex_max cannot be negative!"
        raise ValueError(msg)

    if Ex_max < Ex_min:
        msg = "Ex_max cannot be smaller than Ex_min!"
        raise ValueError(msg)

    prefactors = {   # Factor from the def. of the GSF.
        "M1": 11.5473e-9, # [1/(mu_N**2*MeV**2)].
        # "E1": 1.047e-6,
        "E1": 3.4888977e-7
    }
    if prefactor_E1 is not None:
        """
        Override the E1 prefactor.
        """
        prefactors["E1"] = prefactor_E1
    
    if prefactor_M1 is not None:
        """
        Override the M1 prefactor.
        """
        prefactors["M1"] = prefactor_M1
    
    if prefactor_E2 is not None:
        """
        Override the E2 prefactor.
        """
        prefactors["E2"] = prefactor_E2
    
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
        msg += " load_and_save_to_file = 'overwrite' to re-read data from the"
        msg += " summary file and generate new tmp files."
        raise Exception(msg) from err
    
    if initial_or_final == "initial":
        Ex_initial_or_final = np.copy(transitions[:, 3])   # To avoid altering the raw data.
        spin_initial_or_final_idx = 0
        parity_initial_or_final_idx = 1
    elif initial_or_final == "final":
        Ex_initial_or_final = np.copy(transitions[:, 7])   # To avoid altering the raw data.
        spin_initial_or_final_idx = 4
        parity_initial_or_final_idx = 5
        msg = "Using final states for the energy limits is not correct"
        msg += " and should only be used for comparison with the correct"
        msg += " option which is using initial states for the energy limits."
        warnings.warn(msg, RuntimeWarning)
    else:
        msg = "'initial_or_final' must be either 'initial' or 'final'."
        msg += f" Got {initial_or_final}"
        raise ValueError(msg)

    if abs(Ex_initial_or_final[0]) > 10:
        """
        Adjust energies relative to the ground state energy if they have
        not been adjusted already. The ground state energy is usually
        minus a few tens of MeV and above, so checking absolute value
        above 10 MeV is probably safe. Cant check for equality to zero
        since the initial state will never be zero.
        NOTE: Just check if the value is negative instead?
        """
        Ex_initial_or_final -= E_ground_state

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
    # Ex_max = bin_width*n_bins # Adjust Ex_max to match the round-off in the bin width. NOTE: Unsure if this is needed.

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
    transit_gsf_time = time.perf_counter()
    
    for transition_idx in range(n_transitions):
        """
        Iterate over all transitions in the transitions matrix and add
        up all reduced transition probabilities and the number of
        transitions in the correct bins.
        """
        if (Ex_initial_or_final[transition_idx] < Ex_min) or (Ex_initial_or_final[transition_idx] >= Ex_max):
            """
            Check if transition is within min max limits, skip if not.
            """
            skip_counter["Transit: Energy range"] += 1   # Debug.
            continue

        idx_initial = transitions[transition_idx, 2]
        idx_final = transitions[transition_idx, 6]

        if (idx_initial > include_n_levels) or (idx_final > include_n_levels):
            """
            Include only 'include_n_levels' number of levels. Defaults
            to np.inf (include all).
            """
            skip_counter["Transit: Number of levels"] += 1   # Debug.
            continue

        spin_initial = transitions[transition_idx, 0]/2
        spin_final = transitions[transition_idx, 4]/2

        if filter_spins is not None:
            # if (spin_initial not in filter_spins) or (spin_final not in filter_spins):
            if spin_initial not in filter_spins:
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

        if (parity_initial not in filter_parities) or (parity_final not in filter_parities):
            """
            Skip initial or final parities which are not in the filter
            list. NOTE: Might be wrong to filter on the final parity.
            """
            skip_counter["Transit: Parity"] += 1
            continue

        # Get bin index for E_gamma and Ex. Indices are defined with respect to the lower bin edge.
        E_gamma_idx = int(transitions[transition_idx, 8]/bin_width)
        Ex_initial_or_final_idx = int(Ex_initial_or_final[transition_idx]/bin_width)
        n_transitions_array[E_gamma_idx] += 1    # Count the number of transitions involved in this GSF (Porter-Thomas fluctuations).
        """
        transitions : np.ndarray
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
        """
        spin_initial_or_final = int(transitions[transition_idx, spin_initial_or_final_idx])  # Superfluous int casts?
        parity_initial_or_final = int(transitions[transition_idx, parity_initial_or_final_idx])
        spin_parity_idx = spin_parity_list.index([spin_initial_or_final, parity_initial_or_final])

        try:
            """
            Add B(..) value and increment transition count,
            respectively. NOTE: Hope to remove this try-except by
            implementing suitable input checks to this function. Note to
            the note: Will prob. not be removed to keep the ability to
            compare initial and final.
            """
            B_pixel_sum[Ex_initial_or_final_idx, E_gamma_idx, spin_parity_idx] += \
                transitions[transition_idx, 9]
            B_pixel_count[Ex_initial_or_final_idx, E_gamma_idx, spin_parity_idx] += 1
        except IndexError as err:
            """
            NOTE: This error usually occurs because Ex_max is set to
            limit Ex_final instead of Ex_initial. If so, E_gamma might
            be larger than Ex_max and thus be placed in a B_pixel
            outside of the allocated scope. This error has a larger
            probability of occuring if Ex_max is set to a low value
            because then the probability of 
                
                E_gamma = Ex_initial - Ex_final

            is larger.
            """
            msg = f"{err.__str__()}\n"
            msg += f"{Ex_initial_or_final_idx=}, {E_gamma_idx=}, {spin_parity_idx=}, {transition_idx=}\n"
            msg += f"{B_pixel_sum.shape=}\n"
            msg += f"{transitions.shape=}\n"
            msg += f"{Ex_max=}\n"
            msg += f"2*spin_final: {transitions[transition_idx, 4]}\n"
            msg += f"parity_initial: {transitions[transition_idx, 1]}\n"
            msg += f"Ex_final: {transitions[transition_idx, 7]}\n"
            msg += f"2*spin_initial: {transitions[transition_idx, 0]}\n"
            msg += f"parity_initial: {transitions[transition_idx, 1]}\n"
            msg += f"Ex_initial: {transitions[transition_idx, 3]}\n"
            msg += f"E_gamma: {transitions[transition_idx, 8]}\n"
            msg += f"B(.., i->f): {transitions[transition_idx, 9]}\n"
            msg += f"B(.., f<-i): {transitions[transition_idx, 10]}\n"
            raise Exception(msg) from err

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

    if partial_or_total == "total":
        """
        Use the total level density, rho(E_i), instead of the partial
        level density, rho(E_i, J_i, pi_i). Sum over all (J_i, pi_i)
        pairs and then copy these summed values to all columns in
        rho_ExJpi.
        """
        tmp_sum = rho_ExJpi.sum(axis=1)

        for i in range(rho_ExJpi.shape[1]):
            """
            All columns in rho_ExJpi will be identical. This is for
            compatibility with the following for loop.
            """
            rho_ExJpi[:, i] = tmp_sum

        msg = "Using the total level density is not correct and"
        msg += " should only be used when comparing with the correct"
        msg += " alternative which is using the partial level density."
        warnings.warn(msg, RuntimeWarning)

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

    if include_only_nonzero_in_average:
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
    else:
        """
        NOTE: Probably not necessary to set an upper limit on gSF
        due to the input adjustment of Ex_max.
        """
        gSF_ExJpiavg = gSF[Ex_min_idx:Ex_max_idx + 1, :, :].mean(axis=(0, 2))
        msg = "Including non-zero values when averaging the gamma strength"
        msg += " function is not correct and should be used with care!"
        warnings.warn(msg, RuntimeWarning)
    
    avg_gsf_time = time.perf_counter() - avg_gsf_time

    bins = np.linspace(0, Ex_max, n_bins + 1)
    bins = (bins[:-1] + bins[1:])/2   # Middle point of the bins.
    bins = bins[:len(gSF_ExJpiavg)]

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
        # print(f"{skip_counter = }")
        print(f"{transit_total_skips = }")
        print(f"{n_transitions = }")
        print(f"{n_transitions_included = }")
        print(f"{level_density_total_skips = }")
        print(f"{n_levels = }")
        print(f"{n_levels_included = }")
        print("--------------------------------")

    if return_n_transitions:
        return bins, gSF_ExJpiavg, n_transitions_array
    else:
        return bins, gSF_ExJpiavg

def level_plot(
    levels: np.ndarray,
    include_n_levels: int = 1_000,
    filter_spins: Union[None, list] = None,
    ax: Union[None, plt.Axes] = None
    ):
    """
    Generate a level plot for a single isotope. Spin on the x axis,
    energy on the y axis.

    Parameters
    ----------
    levels : np.ndarray
        NxM array of [[energy, spin, parity], ...]. This is the instance
        attribute 'levels' of ReadKshellOutput.
    
    include_n_levels : int
        The maximum amount of states to plot for each spin. Default set
        to a large number to indicate ≈ no limit.

    filter_spins : Union[None, list]
        Which spins to include in the plot. If None, all spins are
        plotted.

    ax : Union[None, plt.Axes]
        matplotlib Axes to plot on. If None, plt.Figure and plt.Axes is
        generated in this function.
    """
    ax_input = False if (ax is None) else True

    if levels[0, 0] != 0:
        """
        Adjust energies relative to the ground state energy.
        """
        energies = levels[:, 0] - levels[0, 0]
    else:
        energies = levels[:, 0]

    spins = levels[:, 1]/2  # levels[:, 1] is 2*spin.
    parity_symbol = "+" if levels[0, 2] == 1 else "-"
    
    if filter_spins is not None:
        spin_scope = np.unique(filter_spins)    # x values for the plot.
    else:
        spin_scope = np.unique(spins)
    
    counts = {} # Dict to keep tabs on how many states of each spin have been plotted.
    line_width = np.abs(spins[0] - spins[1])/4*0.9

    if not ax_input:
        fig, ax = plt.subplots()

    for i in range(len(energies)):
        if filter_spins is not None:
            if spins[i] not in filter_spins:
                """
                Skip spins which are not in the filter.
                """
                continue

        try:
            counts[spins[i]] += 1
        except KeyError:
            counts[spins[i]] = 1
        
        if counts[spins[i]] > include_n_levels:
            """
            Include only the first 'include_n_levels' amount of states
            for any of the spins.
            """
            continue

        ax.hlines(
            y = energies[i],
            xmin = spins[i] - line_width,
            xmax = spins[i] + line_width,
            color = "black"
        )

    ax.set_xticks(spin_scope)
    ax.set_xticklabels([f"{Fraction(i)}" + f"$^{parity_symbol}$" for i in spin_scope])
    ax.set_xlabel("Spin")
    ax.set_ylabel("E [MeV]")

    if not ax_input:
        plt.show()

def level_density(
    levels: np.ndarray,
    bin_width: Union[int, float],
    include_n_levels: Union[None, int] = None,
    filter_spins: Union[None, int, list] = None,
    filter_parity: Union[None, str, int] = None,
    E_min: Union[None, float, int] = None,
    E_max: Union[None, float, int] = None,
    plot: bool = False,
    save_plot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the level density for a given bin size.

    Parameters
    ----------
    levels : Union[np.ndarray, list]
        Nx4 array of [[E, 2*spin, parity, idx], ...] or 1D array / list
        of only energies.

    bin_width : Union[int, float]
        Energy interval of which to calculate the density.

    include_n_levels : Union[None, int]
        The number of states per spin to include. Example:
        include_n_levels = 100 will include only the 100 lowest laying
        states for each spin.

    filter_spins : Union[None, int, list]
        Keep only the levels which have angular momenta in the filter.
        If None, all angular momenta are kept. Input must be the actual
        angular momenta values and not 2*j.

    filter_parity : Union[None, str, int]
        Keep only levels of parity 'filter_parity'. +1, -1, '+', '-'
        allowed inputs.

    E_min : Union[None, float, int]
        Minimum energy to include in the calculation. If None, the
        minimum energy in the levels array is used.

    E_max : Union[None, float, int]
        Maximum energy to include in the calculation. If None, the
        maximum energy in the levels array is used.

    plot : bool
        For toggling plotting on / off.

    save_plot : bool    
        Toogle saving of plot (as .png with dpi=300) on / off.

    Returns
    -------
    bins : np.ndarray
        The corresponding bins (x value for plotting).

    density : np.ndarray
        The level density.

    Raises
    ------
    ValueError:
        If any filter is given when energy_levels is a list of only
        energy levels.

    TypeError:
        If input parameters are of the wrong type.
    """
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

    if not isinstance(E_min, (type(None), int, float)):
        msg = f"'E_min' must be of type: None, int, float. Got {type(E_min)}."
        raise TypeError(msg)

    if not isinstance(E_max, (type(None), int, float)):
        msg = f"'E_max' must be of type: None, int, float. Got {type(E_max)}."
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

    if levels.ndim == 1:
        """
        'levels' only contain energy values.
        """
        energy_levels = levels
    else:
        """
        'levels' is a multidimensional array on the form
        [[E, 2*spin, parity, idx], ...].
        """
        energy_levels = np.copy(levels) # Copy just in case.
        
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

    if levels.ndim == 1:
        """
        Decide the max value of the energy bins.
        """
        if levels[0] != 0:
            """
            Calculate energies relative to the ground state if not already
            done.
            """
            energy_levels -= energy_levels[0]
            bin_max = levels[-1] - levels[0]    # The max energy of the un-filtered data set.
        else:
            bin_max = levels[-1]

    else:
        """
        Decide the max value of the energy bins.
        """
        if levels[0, 0] != 0:
            """
            Calculate energies relative to the ground state if not already
            done.
            """
            energy_levels -= energy_levels[0]
            bin_max = levels[-1, 0] - levels[0, 0]    # The max energy of the un-filtered data set.
        else:
            bin_max = levels[-1, 0]

    if E_min is not None:
        energy_levels = energy_levels[energy_levels >= E_min]

    if E_max is not None:
        energy_levels = energy_levels[energy_levels <= E_max]

    bins = np.arange(0, bin_max + bin_width, bin_width)
    n_bins = len(bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins - 1):
        counts[i] = np.sum(bins[i] <= energy_levels[energy_levels < bins[i + 1]])
    
    density = (counts/bin_width)[:-1]
    bins = bins[1:]

    if plot:
        fig, ax = plt.subplots()
        ax.step(bins, density, color="black")
        ax.set_ylabel(r"Density [MeV$^{-1}$]")
        ax.set_xlabel("E [MeV]")
        ax.legend([f"{bin_width=} MeV"])
        ax.grid()
        if save_plot:
            fname = "nld.png"
            print(f"NLD saved as '{fname}'")
            fig.savefig(fname=fname, dpi=300)
        plt.show()

    return bins, density

def porter_thomas(
    transitions: np.ndarray,
    Ei: Union[int, float, list],
    BXL_bin_width: Union[int, float],
    j_list: Union[list, None] = None,
    Ei_bin_width: Union[int, float] = 0.1,
    return_chi2: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Calculate the distribution of B(XL)/mean(B(XL)) values scaled to
    a chi-squared distribution of 1 degree of freedom.

    Parameters
    ----------
    transitions : np.ndarray
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

    return_chi2 : bool
        If True, the chi-squared distribution y values will be returned
        as a third return value.

    Returns
    -------
    BXL_bins : np.ndarray
        The BXL bins (x values).

    BXL_counts : np.ndarray
        The number of counts in each BXL_bins bin (y values).

    rv.pdf(BXL_bins) : np.ndarray
        The chi-squared distribution y values.
    """
    pt_prepare_data_time = time.perf_counter()
    if isinstance(Ei, (list, tuple, np.ndarray)):
        """
        If Ei defines a lower and an upper limit.
        """
        Ei_mask = np.logical_and(
            transitions[:, 3] >= Ei[0],
            transitions[:, 3] < Ei[-1]
        )
        BXL = transitions[Ei_mask]
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

    for pii in pii_masks:
        for idxi in idxi_masks:
            for ji in ji_masks:
                mask = np.logical_and(ji, np.logical_and(pii, idxi))
                tmp = BXL[mask][:, 9]   # 9 is B decay.
                if not tmp.size:
                    """
                    Some combinations of masks might not match any
                    levels.
                    """
                    continue

                BXL_tmp.extend(tmp/tmp.mean())

    BXL = np.asarray(BXL_tmp)
    BXL.sort()
    # BXL = BXL/np.mean(BXL)
    n_BXL_after = len(BXL)

    if n_BXL_before != n_BXL_after:
        msg = "The number of BXL values has changed during the Porter-Thomas analysis!"
        msg += f" This should not happen! {n_BXL_before = }, {n_BXL_after = }."
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

    pt_post_process_time = time.perf_counter()
    rv = chi2(1)
    BXL_counts = BXL_counts[1:] # Exclude the first data point because chi2(1) goes to infinity and is hard to work with there.
    BXL_bins = BXL_bins[1:]
    n_BXL_bins -= 1
    # BXL_counts_normalised = BXL_counts/np.trapz(BXL_counts)  # Normalize counts.
    # popt, _ = curve_fit(
    #     f = lambda x, scale: scale*rv.pdf(x),
    #     xdata = BXL_bins,
    #     ydata = BXL_counts,
    #     p0 = [rv.pdf(BXL_bins)[1]/BXL_counts[1]],
    #     method = "lm",
    # )
    # BXL_counts *= popt[0]   # Scale counts to match chi2.
    # BXL_counts_normalised *= np.mean(rv.pdf(BXL_bins)[1:20]/BXL_counts_normalised[1:20])
    """
    Normalise BXL_counts to the chi2(1) distribution, ie. find a
    coefficient which makes BXL_counts become chi2(1) by
    BXL_counts*chi2(1)/BXL_counts = chi2(1). Since any single point in
    the BXL distribution might over or undershoot chi2(1), I have chosen
    to use the mean of 19 ([1:20] slice, pretty arbitrary chosen) of
    these values to make a more stable normalisation coefficient.
    """
    BXL_counts *= np.mean(rv.pdf(BXL_bins)[1:20]/BXL_counts[1:20])
    pt_post_process_time = time.perf_counter() - pt_post_process_time

    if flags["debug"]:
        print("--------------------------------")
        print(f"Porter-Thomas: Prepare data time: {pt_prepare_data_time:.3f} s")
        print(f"Porter-Thomas: Count time: {pt_count_time:.3f} s")
        print(f"Porter-Thomas: Post process time: {pt_post_process_time:.3f} s")
        print(f"{sum(BXL_counts) = }")
        print(f"{Ei = }")
        print(f"{Ei_bin_width = }")
        print("--------------------------------")

    if return_chi2:
        # return BXL_bins, BXL_counts_normalised, rv.pdf(BXL_bins)
        return BXL_bins, BXL_counts, rv.pdf(BXL_bins)
    else:
        # return BXL_bins, BXL_counts_normalised
        return BXL_bins, BXL_counts

def nuclear_shell_model():
    """
    Generate a diagram of the nuclear shell model shell structure.
    """
    plt.rcParams.update({
        "backend": "pgf",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["roman"],
        "legend.fontsize": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
    })
    fig, ax = plt.subplots(figsize=(6.4, 8))
    ax.axis(False)
    fontsize = 15
    x_offset = 0.6
    x_text_offset = x_offset - 0.5

    first_layer_labels = [
        r"$1s$", r"$1p$", r"$1d$", r"$2s$", r"$1f$", r"$2p$", r"$1g$",
        r"$2d$", r"$3s$"
    ]
    first_layer_y = [1, 2.4, 4.2, 4.45, 6.3, 6.8, 9, 10.0, 10.5]
    second_layer_labels = [
        r"$1s_{1/2}$", r"$1p_{3/2}$", r"$1p_{1/2}$", r"$1d_{5/2}$",
        r"$2s_{1/2}$", r"$1d_{3/2}$", r"$1f_{7/2}$", r"$2p_{3/2}$",
        r"$1f_{5/2}$", r"$2p_{1/2}$", r"$1g_{9/2}$", r"$2d_{5/2}$",
        r"$1g_{7/2}$", r"$3s_{1/2}$", r"$2d_{3/2}$"
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
    core_layer_labels = [
        r"$^{16}$O", r"$^{40}$Ca", r"$^{56}$Ni"
    ]
    core_layer_y = [
        second_layer_y[2] + 0.5, second_layer_y[5] + 0.5, second_layer_y[6] + 0.5
    ]

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

    for y, label in zip(core_layer_y, core_layer_labels):
        fig.text(
            x = 0.77 + x_text_offset,
            y = y/14 + 0.067,
            s = label,
            fontsize = fontsize - 1
        )
        fig.text(
            x = 0.73 + x_text_offset,
            y = y/14 + 0.064,
            s = "---------",
            fontsize = fontsize - 1
        )

    # USD
    x1 = 1.35
    x2 = 1.25
    y1 = 4.9
    y2 = 3.83
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
    ax.vlines(
        x = x6 - 0.04,
        ymin = y2,
        ymax = 11,
        color = "mediumorchid",
    )
    fig.text(
        x = 0.15,
        y = 0.66,
        s = "SDPF-SDG",
        fontsize = 12,
        rotation = "vertical",
        color = "mediumorchid"
    )
    # Spectroscopic notation
    fig.text(
        x = 0.45,
        y = 0.93,
        s = r"$s \;\; p \;\; d \;\; f \;\; g \;\; h$",
        fontsize = fontsize - 1,
        color = "black",
    )
    fig.text(
        x = 0.45,
        y = 0.92,
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

    fig.savefig(fname="nuclear_shell_model.png", dpi=500)#, format="eps")
    plt.show()