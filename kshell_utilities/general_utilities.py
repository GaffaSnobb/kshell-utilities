import sys, time
from typing import Union, Tuple
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt

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
    multipole_type: str = "M1",
    initial_or_final: str = "initial",
    plot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the gamma strength function averaged over spins and
    parities.
    
    Author: Jørgen Midtbø.
    Modified by: GaffaSnobb.

    TODO: Ex_final or Ex_initial?
    TODO: Figure out the pre-factors.

    Parameters
    ----------
    levels : np.ndarray
        Array containing energy, spin, and parity for each excited
        state. [[E, 2*spin, parity], ...].

    transitions : np.ndarray
        OLD:
        Mx8 array containing [2*spin_final, parity_initial, Ex_final,
        2*spin_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)]

        NEW:
        [2*spin_initial, parity_initial, Ex_initial, 2*spin_final,
        parity_final, Ex_final, E_gamma, B(.., i->f), B(.., f<-i)]

    bin_width : Union[float, int]
        The width of the energy bins. A bin width of 0.2 contains 20
        states of uniform spacing of 0.01.

    Ex_min : Union[float, int]
        Lower limit for initial level excitation energy, usually in MeV.

    Ex_max : Union[float, int]
        Upper limit for initial level excitation energy, usually in MeV.

    multipole_type : str
        Choose whether to calculate for 'E1', 'M1' or 'E2'. NOTE:
        Currently only M1 is implemented.

    initial_or_final : str
        Choose whether to use the energy of the initial or final state
        for the transition calculations. NOTE: This will be removed in
        a future release since the correct alternative is to use the
        initial energy.

    plot : bool
        Toogle plotting on / off.

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
    if (Ex_min < 0) or (Ex_max < 0):
        msg = "Ex_min and Ex_max cannot be negative!"
        raise ValueError(msg)

    if Ex_max < Ex_min:
        msg = "Ex_max cannot be smaller than Ex_min!"
        raise ValueError(msg)

    prefactors = {   # Factor from the def. of the GSF.
        "M1": 11.5473e-9, # [1/(mu_N**2*MeV**2)].
        "E1": 1.047e-6
    }
    prefactor = prefactors[multipole_type]

    # Extract data to a more readable form:
    n_transitions = len(transitions[:, 0])
    n_levels = len(levels[:, 0])
    E_ground_state = levels[0, 0] # Read out the absolute ground state energy so we can get relative energies later.
    Ex, spins, parities = np.copy(levels[:, 0]), levels[:, 1], levels[:, 2]
    
    if initial_or_final == "initial":
        Ex_initial_or_final = np.copy(transitions[:, 2])   # To avoid altering the raw data.
        spin_initial_or_final_idx = 0
        parity_initial_or_final_idx = 1
    elif initial_or_final == "final":
        """
        NOTE: This option will be removed in a future release.
        """
        Ex_initial_or_final = np.copy(transitions[:, 5])   # To avoid altering the raw data.
        spin_initial_or_final_idx = 3
        parity_initial_or_final_idx = 4
    else:
        msg = "'initial_or_final' must be either 'initial' or 'final'."
        msg += f" Got {initial_or_final}"
        raise ValueError(msg)
        
    if abs(Ex_initial_or_final[0]) > 10:
        """
        Adjust energies relative to the ground state energy if they have
        not been adjusted already. The ground state energy is usually
        ~ -100 MeV so checking absolute value above 10 MeV is probably
        safe. Cant check for equality to zero since the initial state
        will never be zero.
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
            continue

        # Get bin index for E_gamma and Ex. Indices are defined with respect to the lower bin edge.
        E_gamma_idx = int(transitions[transition_idx, 6]/bin_width)
        Ex_initial_or_final_idx = int(Ex_initial_or_final[transition_idx]/bin_width)

        """
        transitions : np.ndarray
            OLD:
            Mx8 array containing [2*spin_final, parity_initial, Ex_final,
            2*spin_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)]
            NEW:
            [2*spin_initial, parity_initial, Ex_initial, 2*spin_final,
            parity_final, Ex_final, E_gamma, B(.., i->f), B(.., f<-i)]
        """
        spin_initial = int(transitions[transition_idx, spin_initial_or_final_idx])
        parity_initial = int(transitions[transition_idx, parity_initial_or_final_idx])
        spin_parity_idx = spin_parity_list.index([spin_initial, parity_initial])

        try:
            """
            Add B(..) value and increment transition count,
            respectively. NOTE: Hope to remove this try-except by
            implementing suitable input checks to this function.
            """
            B_pixel_sum[Ex_initial_or_final_idx, E_gamma_idx, spin_parity_idx] += \
                transitions[transition_idx, 7]
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
            msg += f"2*spin_final: {transitions[transition_idx, 3]}\n"
            msg += f"parity_initial: {transitions[transition_idx, 1]}\n"
            msg += f"Ex_final: {transitions[transition_idx, 5]}\n"
            msg += f"2*spin_initial: {transitions[transition_idx, 0]}\n"
            msg += f"parity_initial: {transitions[transition_idx, 1]}\n"
            msg += f"Ex_initial: {transitions[transition_idx, 2]}\n"
            msg += f"E_gamma: {transitions[transition_idx, 6]}\n"
            msg += f"B(.., i->f): {transitions[transition_idx, 7]}\n"
            msg += f"B(.., f<-i): {transitions[transition_idx, 8]}\n"
            raise Exception(msg) from err

    for levels_idx in range(n_levels):
        """
        Calculate the level density for each (Ex, spin_parity) pixel.
        """
        if Ex[levels_idx] >= Ex_max:
            """
            Skip if level is outside range. Only upper limit since
            decays to states below the lower limit are allowed.
            """
            continue

        Ex_idx = int(Ex[levels_idx]/bin_width)

        spin_parity_idx = \
            spin_parity_list.index([spins[levels_idx], parities[levels_idx]])
        
        rho_ExJpi[Ex_idx, spin_parity_idx] += 1

    rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1.

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

    # Return the average gSF(Eg) over all (Ex,J,parity_initial)

    # return gSF[Ex_min_idx:Ex_max_idx+1,:,:].mean(axis=(0,2))
    # Update 20171009: Took proper care to only average over the non-zero f(Eg,Ex,J,parity_initial) pixels:
    gSF_currentExrange = gSF[Ex_min_idx:Ex_max_idx + 1, :, :]   # NOTE: Probably not necessary to set an upper limit here due to the input adjustment of Ex_max.
    gSF_ExJpiavg = div0(
        numerator = gSF_currentExrange.sum(axis = (0, 2)),
        denominator = (gSF_currentExrange != 0).sum(axis = (0, 2))  # NOTE: Here is where I left off! Why != 0?
    )

    bins = np.linspace(0, Ex_max, n_bins + 1)
    bins = (bins[:-1] + bins[1:])/2   # Middle point of the bins.
    bins = bins[:len(gSF_ExJpiavg)]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(bins, gSF_ExJpiavg)
        ax.set_xlabel("bins")
        ax.set_ylabel("gsf")
        plt.show()

    return bins, gSF_ExJpiavg

def level_plot(
    levels: np.ndarray,
    max_spin_states: int = 1_000,
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
    
    max_spin_states : int
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
        
        if counts[spins[i]] > max_spin_states:
            """
            Include only the first 'max_spin_states' amount of states
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
    energy_levels: Union[np.ndarray, list],
    bin_size: Union[int, float],
    plot: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the level density for a given bin size.

    Parameters
    ----------
    energy_levels : Union[np.ndarray, list]
        1D array of energy levels.

    bin_size : Union[int, float]
        Energy interval of which to calculate the density.

    plot : bool
        For toggling plotting on / off.

    Returns
    -------
    bins : np.ndarray
        The corresponding bins (x value for plotting).

    density : np.ndarray
        The level density.
    """
    if isinstance(energy_levels, list):
        energy_levels = np.array(energy_levels)

    if len(energy_levels.shape) != 1:
        msg = "'energy_levels' input to 'level_density' must be a 1D array or"
        msg += " list containing the energies for the different levels."
        raise ValueError(msg)

    if energy_levels[0] != 0:
        """
        Calculate energies relative to the ground state if not already
        done.
        """
        energy_levels = energy_levels - energy_levels[0]

    bins = np.arange(0, energy_levels[-1] + bin_size, bin_size)
    n_bins = len(bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins - 1):
        counts[i] = np.sum(bins[i] <= energy_levels[energy_levels < bins[i + 1]])
    
    density = (counts/bin_size)[:-1]
    bins = bins[1:]

    if plot:
        _, ax = plt.subplots()

        ax.step(bins, density)

        if plot:
            ax.set_ylabel("Density")
            ax.set_xlabel("Bins")
            ax.legend([f"{bin_size=}"])
            plt.show()

    return bins, density