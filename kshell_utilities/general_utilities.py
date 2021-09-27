from typing import Union, Tuple
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt

def create_spin_parity_list(
    spins: np.ndarray,
    parities: np.ndarray
    ) -> list:
    """
    Example list:
    [[1, +1], [3, +1], [5, +1], [7, +1], [9, +1], [11, +1], [13, +1]].

    Parameters
    ----------
    spins:
        Array of spins for each energy level.

    parities:
        Array of corresponding parities for each energy level.

    Returns
    -------
    spins_parities:
        A nested list of spins and parities [[spin, parity], ...] sorted
        with respect to the spin. N is the number of unique spins in
        'spins'.
    """
    unique_spins, unique_spins_idx = np.unique(spins, return_index=True)
    spins_parities = np.empty((len(unique_spins), 2))
    spins_parities[:, 0] = unique_spins
    spins_parities[:, 1] = parities[unique_spins_idx]

    return spins_parities.tolist()  # Convert to list since list.index is much faster than np.where.

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

def strength_function_average(
    levels: np.ndarray,
    transitions: np.ndarray,
    bin_width: Union[float, int],
    Ex_min: Union[float, int],
    Ex_max: Union[float, int],
    multipole_type: str = "M1",
    initial_or_final: str = "initial"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Author: Jørgen Midtbø.
    Modified by: Jon Dahl.
    Notes from Jørgen:
        20171009: Updated the way we average over Ex, J, parity_initial to only count pixels with non-zero gSF.
        20170815: This function returns the strength function the way we now think is the correct way:
        By taking only the partial level density corresponding to the specific (Ex, J, parity_initial) pixel in the
        calculation of the strength function, and then averaging over all three variables to produce
        <f(Eg)>.
        This code was first developed in the script strength_function_individual_Jpi.py

        # # Update 20170915: Realized a problem with summing vs averaging, adding this list to fix that
        # # Update 20170920: Realized that the fix was wrong, the original was correct.
        # Ex_already_seen = []
        # for i_Ex in range(Nbins):
        #     Ex_already_seen.append([])

        # 20170920: We thought this was more correct, but now think not.
        # if not Ex in Ex_already_seen[i_Eg]:
        #     B_pixel_count[i_Ex,i_Eg,spin_parity_idx] += 1
        #     Ex_already_seen[i_Eg].append(Ex)

    TODO: Ex_final or Ex_initial? Ask about this!
    TODO: Figure out the pre-factors.

    Parameters
    ----------
    levels:
        Array containing energy, spin, and parity for each excited
        state. [[E, 2*spin, parity], ...].

    transitions:
        Mx8 array containing [2*spin_final, parity_initial, Ex_final,
        2*spin_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)]

    bin_width:
        The width of the energy bins. A bin width of 0.2 contains 20
        states of uniform spacing of 0.01.

    Ex_min:
        Lower limit for emitted gamma energy [MeV].

    Ex_max:
        Upper limit for emitted gamma energy [MeV].

    multipole_type:
        Choose whether to calculate for 'M1' or 'E2'.

    initial_or_final:
        Choose whether to use the energy of the initial or final state
        for the transition calculations.

    Variables
    ---------
    Ex:
        The excitation energy of all levels.

    Ex_final:
        The excitation energy of the final state of a transition.

    Ex_initial:
        The excitation energy of the initial state of a transition.

    spins:
        The spins of all levels.

    parities:
        The parities of all levels.

    Returns
    -------
    gSF_ExJpiavg:
        The gamma strength function.

    bins:
        The bins corresponding to gSF_ExJpiavg (x values for plot).
    """
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
        Ex_initial_or_final = np.copy(transitions[:, 5])   # To avoid altering the raw data.
    elif initial_or_final == "final":
        Ex_initial_or_final = np.copy(transitions[:, 2])   # To avoid altering the raw data.
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
    Ex_min_idx = int(np.floor(Ex_min/bin_width)) 
    Ex_max_idx = int(np.floor(Ex_max/bin_width))    
    n_bins = int(np.ceil(Ex_max/bin_width)) # Make sure the number of bins cover the whole Ex region.
    Ex_max = bin_width*n_bins # Adjust Ex_max to match the round-off in the bin width. NOTE: Unsure if this is needed.

    """
    B_pixel_sum[Ex_final_idx, E_gamma_idx, spin_parity_idx] contains the
    summed reduced transition probabilities for all transitions
    contained within the Ex_final_idx bin, E_gamma_idx bin, and
    spin_parity_idx bin. B_pixel_counts counts the number of transitions
    within the same bins.
    """
    spin_parity_list = create_spin_parity_list(spins, parities)
    n_unique_spin_parity_pairs = len(spin_parity_list)
    B_pixel_sum = np.zeros((n_bins, n_bins, n_unique_spin_parity_pairs))     # Summed B(..) values for each pixel.
    B_pixel_count = np.zeros((n_bins, n_bins, n_unique_spin_parity_pairs))   # The number of transitions.
    rho_ExJpi = np.zeros((n_bins, n_unique_spin_parity_pairs))   # (Ex, Jpi) matrix to store level density
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
            NOTE: Is it correct to check the energy of the final state
            here? Why not the initial?
            """
            continue

        # Get bin index for Eg and Ex (initial). Indices are defined with respect to the lower bin edge.
        E_gamma_idx = int(np.floor(transitions[transition_idx, 6]/bin_width))
        Ex_final_idx = int(np.floor(Ex_initial_or_final[transition_idx]/bin_width))

        # Read initial spin and parity of level: NOTE: I think the name / index is wrong. Or do I...?
        spin_initial = int(transitions[transition_idx, 0])
        parity_initial = int(transitions[transition_idx, 1])
        try:
            """
            Get index for current [spin_initial, parity_initial]
            combination in spin_parity_list.
            """
            spin_parity_idx = spin_parity_list.index([spin_initial, parity_initial])
        except ValueError:
            print("Transition skipped due to lack of spin_parity_list.")
            continue

        try:
            """
            Add B(..) value and increment transition count,
            respectively. NOTE: Hope to remove this try-except by
            implementing suitable input checks to this function.
            """
            B_pixel_sum[Ex_final_idx, E_gamma_idx, spin_parity_idx] += \
                transitions[transition_idx, 7]
            B_pixel_count[Ex_final_idx, E_gamma_idx, spin_parity_idx] += 1
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
            msg += f"{Ex_final_idx=}, {E_gamma_idx=}, {spin_parity_idx=}, {transition_idx=}\n"
            msg += f"{B_pixel_sum.shape=}\n"
            msg += f"{transitions.shape=}\n"
            msg += f"{Ex_max=}\n"
            msg += f"2*spin_final: {transitions[transition_idx, 0]}\n"
            msg += f"parity_initial: {transitions[transition_idx, 1]}\n"
            msg += f"Ex_final: {transitions[transition_idx, 2]}\n"
            msg += f"2*spin_initial: {transitions[transition_idx, 3]}\n"
            msg += f"parity_initial: {transitions[transition_idx, 4]}\n"
            msg += f"Ex_initial: {transitions[transition_idx, 5]}\n"
            msg += f"E_gamma: {transitions[transition_idx, 6]}\n"
            msg += f"B(.., i->f): {transitions[transition_idx, 7]}\n"
            raise Exception(msg) from err

    for levels_idx in range(n_levels):
        """
        Count number of levels for each (Ex, J, parity_initial) pixel.
        """

        if Ex[levels_idx] > Ex_max:
            """
            Skip if level is outside range.
            """
            continue

        Ex_idx = int(np.floor(Ex[levels_idx]/bin_width))

        try:
            spin_parity_idx = \
                spin_parity_list.index([spins[levels_idx], parities[levels_idx]])
        except ValueError:
            print("Transition skipped due to lack of spin_parity_list.")
            continue
        
        rho_ExJpi[Ex_idx, spin_parity_idx] += 1

    rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1.

    for spin_parity_idx in range(n_unique_spin_parity_pairs):
        """
        Calculate gamma strength functions for each [Ex, spin,
        parity_initial] individually using the partial level density for
        each [spin, parity_initial].
        """
        for Ex_idx in range(n_bins):
            gSF[Ex_idx, :, spin_parity_idx] = \
                prefactor*rho_ExJpi[Ex_idx, spin_parity_idx]*div0(
                B_pixel_sum[Ex_idx, :, spin_parity_idx],
                B_pixel_count[Ex_idx, :, spin_parity_idx]
                )

    # Return the average gSF(Eg) over all (Ex,J,parity_initial)

    # return gSF[Ex_min_idx:Ex_max_idx+1,:,:].mean(axis=(0,2))
    # Update 20171009: Took proper care to only average over the non-zero f(Eg,Ex,J,parity_initial) pixels:
    gSF_currentExrange = gSF[Ex_min_idx:Ex_max_idx + 1, :, :]
    gSF_ExJpiavg = div0(
        gSF_currentExrange.sum(axis = (0, 2)),
        (gSF_currentExrange != 0).sum(axis = (0, 2))
    )

    bins = np.linspace(0, Ex_max, n_bins + 1)   # Not used in this function, only returned.
    bins = (bins[:-1] + bins[1:])/2   # Middle point of the bins.
    bins = bins[:len(gSF_ExJpiavg)]

    return gSF_ExJpiavg, bins

def level_plot(
    levels: np.ndarray,
    max_spin_states: int = 1_000,
    filter_spins: Union[None, list] = None
    ):
    """
    Generate a level plot for a single isotope. Spin on the x axis,
    energy on the y axis.

    TODO: fig, ax as argument to make plot which can later be adjusted.

    Parameters
    ----------
    levels:
        NxM array of [[energy, spin, parity], ...]. This is the instance
        attribute 'levels' of ReadKshellOutput.
    
    max_spin_states:
        The maximum amount of states to plot for each spin. Default set
        to a large number to indicate ≈ no limit.

    filter_spins:
        Which spins to include in the plot. If None, all spins are
        plotted.
    """
    energies = levels[:, 0] - levels[0, 0]  # Energies relative to the ground state energy.
    spins = levels[:, 1]/2  # levels[:, 1] is 2*spin.
    parity_symbol = "+" if levels[0, 2] == 1 else "-"
    
    if filter_spins is not None:
        spin_scope = np.unique(filter_spins)    # x values for the plot.
    else:
        spin_scope = np.unique(spins)
    
    counts = {} # Dict to keep tabs on how many states of each spin has been plotted.
    line_width = np.abs(spins[0] - spins[1])/2*0.9

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
    plt.show()