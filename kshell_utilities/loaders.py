from __future__ import annotations
import time, sys, ast, warnings, re
from fractions import Fraction
from typing import TextIO
import numpy as np
from numpy.typing import NDArray
from .kshell_exceptions import KshellDataStructureError
from .data_structures import (
    Interaction, Partition, OrbitalParameters, Configuration
)
from .parameters import (
    spectroscopic_conversion, shell_model_order, flags
)
from .partition_tools import (
    _calculate_configuration_parity, _sanity_checks, configuration_energy
)
WEISSKOPF_THRESHOLD: float = -0.001

def _weisskopf_unit(multipole_type: str, mass: int) -> float:
    """
    Generate the Weisskopf unit_weisskopf for input multipolarity and
    mass. Ref. Bohr and Mottelson, Vol.1, p. 389.

    Parameters
    ----------
    multipole_type : str
        The electromagnetic character and angular momentum of the gamma
        radiation. Examples: 'E1', 'M1', 'E2'.

    mass : int
        Mass of the nucleus.

    Returns
    -------
    B_weisskopf : float
        Reduced transition probability in the Weisskopf estimate.
    """
    l = int(multipole_type[1:])
    if multipole_type[0].upper() == "E":
        B_weisskopf = 1.2**(2*l)/(4*np.pi)*(3/(l + 3))**2*mass**(2*l/3)  
    
    elif multipole_type[0].upper() == "M":
        B_weisskopf = 10/np.pi*1.2**(2*l - 2)*(3/(l + 3))**2*mass**((2*l - 2)/3) 

    else:
        msg = f"Got invalid multipole type: '{multipole_type}'."
        raise ValueError(msg)
    
    return B_weisskopf

def _load_transition_logfile(
    path: str
):
    """
    Please note that the structure of the loaded data is a bit different
    from how it appears in the logfile.

    Structure when loaded:
    [2*spin_initial, parity_initial, idx_initial, Ex_initial,
    2*spin_final, parity_final, idx_final, Ex_final, E_gamma,
    B(.., i->f), B(.., f<-i)]
    """
    transitions_E1: list[list[float]] = []
    transitions_M1: list[list[float]] = []
    transitions_E2: list[list[float]] = []
    transitions: list[list[float]]

    with open(path, "r") as infile:
        for line in infile:
            if "left  Z,N,A,M,prty:" in line:
                """
                left  Z,N,A,M,prty:   23  27  50   2   1
                right Z,N,A,M,prty:   23  27  50   4  -1
                """
                tmp = line.split()
                pi_f = int(tmp[-1])
                j_f_expected = int(tmp[-2])
                mass = int(tmp[-3])

                tmp = infile.readline().split()
                pi_i = int(tmp[-1])
                j_i_expected = int(tmp[-2])

                if (pi_i == pi_f) and (j_f_expected == j_i_expected):
                    is_diag = True
                else:
                    is_diag = False
                
                break
    
    with open(path, "r") as infile:
        for _ in range(3):
            """
            Technically there should never be three tables (E1, M1, E2)
            in any logfile since a ji -> jf which supports M1 does not
            support E1. However, there might be the case where just the
            header and then an empty table is written to the file in
            which case some information might be skipped if we do not do
            this three times.
            """
            for line in infile:
                if ("E1 transition" in line) or ("M1 transition" in line) or ("E2 transition" in line):
                    multipolarity = line.split()[0]
                    infile.readline()   # Skip header.

                    if multipolarity == "E1":
                        transitions = transitions_E1
                    elif multipolarity == "M1":
                        transitions = transitions_M1
                    elif multipolarity == "E2":
                        transitions = transitions_E2
                    else:
                        msg = f"Invalid multipolarity from file '{path}'! Got '{multipolarity}'."
                        raise KshellDataStructureError(msg)
                    
                    B_weisskopf = _weisskopf_unit(
                        multipole_type = multipolarity,
                        mass = mass,
                    )
                    break
            
            for line in infile:
                tmp = line.split() # ['2', '1', '-393.605', '4', '1', '-392.585', '1.021', '0.00401764', '0.00000538', '0.00000323', '0.00000000']
                
                try:
                    j_f = int(tmp[0])
                except IndexError:
                    """
                    Reached blank line after the table of values.
                    """
                    break

                idx_f = int(tmp[1]) - 1
                E_f = float(tmp[2])
                j_i = int(tmp[3])
                idx_i = int(tmp[4]) - 1
                E_i = float(tmp[5])
                E_gamma = float(tmp[6])
                # M_red = float(tmp[7])   # Not in use.
                B_if = float(tmp[8])    # Decay.
                B_fi = float(tmp[9])    # Excite.
                # Mom = float(tmp[10])    # Not in use.

                B_weisskopf_if  = B_if/B_weisskopf
                B_weisskopf_fi = B_fi/B_weisskopf
                
                if (j_f == j_i) and (idx_f == idx_i): continue
                if is_diag and (E_gamma < 0): continue
                if (B_weisskopf_if < WEISSKOPF_THRESHOLD): continue # NOTE: I might not need the Weisskoppf stuff.
                if (B_weisskopf_fi < WEISSKOPF_THRESHOLD): continue
                if abs(E_f) < 1e-3: E_f = 0.
                if abs(E_i) < 1e-3: E_i = 0.

                assert j_f == j_f_expected
                assert j_i == j_i_expected

                if E_gamma < 0:
                    """
                    TODO: Ask about why this occurs. Why not remove these?
                    """
                    j_f, j_i = j_i, j_f
                    idx_f, idx_i = idx_i, idx_f
                    E_f, E_i = E_i, E_f
                    B_if, B_fi = B_fi, B_if
                    E_gamma = -E_gamma

                transitions.append([
                    j_i, pi_i, idx_i, E_i, j_f, pi_f, idx_f, E_f, E_gamma, B_if, B_fi
                ])

    return transitions_E1, transitions_M1, transitions_E2

def _load_energy_logfile(
    path: str
) -> NDArray:
    """
    Read KSHELL energy logfiles. NOTE: I needed to read additional info
    from the log files and decided it is easier to completely skip
    generating a summary file and just read straight from the log files.
    Everything is temporary stored as binary Numpy arrays for fast
    loading and it seemed unnecessary to do two steps of data
    restructuring.

    The data is stored in a 2D numpy array with each row as:
    ```
        [E, 2*spin, parity, idx, Hcm]
    ```
    `idx` will for each individual energy log file array be the same as
    the index and it will correspond to the energy order of which the
    levels appear. It is needed when several energy log file arrays are
    concatenated so that we can keep track of the energy order per j pi.
    """
    with open(path, "r") as infile:
        for line in infile:
            if "N_EIGEN" in line:
                """
                The number of energy eigenstates in this file is stored
                as:
                N_EIGEN =         200,
                """
                tmp = line.split("=")[-1]
                tmp = tmp.split(",")[0]
                tmp = tmp.strip()
                n_eigen = int(tmp)
                break

        for line in infile:
            if ("M =" in line) and ("parity =" in line):
                tmp = line.split()  # ['M', '=', '0/2', ':', 'parity', '=', '+']
                j_expected = int(Fraction(tmp[2])*2)
                parity_expected = +1 if (tmp[6] == "+") else -1
                break

        levels = np.zeros(shape=(n_eigen, 5), dtype=np.float64)
        idx_prev = -1    # Has to have some starting value.
        E_prev = -np.inf

        for line in infile:
            if "-------------------------------------------------" in line:
                tmp = infile.readline().split() # ['1', '<H>:', '-392.15904', '<JJ>:', '-0.00000', 'J:', '0/2', 'prty', '1']
                idx_current = int(tmp[0]) - 1
                E_current = float(tmp[2])
                j = int(Fraction(tmp[6])*2)
                parity = int(tmp[8])
                tmp = infile.readline().split() # ['<Hcm>:', '0.00000', '<TT>:', '6.00000', 'T:', '4/2']
                Hcm = float(tmp[1])

                assert j == j_expected
                assert parity == parity_expected
                assert idx_prev < idx_current
                assert E_prev < E_current

                levels[idx_current] = E_current, j, parity, idx_current, Hcm

                idx_prev = idx_current
                E_prev = E_current

            if idx_prev == (n_eigen - 1):
                """
                All energy eigenstates have been loaded.
                """
                break

    return levels

def _load_obtd(
    path: str,
    obtd_dict: dict[tuple[int, ...]],
) -> None:
    """
    Read one-body transition densities from the OBTD files from KSHELL.
    The OBTD is defined as
    ```
        \text{OBTD} = \langle \Psi_f | c_\alpha^\dagger c_\beta | \Psi_i \rangle.
    ```

    Take for example the file `OBTD_L_V50_GCLSTsdpfsdgix5pn_j0p_V50_GCLSTsdpfsdgix5pn_j2p.dat`.
    It contains OBTDs from all 0+ levels to all 1+ levels (remember that
    the angular momentum is stored as 2 times its value to avoid
    fractions). The OBTD values will be stored in a 3D NDArray. Each 2D
    slice in the array contains the OBTD data for one of the 0+ levels
    to one of the 1+ levels. The other 2D slices thus contain OBTD data
    for the other 0+ to 1+ levels. In the case of 200 0+ levels and 200
    1+ levels, there will be 200*200 2D slices in the array.

    The `obtd_dict` provides a view to the 2D slices based on the dict's
    keys which are explained below.

    Parameters
    ----------
    path : str
        Path (with filename) to the one-body transition density file.

    obtd_dict : dict[tuple[int, ...]]
        Each key is a tuple of `(j_i, pi_i, idx_i, j_f, pi_f, idx_f)`
        and each value is a view of the correct 2D slice of the `obtd`
        Numpy array which stores the OBTD values. A view of the entire
        3D matrix is also provided by the key (j_i, pi_i, j_f, pi_f).

    Variables
    ---------
    obtd : NDArray
        A 3D array. Each 2D slice has columns of the following values:

        ```
        i  j      OBTD    <i||L||j>  OBTD*<||>
        1  1     0.00000     5.79655     0.00000
        1  2     0.00000     1.54919     0.00000
        1  9     0.00000     0.00000     0.00000
        1 10     0.00000     0.00000     0.00000
        1 11     0.00000     0.00000     0.00000
        2  1     0.00000    -1.54919     0.00000
        ...
        ```

        I think that `i` and `j` are the same as `\alpha` and `\beta` in
        the definition of the OBTD, not completely sure yet. I do know
        that `i` and `j` refer to orbitals (not m substates) in the
        model space. The dim of the array will be (no. OBTDs per
        transition, 5, the number of 0+ levels times the number of 1+
        levels). In the case of the example file, the dim is exactly
        (92, 5, 200*200). NOTE: Currently, I removed the two final
        columns as I dont have any use for them yet, and if I store
        <i||L||j>, what about <i||S||j>?
    """

    timing = time.perf_counter()
    sect_header_pattern = r'J1=\s*(\d+)/2\s*\(\s*(\d+)\)\s*J2=\s*(\d+)/2\s*\(\s*(\d+)\)'
    filename_pattern = r'_j(\d+)(p|n)'
    filename_match = re.findall(filename_pattern, path.split("/")[-1])
    j_i = int(filename_match[0][0])
    pi_i = +1 if (filename_match[0][1] == "p") else -1
    j_f = int(filename_match[1][0])
    pi_f = +1 if (filename_match[1][1] == "p") else -1

    n_initial_levels = 0
    n_final_levels = 0
    with open(path, "r") as infile:
        """
        The file is first iterated once to extract some needed metadata
        about how many elements and how many transitions there are.
        """
        for line in infile:
            if "# of elements" in line:
                """
                This is the number of OBTDs per transition.
                """
                n_elements = int(line.split("=")[-1])
                break

        for line in infile:
            if "w.f." in line:
                """
                NOTE: I could seek to the end of the file and backtrack
                a bit instead of reading through the entire file just to
                get two numbers at the very end.

                Example:
                w.f.  J1=  0/2(    1)     J2=  2/2(    1)   <----- This one!
                B(L;=>), B(L ;<=)      0.05162     0.01721
                <||L||>      1    2     0.39870    -0.26442
                ...
                """
                match_ = re.search(sect_header_pattern, line)
                if not match_:
                    msg = "Unexpected pattern in OBTD file!"
                    raise RuntimeError(msg)
                
                _, n_initial_levels, _, n_final_levels = map(int, match_.groups())

    if (n_initial_levels == 0) or (n_final_levels == 0):
        """
        For example, in the file

        OBTD_E2_V50_GCLSTsdpfsdgix5pn_j0p_V50_GCLSTsdpfsdgix5pn_j2p.dat

        KSHELL has only written the header of the file, but no content.
        Make sure at no dict entry is made for key (0, +1, 2, +1) from
        this file because subsequent files with the same key might
        actually contain OBTD information which would have been skipped.
        """
        msg = f"No OBTDs found in {path.split('/')[-1]}! Skipping..."
        print(msg)
        return

    master_key = (j_i, pi_i, j_f, pi_f)
    if master_key in obtd_dict:
        """
        For example, the files
        
        OBTD_L_V50_GCLSTsdpfsdgix5pn_j8p_V50_GCLSTsdpfsdgix5pn_j8p.dat
        OBTD_S_V50_GCLSTsdpfsdgix5pn_j8p_V50_GCLSTsdpfsdgix5pn_j8p.dat

        contain the exact same OBTDs so we do not need to read both
        files. NOTE: Since these files contain different information
        about L and S, I might read both files in the future. For now,
        skip one of them.
        """
        msg = f"OBTDs for {master_key} already exists! Skipping file {path.split('/')[-1]}..."
        print(msg)
        return

    # obtd = np.zeros(shape=(n_elements, 5, n_initial_levels*n_final_levels), dtype=np.float64)
    obtd = np.zeros(shape=(n_elements, 3, n_initial_levels*n_final_levels), dtype=np.float64)
    obtd_dict[master_key] = obtd    # Provide view to the entire matrix in case vectorised operations are needed on the complete matrix.
    
    with open(path, "r") as infile:
        for transit_idx in range(n_initial_levels*n_final_levels):
            for line in infile:
                if "w.f." in line:
                    """
                    Example:
                    w.f.  J1=  0/2(    1)     J2=  2/2(    1)   <----- This one!
                    B(L;=>), B(L ;<=)      0.05162     0.01721
                    <||L||>      1    2     0.39870    -0.26442
                    ...
                    """
                    match_ = re.search(sect_header_pattern, line)
                    if not match_:
                        msg = "Unexpected pattern in OBTD file!"
                        raise RuntimeError(msg)
                    
                    j_i_current, idx_i_current, j_f_current, idx_f_current = map(int, match_.groups())
                    idx_i_current -= 1  # Make indices start from 0.
                    idx_f_current -= 1
                    key = (j_i_current, pi_i, idx_i_current, j_f_current, pi_f, idx_f_current)
                    key_with_transit_idx = key + (transit_idx,) # This key is used for np.load and np.save. Need the transit index to know which 2D slice.
                    
                    assert j_i_current == j_i   # Check that the ang. momentum in the filename agrees with the contents of the file.
                    assert j_f_current == j_f
                    assert idx_i_current < n_initial_levels
                    assert idx_f_current < n_final_levels
                    assert key not in obtd_dict # Something is wrong if the key already exists. Each entry should be unique.
                    assert key_with_transit_idx not in obtd_dict
                    
                    obtd_dict[key] = obtd[:, :, transit_idx]
                    obtd_dict[key_with_transit_idx] = obtd[:, :, transit_idx]
                    
                    break

            for line in infile:
                """
                Find the header of a section. Example:

                w.f.  J1=  0/2(    1)     J2=  2/2(    2)
                B(L;=>), B(L ;<=)      0.05162     0.01721
                <||L||>      1    2     0.39870    -0.26442

                i  j      OBTD    <i||L||j>  OBTD*<||>      <----- This one!
                1  1     0.00000     5.79655     0.00000
                1  2     0.00000     1.54919     0.00000
                1  9     0.00000     0.00000     0.00000
                ...

                NOTE: Only the three first columns are currently read.
                Dont know if I will have use for the final two columns
                but they can easily be added later. But then there is
                also the issue of storing <i||S||j> too.
                """
                if "OBTD" in line: break

            for obtd_idx in range(n_elements):
                obtd[obtd_idx, :, transit_idx] = [float(elem) for elem in infile.readline().split()[:3]]
                obtd[obtd_idx, 0, transit_idx] -= 1 # Make indices start from 0.
                obtd[obtd_idx, 1, transit_idx] -= 1
    
    assert np.all(obtd[:, :, 0] == obtd_dict[(j_i, pi_i, 0, j_f, pi_f, 0)])
    assert np.all(obtd[:, :, 0] == obtd_dict[(j_i, pi_i, 0, j_f, pi_f, 0, 0)])
    assert np.all(obtd[:, :, 0] == obtd_dict[(j_i, pi_i, j_f, pi_f)][:, :, 0])
    assert np.all(obtd[:, :, -1] == obtd_dict[(j_i, pi_i, n_initial_levels - 1, j_f, pi_f, n_final_levels - 1)])
    print(f"OBTD {path.split('/')[-1]} loaded", end="")
    
    timing = time.perf_counter() - timing
    if flags["debug"]:
        print(f" in {timing:.2f} s", end="")

    print()

def _load_obtd_parallel_wrapper(paths: list[str]) -> dict[tuple[int, ...]]:
    obtd_dict: dict[tuple[int, ...]] = {}
    for path in paths:
        _load_obtd(path=path, obtd_dict=obtd_dict)

    return obtd_dict

def load_interaction(
    filename_interaction: str,
    interaction: Interaction | None = None,
) -> Interaction:
    
    if interaction is None:
        interaction = Interaction()
    
    interaction.name = filename_interaction
    with open(filename_interaction, "r") as infile:
        """
        Extract information from the interaction file about the orbitals
        in the model space.
        """
        for line in infile:
            """
            Example
            -------
            ! GXPF1A  pf-shell 
            ! M. Honma, T. Otsuka, B. A. Brown, and T. Mizusaki, 
            !   Eur. Phys. J. A 25, Suppl. 1, 499 (2005). 
            !
            ! default input parameters 
            !namelist eff_charge = 1.5, 0.5
            !namelist orbs_ratio = 2, 3, 4, 6, 7, 8
            !
            ! model space
            4   4    20  20
            ...
            """
            if line[0] != "!":
                tmp = line.split()
                interaction.model_space_proton.n_orbitals = int(tmp[0])
                interaction.model_space_neutron.n_orbitals = int(tmp[1])
                interaction.model_space.n_orbitals = (
                    interaction.model_space_proton.n_orbitals + interaction.model_space_neutron.n_orbitals
                )
                interaction.n_core_protons = int(tmp[2])
                interaction.n_core_neutrons = int(tmp[3])
                break

        for line in infile:
            """
            Example
            -------
            4   4    20  20
            1       0   3   7  -1    !  1 = p 0f_7/2
            2       1   1   3  -1    !  2 = p 1p_3/2
            3       0   3   5  -1    !  3 = p 0f_5/2
            4       1   1   1  -1    !  4 = p 1p_1/2
            5       0   3   7   1    !  5 = n 0f_7/2
            6       1   1   3   1    !  6 = n 1p_3/2
            7       0   3   5   1    !  7 = n 0f_5/2
            8       1   1   1   1    !  8 = n 1p_1/2
            ! interaction
            ...
            """
            if line[0] == "!": break
            idx, n, l, j, tz = [int(i) for i in line.split("!")[0].split()]
            idx -= 1
            nucleon = "p" if tz == -1 else "n"
            name = f"{n}{spectroscopic_conversion[l]}{j}"
            tmp_orbital = OrbitalParameters(
                idx = idx,
                n = n,
                l = l,
                j = j,
                jz = tuple(range(-j, j+1, 2)),
                tz = tz,
                nucleon = nucleon,
                name = f"{nucleon}{name}",
                parity = (-1)**l,
                order = shell_model_order[name],
                ho_quanta = 2*n + l,
                degeneracy = j + 1,
            )
            interaction.model_space.orbitals.append(tmp_orbital)
            interaction.model_space.major_shell_names.add(shell_model_order[name].major_shell_name)

            if tz == -1:
                interaction.model_space_proton.orbitals.append(tmp_orbital)
                interaction.model_space_proton.major_shell_names.add(shell_model_order[name].major_shell_name)
                interaction.model_space_proton.all_jz_values += tmp_orbital.jz
            elif tz == +1:
                interaction.model_space_neutron.orbitals.append(tmp_orbital)
                interaction.model_space_neutron.major_shell_names.add(shell_model_order[name].major_shell_name)
                interaction.model_space_neutron.all_jz_values += tmp_orbital.jz
            else:
                msg = f"Valid values for tz are -1 and +1, got {tz=}"
                raise ValueError(msg)
            
        interaction.model_space.all_jz_values = interaction.model_space_proton.all_jz_values + interaction.model_space_neutron.all_jz_values
            
        for line in infile:
            """
            Example
            -------
            ! GCLSTsdpfsdgix5pn.int
            !   p-n formalism
                24      0
            ...
            """
            if line[0] != "!":
                tmp = line.split()
                if int(tmp[1]) != 0: raise NotImplementedError
                interaction.n_spe = int(tmp[0])
                break

        for line in infile:
            """
            Example
            -------
            1   1     -8.62400
            2   2     -5.67930
            3   3     -1.38290
            4   4     -4.13700
            5   5     -8.62400
            6   6     -5.67930
            7   7     -1.38290
            8   8     -4.13700
            518   1  42  -0.30000
            ...
            """
            tmp = line.split()
            if len(tmp) != 3: break
            interaction.spe.append(float(tmp[2]))

        try:
            interaction.n_tbme = int(tmp[0])
            interaction.tbme_mass_dependence_method = int(tmp[1])
            interaction.tbme_mass_dependence_denominator = int(tmp[2])
            interaction.tbme_mass_dependence_exponent = float(tmp[3])
        except IndexError:
            """
            I dont really know what this is yet.

            NOTE: 2023-10-11: Pretty sure that I can just leave the TBMEs as
            they are if no mass dependence is specified in the interaction
            file.
            """
            msg = "Interactions with no mass dependence have not yet been implemented."
            raise NotImplementedError(msg)

        for line in infile:
            """
            NOTE: This way of structuring the TBMEs is taken from
            espe.py.
            """
            i0, i1, i2, i3, j, tbme = line.split()
            i0 = int(i0) - 1
            i1 = int(i1) - 1
            i2 = int(i2) - 1
            i3 = int(i3) - 1
            j = 2*int(j)
            tbme = float(tbme)

            if (i0, i1, i2, i3, j) in interaction.tbme:
                """
                I dont yet understand why I should check the TBME value
                when I already know that the indices are the same. This
                is how it is done in espe.py (why check > 1.e-3 and not
                < 1.e-3?):
                if (i,j,k,l,J) in vtb:
                    if abs( v - vtb[i,j,k,l,J] )>1.e-3:
                            print( 'WARNING duplicate TBME', i+1,j+1,k+1,l+1,J,v,vtb[(i,j,k,l,J)] )

                """
                warnings.warn(f"Duplicate TBME! {i0 + 1}, {i1 + 1}, {i2 + 1}, {i3 + 1}, {j}, {tbme}, {interaction.tbme[(i0, i1, i2, i3, j)]}")

            interaction.tbme[(i0, i1, i2, i3, j)] = tbme
            s01 = (-1)**((interaction.model_space.orbitals[i0].j + interaction.model_space.orbitals[i1].j)/2 - j + 1)
            s23 = (-1)**((interaction.model_space.orbitals[i2].j + interaction.model_space.orbitals[i3].j)/2 - j + 1)
            
            if i0 != i1:
                interaction.tbme[(i1, i0, i2, i3, j)] = tbme*s01
            if i2 != i3:
                interaction.tbme[(i0, i1, i3, i2, j)] = tbme*s23
            if (i0 != i1) and (i2 != i3):
                interaction.tbme[(i1, i0, i3, i2, j)] = tbme*s01*s23
            
            if (i0, i1) != (i2, i3):
                interaction.tbme[(i2, i3, i0, i1, j)] = tbme
                if i0 != i1:
                    interaction.tbme[(i2, i3, i1, i0, j)] = tbme*s01
                if i2 != i3:
                    interaction.tbme[(i3, i2, i0, i1, j)] = tbme*s23
                if (i0 != i1) and (i2!=i3):
                    interaction.tbme[(i3, i2, i1, i0, j)] = tbme*s01*s23

    assert len(interaction.spe) == interaction.n_spe
    # assert len(interaction.tbme) == interaction.n_tbme

    interaction.vm = np.zeros((interaction.model_space.n_orbitals, interaction.model_space.n_orbitals), dtype=float)
    for i0 in range(interaction.model_space.n_orbitals):
        """
        Non-diagonal. TODO: Make a better description when I figure out
        what vm is.
        """
        for i1 in range(interaction.model_space.n_orbitals):
            j_min = abs(interaction.model_space.orbitals[i0].j - interaction.model_space.orbitals[i1].j)//2
            j_max =    (interaction.model_space.orbitals[i0].j + interaction.model_space.orbitals[i1].j)//2
            
            skip = 2 if (i0 == i1) else 1
            
            v: float = 0.0
            d: int = 0
            for j in range(j_min, j_max + 1, skip):
                """
                Using j_max + 1, not j_max + skip, because when i0 == i1
                both nucleons are in the same orbital and they cannot
                both have the same angular momentum z component.

                skip = 2 when i0 == i1 because only even numbered j are
                allowed when identical particles are in the same
                orbital.
                """
                try:
                    tbme = interaction.tbme[(i0, i1, i0, i1, j)]
                except KeyError:
                    """
                    I am unsure if this should be allowed at all and
                    should rather raise an exception.
                    """
                    warnings.warn(f"TBME entry not found! ({i0 + 1}, {i1 + 1}, {i0 + 1}, {i1 + 1}, {j})")
                    tbme = 0.0
                
                degeneracy = 2*j + 1
                v += degeneracy*tbme
                d += degeneracy
            
            interaction.vm[i0, i1] = v/d

    interaction.model_space.n_major_shells = len(interaction.model_space.major_shell_names)
    interaction.model_space_proton.n_major_shells = len(interaction.model_space_proton.major_shell_names)
    interaction.model_space_neutron.n_major_shells = len(interaction.model_space_neutron.major_shell_names)

    if not all(orb.idx == i for i, orb in enumerate(interaction.model_space.orbitals)):
        """
        Make sure that the list indices are the same as the orbit
        indices.
        """
        msg = (
            "The orbitals in the model space are not indexed correctly!"
        )
        raise KshellDataStructureError(msg)
    
    return interaction

def load_partition(
    filename_partition: str,
    interaction: Interaction,
    partition_proton: Partition | None = None,
    partition_neutron: Partition | None = None,
    partition_combined: Partition | None = None,
) -> tuple[Partition, Partition, Partition] | str:
    
    is_return_partitions = False
    if partition_proton is None:
        partition_proton = Partition()
        is_return_partitions = True
    
    if partition_neutron is None:
        partition_neutron = Partition()
        is_return_partitions = True
    
    if partition_combined is None:
        partition_combined = Partition()
        is_return_partitions = True
    
    header: str = ""
    with open(filename_partition, "r") as infile:
        # truncation_info: str = infile.readline()    # Eg. hw trucnation,  min hw = 0 ,   max hw = 1
        # hw_min, hw_max = [int(i.split("=")[1].strip()) for i in truncation_info.split(",")[1:]] # NOTE: No idea what happens if no hw trunc is specified.
        for line in infile:
            """
            Extract the information from the header before partitions
            are specified. Example:

            # hw trucnation,  min hw = 0 ,   max hw = 1
            # partition file of gs8.snt  Z=20  N=31  parity=-1
            20 31 -1
            # num. of  proton partition, neutron partition
            86 4
            # proton partition
            ...
            """
            if "#" not in line:
                tmp = [int(i) for i in line.split()]

                try:
                    """
                    For example:
                    20 31 -1
                    """
                    n_valence_protons, n_valence_neutrons, parity_partition = tmp

                    partition_proton.parity = parity_partition
                    partition_neutron.parity = parity_partition
                    partition_combined.parity = parity_partition
                    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
                    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
                    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
                except ValueError:
                    """
                    For example:
                    86 4
                    """
                    n_proton_configurations, n_neutron_configurations = tmp
                    infile.readline()   # Skip header.
                    break

            header += line

        ho_quanta_min: int = +1000  # The actual number of harmonic oscillator quanta will never be smaller or larger than these values.
        ho_quanta_max: int = -1000
        for line in infile:
            """
            Extract proton configurations.
            """
            if "# neutron partition" in line: break

            configuration = [int(i) for i in line.split()[1:]]

            parity_tmp = _calculate_configuration_parity(
                configuration = configuration,
                model_space = interaction.model_space_proton.orbitals
            )
            if   parity_tmp == -1: partition_proton.n_existing_negative_configurations += 1
            elif parity_tmp == +1: partition_proton.n_existing_positive_configurations += 1

            assert len(interaction.model_space_proton.orbitals) == len(configuration)

            ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                n*orb.ho_quanta for n, orb in zip(configuration, interaction.model_space_proton.orbitals)
            ])
            ho_quanta_min = min(ho_quanta_min, ho_quanta_tmp)
            ho_quanta_max = max(ho_quanta_max, ho_quanta_tmp)
            
            partition_proton.configurations.append(
                Configuration(
                    configuration = configuration,
                    parity = parity_tmp,
                    ho_quanta = ho_quanta_tmp,
                    energy = None,
                    is_original = True,
                )
            )
        partition_proton.ho_quanta_min_this_parity = ho_quanta_min
        partition_proton.ho_quanta_max_this_parity = ho_quanta_max
        ho_quanta_min: int = +1000  # Reset for neutrons.
        ho_quanta_max: int = -1000
        
        for line in infile:
            """
            Extract neutron configurations.
            """
            if "# partition of proton and neutron" in line: break

            configuration = [int(i) for i in line.split()[1:]]

            parity_tmp = _calculate_configuration_parity(
                configuration = configuration,
                model_space = interaction.model_space_neutron.orbitals
            )
            if   parity_tmp == -1: partition_neutron.n_existing_negative_configurations += 1
            elif parity_tmp == +1: partition_neutron.n_existing_positive_configurations += 1

            assert len(interaction.model_space_neutron.orbitals) == len(configuration)

            ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                n*orb.ho_quanta for n, orb in zip(configuration, interaction.model_space_neutron.orbitals)
            ])
            ho_quanta_min = min(ho_quanta_min, ho_quanta_tmp)
            ho_quanta_max = max(ho_quanta_max, ho_quanta_tmp)
            
            partition_neutron.configurations.append(
                Configuration(
                    configuration = configuration,
                    parity = parity_tmp,
                    ho_quanta = ho_quanta_tmp,
                    energy = None,
                    is_original = True,
                )
            )
        partition_neutron.ho_quanta_min_this_parity = ho_quanta_min
        partition_neutron.ho_quanta_max_this_parity = ho_quanta_max
        ho_quanta_min: int = +1000  # Reset for combined.
        ho_quanta_max: int = -1000
        n_combined_configurations = int(infile.readline())

        for line in infile:
            """
            Extract the combined pn configurations.
            """
            proton_idx, neutron_idx = line.split()
            proton_idx = int(proton_idx) - 1
            neutron_idx = int(neutron_idx) - 1
            parity_tmp = partition_proton.configurations[proton_idx].parity*partition_neutron.configurations[neutron_idx].parity
            assert parity_partition == parity_tmp

            if   parity_tmp == -1: partition_combined.n_existing_negative_configurations += 1
            elif parity_tmp == +1: partition_combined.n_existing_positive_configurations += 1

            ho_quanta_tmp = (
                partition_proton.configurations[proton_idx].ho_quanta + 
                partition_neutron.configurations[neutron_idx].ho_quanta
            )
            ho_quanta_min = min(ho_quanta_min, ho_quanta_tmp)
            ho_quanta_max = max(ho_quanta_max, ho_quanta_tmp)

            energy = configuration_energy(
                interaction = interaction,
                proton_configuration = partition_proton.configurations[proton_idx],
                neutron_configuration = partition_neutron.configurations[neutron_idx],
            )

            partition_combined.configurations.append(
                Configuration(
                    configuration = [proton_idx, neutron_idx],
                    parity = parity_partition,
                    ho_quanta = ho_quanta_tmp,
                    energy = energy,
                    is_original = True,
                )
            )
        partition_combined.ho_quanta_min_this_parity = ho_quanta_min
        partition_combined.ho_quanta_max_this_parity = ho_quanta_max

        partition_combined.ho_quanta_min = min(ho_quanta_min, partition_combined.ho_quanta_min_opposite_parity)
        partition_combined.ho_quanta_max = max(ho_quanta_max, partition_combined.ho_quanta_max_opposite_parity)

    energies = [configuration.energy for configuration in partition_combined.configurations]
    partition_combined.min_configuration_energy = min(energies)
    partition_combined.max_configuration_energy = max(energies)
    partition_combined.max_configuration_energy_original = partition_combined.max_configuration_energy

    _sanity_checks(
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
        interaction = interaction,
    )
    assert len(partition_proton.configurations) == n_proton_configurations
    assert len(partition_neutron.configurations) == n_neutron_configurations
    assert len(partition_combined.configurations) == n_combined_configurations

    assert (
        partition_proton.n_existing_negative_configurations + partition_proton.n_existing_positive_configurations + 
        partition_proton.n_new_negative_configurations + partition_proton.n_new_positive_configurations
    ) == n_proton_configurations
    assert (
        partition_neutron.n_existing_negative_configurations + partition_neutron.n_existing_positive_configurations + 
        partition_neutron.n_new_negative_configurations + partition_neutron.n_new_positive_configurations
    ) == n_neutron_configurations
    assert (
        partition_combined.n_existing_negative_configurations + partition_combined.n_existing_positive_configurations + 
        partition_combined.n_new_negative_configurations + partition_combined.n_new_positive_configurations
    ) == n_combined_configurations

    if is_return_partitions:
        return partition_proton, partition_neutron, partition_combined
    else:
        return header

def _parity_string_to_integer(parity: str):
    if parity == "+":
        res = 1
    elif parity == "-":
        res = -1
    else:
        msg = f"Invalid parity read from file. Got: '{parity}'."
        raise KshellDataStructureError(msg)

    return res

def _load_energy_levels(infile: TextIO) -> tuple[list, int]:
    """
    Load excitation energy, spin and parity into a list of structure:
    levels = [[energy, spin, parity], ...].
    
    Parameters
    ----------
    infile : TextIO
        The KSHELL summary file at the starting position of the level
        data.

    Returns
    -------
    levels : list
        List of level data.
        
    negative_spin_counts : int
        The number of negative spin levels encountered.

    Example
    -------
    Energy levels

    N    J prty N_Jp    T     E(MeV)  Ex(MeV)  log-file

    1   5/2 +     1   3/2    -16.565    0.000  log_O19_sdpf-mu_m1p.txt 
    2   3/2 +     1   3/2    -15.977    0.588  log_O19_sdpf-mu_m1p.txt 
    3   1/2 +     1   3/2    -15.192    1.374  log_O19_sdpf-mu_m1p.txt 
    4   9/2 +     1   3/2    -13.650    2.915  log_O19_sdpf-mu_m1p.txt 
    5   7/2 +     1   3/2    -13.267    3.298  log_O19_sdpf-mu_m1p.txt 
    6   5/2 +     2   3/2    -13.074    3.491  log_O19_sdpf-mu_m1p.txt
    """
    levels = []
    negative_spin_counts = 0
    for _ in range(3): infile.readline()
    for line in infile:
        try:
            tmp = line.split()
            
            if tmp[1] == "-1":
                """
                -1 spin states in the KSHELL data file indicates
                bad states which should not be included.
                """
                negative_spin_counts += 1  # Debug.
                continue
            
            parity = 1 if tmp[2] == "+" else -1
            energy = float(tmp[5])
            spin = 2*float(Fraction(tmp[1]))
            idx = int(tmp[3])
            levels.append([energy, spin, parity, idx])
        except IndexError:
            """
            End of energies.
            """
            break

    return levels, negative_spin_counts

def _load_transition_probabilities_old(infile: TextIO) -> tuple[list, int]:
    """
    For summary files with old syntax (pre 2021-11-24).

    Parameters
    ----------
    infile : TextIO
        The KSHELL summary file at the starting position of either of
        the transition probability sections.

    Returns
    -------
    transitions : list
        List of transition data.
        
    negative_spin_counts : int
        The number of negative spin levels encountered.
    """
    negative_spin_counts = 0
    transitions = []
    for _ in range(2): infile.readline()
    for line in infile:
        try:
            """
            Example of possible lines in file:
            J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
            2+(11) 18.393 2+(10) 17.791 0.602 0.1(  0.0) 0.1( 0.0)
            3/2+( 1) 0.072 5/2+( 1) 0.000 0.071 0.127( 0.07) 0.084( 0.05)
            2+(10) 17.791 2+( 1) 5.172 12.619 0.006( 0.00) 0.006( 0.00)
            3+( 8) 19.503 2+(11) 18.393 1.111 0.000( 0.00) 0.000( 0.00)
            1+( 7) 19.408 2+( 9) 16.111 3.297 0.005( 0.00) 0.003( 0.00)
            5.0+(60) 32.170  4.0+(100) 31.734  0.436    0.198( 0.11)    0.242( 0.14)
            4.0-( 3)  3.191  3.0+(10)  3.137  0.054      0.0(  0.0)      0.0(  0.0)
            0.0+(46)', '47.248', '1.0+(97)', '45.384', '1.864', '23.973(13.39)', '7.991(', '4.46)
            """
            tmp = line.split()
            len_tmp = len(tmp)
            case_ = None # Used for identifying which if-else case reads wrong.
            
            # Location of initial parity is common for all cases.
            parity_idx = tmp[0].index("(") - 1 # Find index of initial parity.
            parity_initial = 1 if tmp[0][parity_idx] == "+" else -1
            parity_initial_symbol = tmp[0][parity_idx]
            
            # Location of initial spin is common for all cases.
            spin_initial = float(Fraction(tmp[0][:parity_idx]))
            
            if (tmp[1][-1] != ")") and (tmp[3][-1] != ")") and (len_tmp == 9):
                """
                Example:
                J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                2+(11)   18.393  2+(10)    17.791  0.602    0.1(    0.0)    0.1(    0.0)
                5.0+(60) 32.170  4.0+(100) 31.734  0.436    0.198( 0.11)    0.242( 0.14)
                """
                case_ = 0
                E_gamma = float(tmp[4])
                Ex_initial = float(tmp[1])
                reduced_transition_prob_decay = float(tmp[5][:-1])
                reduced_transition_prob_excite = float(tmp[7][:-1])
                parity_final_symbol = tmp[2].split("(")[0][-1]
                spin_final = float(Fraction(tmp[2].split(parity_final_symbol)[0]))
                Ex_final = float(tmp[3])
                idx_initial = int(tmp[0].split("(")[1].split(")")[0])
                idx_final = int(tmp[2].split("(")[1].split(")")[0])

            elif (tmp[1][-1] != ")") and (tmp[3][-1] == ")") and (len_tmp == 10):
                """
                Example:
                J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                2+(10) 17.791 2+( 1) 5.172 12.619 0.006( 0.00) 0.006( 0.00)
                """
                case_ = 1
                E_gamma = float(tmp[5])
                Ex_initial = float(tmp[1])
                reduced_transition_prob_decay = float(tmp[6][:-1])
                reduced_transition_prob_excite = float(tmp[8][:-1])
                parity_final_symbol = tmp[2].split("(")[0][-1]
                # spin_final = float(Fraction(tmp[2][:-2]))
                spin_final = float(Fraction(tmp[2].split(parity_final_symbol)[0]))
                Ex_final = float(tmp[4])
                idx_initial = int(tmp[0].split("(")[1].split(")")[0])
                idx_final = int(tmp[3][0])
            
            elif (tmp[1][-1] == ")") and (tmp[4][-1] != ")") and (len_tmp == 10):
                """
                Example:
                J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                3+( 8)   19.503 2+(11)    18.393 1.111 0.000( 0.00) 0.000( 0.00)
                1.0+( 1) 5.357  0.0+(103) 0.000  5.357 0.002( 0.00) 0.007( 0.00)
                4.0-( 3)  3.191  3.0+(10)  3.137  0.054      0.0(  0.0)      0.0(  0.0)
                """
                case_ = 2
                E_gamma = float(tmp[5])
                Ex_initial = float(tmp[2])
                reduced_transition_prob_decay = float(tmp[6][:-1])
                reduced_transition_prob_excite = float(tmp[8][:-1])
                parity_final_symbol = tmp[3].split("(")[0][-1]
                spin_final = float(Fraction(tmp[3].split(parity_final_symbol)[0]))
                Ex_final = float(tmp[4])
                idx_initial = int(tmp[1][0])
                idx_final = int(tmp[3].split("(")[1].split(")")[0])

            elif (tmp[1][-1] == ")") and (tmp[4][-1] == ")") and (len_tmp == 11):
                """
                Example:
                J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                1+( 7) 19.408 2+( 9) 16.111 3.297 0.005( 0.00) 0.003( 0.00)
                """
                case_ = 3
                E_gamma = float(tmp[6])
                Ex_initial = float(tmp[2])
                reduced_transition_prob_decay = float(tmp[7][:-1])
                reduced_transition_prob_excite = float(tmp[9][:-1])
                parity_final_symbol = tmp[3].split("(")[0][-1]
                # spin_final = float(Fraction(tmp[3][:-2]))
                spin_final = float(Fraction(tmp[3].split(parity_final_symbol)[0]))
                Ex_final = float(tmp[5])
                idx_initial = int(tmp[1][0])
                idx_final = int(tmp[4][0])

            elif (tmp[5][-1] == ")") and (tmp[2][-1] == ")") and (len_tmp == 8):
                """
                Example:
                J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                0.0+(46) 47.248  1.0+(97) 45.384  1.864   23.973(13.39)    7.991( 4.46)
                """
                case_ = 4
                E_gamma = float(tmp[4])
                Ex_initial = float(tmp[1])
                reduced_transition_prob_decay = float(tmp[5].split("(")[0])
                reduced_transition_prob_excite = float(tmp[6][:-1])
                parity_final_symbol = tmp[2].split("(")[0][-1]
                spin_final = float(Fraction(tmp[2].split(parity_final_symbol)[0]))
                Ex_final = float(tmp[3])
                idx_initial = int(tmp[0].split("(")[1].split(")")[0])
                idx_final = int(tmp[2].split("(")[1].split(")")[0])

            else:
                msg = "ERROR: Structure not accounted for!"
                msg += f"\n{line=}"
                raise KshellDataStructureError(msg)

            if parity_final_symbol == "+":
                parity_final = 1
            elif parity_final_symbol == "-":
                parity_final = -1
            else:
                msg = f"Could not properly read the final parity! {case_=}"
                raise KshellDataStructureError(msg)

            if (spin_final == -1) or (spin_initial == -1):
                """
                -1 spin states in the KSHELL data file indicates
                bad states which should not be included.
                """
                negative_spin_counts += 1  # Debug.
                continue

            # reduced_transition_prob_decay_list.append([
            #     2*spin_initial, parity_initial, Ex_initial, 2*spin_final,
            #     parity_final, Ex_final, E_gamma, reduced_transition_prob_decay,
            #     reduced_transition_prob_excite
            # ])
            transitions.append([
                2*spin_initial, parity_initial, idx_initial, Ex_initial,
                2*spin_final, parity_final, idx_final, Ex_final, E_gamma,
                reduced_transition_prob_decay, reduced_transition_prob_excite
            ])

        except ValueError as err:
            """
            One of the float conversions failed indicating that
            the structure of the line is not accounted for.
            """
            msg = "\n" + err.__str__() + f"\n{case_=}" + f"\n{line=}"
            raise KshellDataStructureError(msg)

        except IndexError:
            """
            End of probabilities.
            """
            break

    return transitions, negative_spin_counts
    
def _load_transition_probabilities(infile: TextIO) -> tuple[list, int]:
    """
    For summary files with new syntax (post 2021-11-24).

    Parameters
    ----------
    infile : TextIO
        The KSHELL summary file at the starting position of either of
        the transition probability sections.

    Returns
    -------
    transitions : list
        List of transition data.

    negative_spin_counts : int
        The number of negative spin levels encountered.

    Example
    -------
    B(E2)  ( > -0.0 W.u.)  mass = 50    1 W.u. = 10.9 e^2 fm^4
    e^2 fm^4 (W.u.)
    J_i  pi_i idx_i Ex_i    J_f  pi_f idx_f Ex_f      dE         B(E2)->         B(E2)->[wu]     B(E2)<-         B(E2)<-[wu]
    5    +    1     0.036   6    +    1     0.000     0.036     70.43477980      6.43689168     59.59865983      5.44660066
    4    +    1     0.074   6    +    1     0.000     0.074     47.20641983      4.31409897     32.68136758      2.98668391
    """
    negative_spin_counts = 0
    transitions = []
    for _ in range(2): infile.readline()
    for line in infile:
        line_split = line.split()
        if not line_split: break
        
        spin_initial = float(Fraction(line_split[0]))
        parity_initial = _parity_string_to_integer(line_split[1])
        idx_initial = int(line_split[2])
        Ex_initial = float(line_split[3])

        spin_final = float(Fraction(line_split[4]))
        parity_final = _parity_string_to_integer(line_split[5])
        idx_final = int(line_split[6])
        Ex_final = float(line_split[7])

        E_gamma = float(line_split[8])
        reduced_transition_prob_decay = float(line_split[9])
        reduced_transition_prob_excite = float(line_split[11])

        if (spin_final < 0) or (spin_initial < 0):
            """
            -1 spin states in the KSHELL data file indicates
            bad states which should not be included.
            """
            negative_spin_counts += 1  # Debug.
            continue

        # reduced_transition_prob_decay_list.append([
        #     2*spin_initial, parity_initial, Ex_initial, 2*spin_final,
        #     parity_final, Ex_final, E_gamma, reduced_transition_prob_decay,
        #     reduced_transition_prob_excite
        # ])
        transitions.append([
            2*spin_initial, parity_initial, idx_initial, Ex_initial,
            2*spin_final, parity_final, idx_final, Ex_final, E_gamma,
            reduced_transition_prob_decay, reduced_transition_prob_excite
        ])

    return transitions, negative_spin_counts

def _generic_loader(arg_list: list) -> tuple[list, int]:
    """
    Constructed for parallel loading, but can be used in serial as well.
    """
    fname, condition, loader, thread_idx = arg_list
    
    if flags["parallel"]:
        print(f"Thread {thread_idx} loading {condition} values...")
    else:
        print(f"Loading {condition} values...")
        
    load_time = time.perf_counter()
    
    with open(fname, "r") as infile:
        for line in infile:
            if condition in line:
                ans = loader(infile)
                break
        else:
            ans = [], 0
    
    load_time = time.perf_counter() - load_time
    
    if not ans[0]:
        print(f"No {condition} transitions found in {fname}")
    else:
        print(f"Thread {thread_idx} finished loading {condition} values in {load_time:.2f} s")

    return ans

def _load_transition_probabilities_jem(infile: TextIO) -> tuple[list, int]:
    """
    JEM has modified the summary files from KSHELL with a slightly
    different syntax. This function reads that syntax. Note also that
    these summary files have 2*J, not J.

    Parameters
    ----------
    infile : TextIO
        The KSHELL summary file at the starting position of either of
        the transition probability sections.

    Returns
    -------
    transitions : list
        List of transition data.

    negative_spin_counts : int
        The number of negative spin levels encountered.

    Example
    -------
    B(M1)  larger than 1e-08 mu_N^2
    2Ji        Ei      2Jf        Ef       Ex            B(M1)->         B(M1)<- 
    2 - (   1) -35.935   0 - (   8) -35.583   0.352      0.00428800      0.01286400
    0 - (   8) -35.583   2 - (   2) -35.350   0.233      0.45171030      0.15057010
    0 - (   8) -35.583   2 - (   3) -34.736   0.847      0.04406500      0.01468830
    """
    negative_spin_counts = 0
    transitions = []

    infile.readline()   # Skip header line.
    for line in infile:
        line_split = line.split()
        if not line_split: break

        spin_initial = int(line_split[0])
        parity_initial = _parity_string_to_integer(line_split[1])
        idx_initial = int(line_split[3].strip(")"))
        Ex_initial = float(line_split[4])
        spin_final = int(line_split[5])
        parity_final = _parity_string_to_integer(line_split[6])
        idx_final = int(line_split[8].strip(")"))
        Ex_final = float(line_split[9])
        E_gamma = float(line_split[10])
        reduced_transition_prob_decay = float(line_split[11])
        reduced_transition_prob_excite = float(line_split[12])

        if (spin_final < 0) or (spin_initial < 0):
            """
            -1 spin states in the KSHELL data file indicates
            bad states which should not be included.
            """
            negative_spin_counts += 1  # Debug.
            continue

        transitions.append([
            spin_initial, parity_initial, idx_initial, Ex_initial,
            spin_final, parity_final, idx_final, Ex_final, E_gamma,
            reduced_transition_prob_decay, reduced_transition_prob_excite
        ])
    
    return transitions, negative_spin_counts
