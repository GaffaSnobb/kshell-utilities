import os, sys, multiprocessing
from fractions import Fraction
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

atomic_numbers = {
    "oxygen": 8, "fluorine": 9, "neon": 10, "sodium": 11, "magnesium": 12,
    "aluminium": 13, "silicon": 14, "phosphorus": 15, "sulfur": 16,
    "chlorine": 17, "argon": 18
}
# atomic_numbers_reversed = dict([(y, x) for x,y in ksutil.atomic_numbers.items()])
atomic_numbers_reversed = {
    8: 'oxygen', 9: 'fluorine', 10: 'neon', 11: 'sodium', 12: 'magnesium',
    13: 'aluminium', 14: 'silicon', 15: 'phosphorus', 16: 'sulfur',
    17: 'chlorine', 18: 'argon'
}

def generate_states(
    start: int = 0,
    stop: int = 14,
    n_states: int = 100
    ):
 
    def correct_syntax(lst):
        for elem in lst:
            print(elem, end=", ")
    
    positive = [f"{i:g}{'+'}{n_states}" for i in np.arange(start, stop+1, 0.5)]
    negative = [f"{i:g}{'-'}{n_states}" for i in np.arange(start, stop+1, 0.5)]

    correct_syntax(positive)
    print("\n")
    correct_syntax(negative)

def create_jpi_list(
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
        A Nx2 array of spins and parities [[spin, parity], ...] sorted
        with respect to the spin. N is the number of unique spins in
        'spins'.
    """
    unique_spins, unique_spins_idx = np.unique(spins, return_index=True)
    spins_parities = np.empty((len(unique_spins), 2))
    spins_parities[:, 0] = unique_spins
    spins_parities[:, 1] = parities[unique_spins_idx]

    return spins_parities.tolist()  # Convert to list since list.index is much faster than np.where.

class DataStructureNotAccountedForError(Exception):
    pass

class ReadKshellOutput:
    """
    Implemented as class just to avoid returning a bunch of values.
    Access instance attributes instead.
    TODO: Implement call to level_plot as a method.
    TODO: Why is initial parity twice in self.transitions? Fix.
    TODO: Skip -1 spin states. These are bad outputs from KSHELL.

    Attributes
    ----------
    self.Ex:
        Array of excitation energies zeroed at the ground state energy.

    self.BM1:
        Array of [[E, B_decay_prob, E_gamma], ...].
        Reduced transition probabilities for M1.

    self.BE2:
        Array of [[E, B_decay_prob, E_gamma], ...].
        Reduced transition probabilities for E2.

    self.levels:
        Array containing energy, spin, and parity for each excited
        state. [[E, 2J, parity], ...].

    self.transitions:
        Mx8 array containing [2J_final, parity_initial, Ex_final,
        2J_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)]
    """
    def __init__(self, path: str, load_and_save_to_file: bool):
        """
        TODO: Not necessarry to call all _extract_... methods when
        a directory path is passed.

        Parameters
        ----------
        path : string
            Path of KSHELL output file directory, or path to a specific
            KSHELL data file.

        load_and_save_to_file:
            Toggle saving data as .npy files on / off. If 'overwrite',
            saved .npy files are overwritten.
        """

        self.path = path
        self.load_and_save_to_file = load_and_save_to_file
        # Some attributes might not be altered, depending on the input file.
        self.fname_summary = None
        self.fname_ptn = None
        self.nucleus = None
        self.model_space = None
        self.proton_partition = None
        self.neutron_partition = None
        self.Ex = None
        self.BM1 = None
        self.BE2 = None
        self.levels = None
        self.transitions = None
        self.transitions_BM1 = None
        self.transitions_BE2 = None
        self.truncation = None
        # Debug.
        self.minus_one_spin_counts = np.array([0, 0])  # The number of skipped -1 spin states for [levels, transitions].

        if isinstance(self.load_and_save_to_file, str) and (self.load_and_save_to_file != "overwrite"):
            msg = "Allowed values for 'load_and_save_to_file' are: 'True', 'False', 'overwrite'."
            msg += f" Got '{self.load_and_save_to_file}'."
            raise ValueError(msg)

        if os.path.isdir(path):
            """
            If input 'path' is a directory containing KSHELL files,
            extract info from both summary and .ptn file.
            """
            for elem in os.listdir(path):
                if elem.startswith("summary"):
                    self.fname_summary = f"{path}/{elem}"
                    self._extract_info_from_summary_fname()
                    self._read_summary()

                elif elem.endswith(".ptn"):
                    self.fname_ptn = f"{path}/{elem}"
                    self._extract_info_from_ptn_fname()
                    self.read_ptn()

        else:
            """
            'path' is a single file, not a directory.
            """
            fname = path.split("/")[-1]

            if fname.startswith("summary"):
                self.fname_summary = path
                self._extract_info_from_summary_fname()
                self._read_summary()

            elif fname.endswith(".ptn"):
                self.fname_ptn = path
                self._extract_info_from_ptn_fname()
                self._read_ptn()

            else:
                msg = f"Handling for file {fname} is not implemented."
                raise DataStructureNotAccountedForError(msg)

    def _extract_info_from_ptn_fname(self):
        """
        Extract nucleus and model space name.
        """
        fname_split = self.fname_ptn.split("/")[-1]
        fname_split = fname_split.split("_")
        self.nucleus = fname_split[0]
        self.model_space = fname_split[1]

    def _read_ptn(self):
        """
        Read KSHELL partition file (.ptn) and extract proton partition,
        neutron partition, and particle-hole truncation data. Save as
        instance attributes.
        """

        line_number = 0
        line_number_inner = 0
        self.truncation = []

        with open(self.fname_ptn, "r") as infile:
            for line in infile:
                line_number += 1
                
                if line.startswith("# proton partition"):
                    for line_inner in infile:
                        """
                        Read until next '#'.
                        """
                        line_number_inner += 1
                        if line_inner.startswith("#"):
                            line = line_inner
                            break
                    
                    self.proton_partition = np.loadtxt(
                        fname = self.fname_ptn,
                        skiprows = line_number,
                        max_rows = line_number_inner
                    )
                    line_number += line_number_inner
                    line_number_inner = 0
                
                if line.startswith("# neutron partition"):
                    for line_inner in infile:
                        """
                        Read until next '#'.
                        """
                        line_number_inner += 1
                        if line_inner.startswith("#"):
                            line = line_inner
                            break
                    
                    self.neutron_partition = np.loadtxt(
                        fname = self.fname_ptn,
                        skiprows = line_number,
                        max_rows = line_number_inner
                    )
                    line_number += line_number_inner
                    line_number_inner = 0

                if line.startswith("# particle-hole truncation"):
                    for line_inner in infile:
                        """
                        Loop over all particle-hole truncation lines.
                        """
                        line_number += 1
                        line_inner_split = line_inner.split()

                        if (len(line_inner_split) < 2):
                            """
                            Condition will probably not get fulfilled.
                            Safety precaution due to indexing in this
                            loop.
                            """
                            break

                        if (line_inner_split[1]).startswith("["):
                            """
                            '[' indicates that 'line_inner' is still
                            containing truncation information.
                            """
                            for colon_index, elem in enumerate(line_inner_split):
                                """
                                Find the index of the colon ':' to
                                decide the orbit numbers and occupation
                                numbers.
                                """
                                if (elem == ":"): break

                            occupation = [int(occ) for occ in line_inner_split[colon_index + 1:]]   # [min, max].
                            orbit_numbers = "".join(line_inner_split[1:colon_index])
                            orbit_numbers = orbit_numbers.replace("[", "")
                            orbit_numbers = orbit_numbers.replace("]", "")
                            orbit_numbers = orbit_numbers.replace(" ", "")  # This can prob. be removed because of the earlier split.
                            orbit_numbers = orbit_numbers.split(",")
                            orbit_numbers = [int(orbit) for orbit in orbit_numbers]
                            
                            for orbit in orbit_numbers:
                                self.truncation.append((orbit, occupation))
                        
                        else:
                            """
                            Line does not contain '[' and thus does not
                            contain truncation information.
                            """
                            break

    def _extract_info_from_summary_fname(self):
        """
        Extract nucleus and model space name.
        """
        fname_split = self.fname_summary.split("/")[-1]  # Remove path.
        fname_split = fname_split.split("_")
        self.nucleus = fname_split[1]
        self.model_space = fname_split[2][:-4]  # Remove .txt and keep model space name.

    def _read_summary(self):
        """
        Read energy level data, transition probabilities and transition
        strengths from KSHELL output files.

        TODO: Change all the substring indexing to something more
        rigorous, like string.split and similar.

        Raises
        ------
        DataStructureNotAccountedForError
            If the KSHELL file has unexpected structure / syntax.
        """
        npy_path = "tmp"
        base_fname = self.path.split("/")[-1][:-4]

        try:
            os.mkdir(npy_path)
        except FileExistsError:
            pass
            
        levels_fname = f"{npy_path}/{base_fname}_levels.npy"
        transitions_fname = f"{npy_path}/{base_fname}_transitions.npy"
        transitions_BM1_fname = f"{npy_path}/{base_fname}_transitions_BM1.npy"
        transitions_BE2_fname = f"{npy_path}/{base_fname}_transitions_BE2.npy"
        Ex_fname = f"{npy_path}/{base_fname}_Ex.npy"
        BM1_fname = f"{npy_path}/{base_fname}_BM1.npy"
        BE2_fname = f"{npy_path}/{base_fname}_BE2.npy"
        debug_fname = f"{npy_path}/{base_fname}_debug.npy"

        fnames = [
            levels_fname, transitions_fname, Ex_fname, BM1_fname, BE2_fname,
            transitions_BM1_fname, transitions_BE2_fname, debug_fname
        ]

        if self.load_and_save_to_file != "overwrite":
            """
            Do not load files if overwrite parameter has been passed.
            """
            if all([os.path.isfile(fname) for fname in fnames]) and self.load_and_save_to_file:
                """
                If all files exist, load them. If any of the files does
                not exist, all will be generated.
                """
                self.Ex = np.load(file=Ex_fname, allow_pickle=True)
                self.BM1 = np.load(file=BM1_fname, allow_pickle=True)
                self.BE2 = np.load(file=BE2_fname, allow_pickle=True)
                self.levels = np.load(file=levels_fname, allow_pickle=True)
                self.transitions = np.load(file=transitions_fname, allow_pickle=True)
                self.transitions_BM1 = np.load(file=transitions_BM1_fname, allow_pickle=True)
                self.transitions_BE2 = np.load(file=transitions_BE2_fname, allow_pickle=True)
                self.debug = np.load(file=debug_fname, allow_pickle=True)
                print("Summary data loaded from .npy!")
                return

        def load_energy_levels(infile):
            for _ in range(3): infile.readline()
            for line in infile:
                try:
                    tmp = line.split()
                    
                    if tmp[1] == "-1":
                        """
                        -1 spin states in the KSHELL data file indicates
                        bad states which should not be included.
                        """
                        self.minus_one_spin_counts[0] += 1  # Debug.
                        continue
                    
                    self.Ex.append(float(tmp[6]))
                    parity = 1 if tmp[2] == "+" else -1
                    self.levels.append([float(tmp[5]), 2*float(Fraction(tmp[1])), parity])
                except IndexError:
                    """
                    End of energies.
                    """
                    break

        def load_transition_probabilities(infile, reduced_transition_prob_list):
            """
            Parameters
            ----------
            infile:
                The KSHELL summary file.

            reduced_transition_prob_list:
                List for storing B(M1) or B(E2) values.
            """
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
                    0.0+(46)', '47.248', '1.0+(97)', '45.384', '1.864', '23.973(13.39)', '7.991(', '4.46)
                    """
                    tmp = line.split()
                    len_tmp = len(tmp)
                    case = None # Used for identifying which if-else case reads wrong.
                    
                    # Location of initial parity is common for all cases.
                    parity_idx = tmp[0].index("(") - 1 # Find index of initial parity.
                    parity_initial = 1 if tmp[0][parity_idx] == "+" else -1
                    parity_symbol = tmp[0][parity_idx]
                    
                    # Location of initial spin is common for all cases.
                    J_initial = float(Fraction(tmp[0][:parity_idx]))
                    
                    if (tmp[1][-1] != ")") and (tmp[3][-1] != ")") and (len_tmp == 9):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        2+(11)   18.393  2+(10)    17.791  0.602    0.1(    0.0)    0.1(    0.0)
                        5.0+(60) 32.170  4.0+(100) 31.734  0.436    0.198( 0.11)    0.242( 0.14)
                        """
                        case = 0
                        E_gamma = float(tmp[4])
                        Ex_initial = float(tmp[1])
                        reduced_transition_prob = float(tmp[5][:-1])    # B(M1) or B(E2).
                        J_final = float(Fraction(tmp[2].split(parity_symbol)[0]))
                        Ex_final = float(tmp[3])

                    elif (tmp[1][-1] != ")") and (tmp[3][-1] == ")") and (len_tmp == 10):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        2+(10) 17.791 2+( 1) 5.172 12.619 0.006( 0.00) 0.006( 0.00)
                        """
                        case = 1
                        E_gamma = float(tmp[5])
                        Ex_initial = float(tmp[1])
                        reduced_transition_prob = float(tmp[6][:-1])
                        J_final = float(Fraction(tmp[2][:-2]))
                        Ex_final = float(tmp[4])
                    
                    elif (tmp[1][-1] == ")") and (tmp[4][-1] != ")") and (len_tmp == 10):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        3+( 8)   19.503 2+(11)    18.393 1.111 0.000( 0.00) 0.000( 0.00)
                        1.0+( 1) 5.357  0.0+(103) 0.000  5.357 0.002( 0.00) 0.007( 0.00)
                        """
                        case = 2
                        E_gamma = float(tmp[5])
                        Ex_initial = float(tmp[2])
                        reduced_transition_prob = float(tmp[6][:-1])
                        J_final = float(Fraction(tmp[3].split(parity_symbol)[0]))
                        Ex_final = float(tmp[4])

                    elif (tmp[1][-1] == ")") and (tmp[4][-1] == ")") and (len_tmp == 11):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        1+( 7) 19.408 2+( 9) 16.111 3.297 0.005( 0.00) 0.003( 0.00)
                        """
                        case = 3
                        E_gamma = float(tmp[6])
                        Ex_initial = float(tmp[2])
                        reduced_transition_prob = float(tmp[7][:-1])
                        J_final = float(Fraction(tmp[3][:-2]))
                        Ex_final = float(tmp[5])

                    elif (tmp[5][-1] == ")") and (tmp[2][-1] == ")") and (len_tmp == 8):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        0.0+(46) 47.248  1.0+(97) 45.384  1.864   23.973(13.39)    7.991( 4.46)
                        """
                        case = 4
                        E_gamma = float(tmp[4])
                        Ex_initial = float(tmp[1])
                        reduced_transition_prob = float(tmp[5].split("(")[0])
                        J_final = float(Fraction(tmp[2].split(parity_symbol)[0]))
                        Ex_final = float(tmp[3])

                    else:
                        msg = "ERROR: Structure not accounted for!"
                        msg += f"\n{line=}"
                        raise DataStructureNotAccountedForError(msg)

                    if (J_final == -1) or (J_initial == -1):
                        """
                        -1 spin states in the KSHELL data file indicates
                        bad states which should not be included.
                        """
                        self.minus_one_spin_counts[1] += 1  # Debug.
                        continue
                    
                    reduced_transition_prob_list.append([
                        Ex_initial, reduced_transition_prob, E_gamma
                    ])
                    self.transitions.append([
                        2*J_final, parity_initial, Ex_final, 2*J_initial,
                        parity_initial, Ex_initial, E_gamma,
                        reduced_transition_prob
                    ])

                except ValueError as err:
                    """
                    One of the float conversions failed indicating that
                    the structure of the line is not accounted for.
                    """
                    msg = "\n" + err.__str__() + f"\n{case=}" + f"\n{line=}"
                    raise DataStructureNotAccountedForError(msg)

                except IndexError:
                    """
                    End of probabilities.
                    """
                    break

        with open(self.fname_summary, "r") as infile:
            for line in infile:
                tmp = line.split()
                try:
                    if tmp[0] == "Energy":
                        self.Ex = []
                        self.levels = [] # [Ei, 2*J_initial, parity].
                        load_energy_levels(infile)
                    
                    elif tmp[0] == "B(E2)":
                        self.BE2 = []
                        if self.transitions is None:
                            self.transitions = []
                        load_transition_probabilities(infile, self.BE2)
                    
                    elif tmp[0] == "B(M1)":
                        self.BM1 = []
                        if self.transitions is None:
                            self.transitions = []
                        load_transition_probabilities(infile, self.BM1)
                
                except IndexError:
                    """
                    Skip blank lines.
                    """
                    continue

        self.levels = np.array(self.levels)
        self.transitions = np.array(self.transitions)
        self.Ex = np.array(self.Ex)
        self.BM1 = np.array(self.BM1)
        self.BE2 = np.array(self.BE2)
        self.debug = "DEBUG\n"
        self.debug += f"skipped -1 states in levels: {self.minus_one_spin_counts[0]}\n"
        self.debug += f"skipped -1 states in transitions: {self.minus_one_spin_counts[1]}\n"
        self.debug = np.array(self.debug)

        try:
            self.transitions_BE2 = self.transitions[:len(self.BE2)]
            self.transitions_BM1 = self.transitions[len(self.BE2):]
        except TypeError:
            """
            TypeError: len() of unsized object because self.BE2 = None.
            """
            self.transitions_BE2 = np.array(None)
            self.transitions_BM1 = np.array(None)

        if self.load_and_save_to_file:
            np.save(file=levels_fname, arr=self.levels, allow_pickle=True)
            np.save(file=transitions_fname, arr=self.transitions, allow_pickle=True)
            np.save(file=transitions_BM1_fname, arr=self.transitions_BM1, allow_pickle=True)
            np.save(file=transitions_BE2_fname, arr=self.transitions_BE2, allow_pickle=True)
            np.save(file=Ex_fname, arr=self.Ex, allow_pickle=True)
            np.save(file=BM1_fname, arr=self.BM1, allow_pickle=True)
            np.save(file=BE2_fname, arr=self.BE2, allow_pickle=True)
            np.save(file=debug_fname, arr=self.debug, allow_pickle=True)

    @property
    def help(self):
        """
        Generate a list of instance attributes without magic methods.

        Returns
        -------
        help_list : list
            A list of non-magic instance attributes.
        """
        help_list = []
        for elem in dir(self):
            if not elem.startswith("__"):   # Omit magic methods.
                help_list.append(elem)
        
        return help_list

def _process_kshell_output_in_parallel(args):
    """
    Simple wrapper for parallelizing loading of KSHELL files.
    """
    filepath, load_and_save_to_file = args
    print(filepath)
    return ReadKshellOutput(filepath, load_and_save_to_file)

def loadtxt(
    path: str,
    is_directory: bool = False,
    filter_: Union[None, str] = None,
    load_and_save_to_file: Union[bool, str] = True
    ) -> list:
    """
    Wrapper for using ReadKshellOutput class as a function.

    Parameters
    ----------
    path:
        Filename (and path) of KSHELL output data file, or path to
        directory containing sub-directories with KSHELL output data.
    
    is_directory:
        If True, and 'path' is a directory containing sub-directories
        with KSHELL data files, the contents of 'path' will be scanned
        for KSHELL data files. Currently supports only summary files.

    filter_:
        NOTE: Shouldnt the type be list, not str?

    load_and_save_to_file:
        Toggle saving data as .npy files on / off. If 'overwrite', saved
        .npy files are overwritten.

    Returns
    -------
    data : list
        List of instances with data from KSHELL data file as attributes.
    """
    all_fnames = None
    data = []
    if (is_directory) and (not os.path.isdir(path)):
        msg = f"{path} is not a directory"
        raise NotADirectoryError(msg)

    elif (not is_directory) and (not os.path.isfile(path)):
        msg = f"{path} is not a file"
        raise FileNotFoundError(msg)

    elif (is_directory) and (os.path.isdir(path)):
        all_fnames = {}

        for element in sorted(os.listdir(path)):
            """
            List all content in path.
            """
            if os.path.isdir(path + element):
                """
                If element is a directory, enter it to find data files.
                """
                all_fnames[element] = []    # Create blank list entry in dict for current element.
                for isotope in os.listdir(path + element):
                    """
                    List all content in the element directory.
                    """
                    if isotope.startswith("summary") and isotope.endswith(".txt"):
                        """
                        Extract summary data files.
                        """
                        try:
                            """
                            Example: O16.
                            """
                            n_neutrons = int(isotope[9:11])
                        except ValueError:
                            """
                            Example: Ne20.
                            """
                            n_neutrons = int(isotope[10:12])

                        n_neutrons -= atomic_numbers[element.split("_")[1]]
                        all_fnames[element].append([element + "/" + isotope, n_neutrons])
        
        pool = multiprocessing.Pool()
        for key in all_fnames:
            """
            Sort each list in the dict by the number of neutrons. Loop
            over all directories in 'all_fnames' and extract KSHELL data
            and append to a list.
            """
            if filter_ is not None:
                if key.split("_")[1] not in filter_:
                    """
                    Skip elements not in filter_.
                    """
                    continue

            all_fnames[key].sort(key=lambda tup: tup[1])   # Why not do this when directory is listed?
            sub_fnames = all_fnames[key]
            arg_list = [(path + i[0], load_and_save_to_file) for i in sub_fnames]
            data += pool.map(_process_kshell_output_in_parallel, arg_list)

    else:
        """
        Only a single KSHELL data file.
        """
        data.append(ReadKshellOutput(path, load_and_save_to_file))

    if not data:
        msg = "No KSHELL data loaded. Most likely error is that the given"
        msg += f" directory has no KSHELL data files. {path=}"
        raise RuntimeError(msg)

    return data

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
    Jpi_list: list,
    bin_width: Union[float, int],
    Ex_min: Union[float, int],
    Ex_max: Union[float, int],
    multipole_type: str = "M1"
    ) -> np.ndarray:
    """
    Author: Jørgen Midtbø.
    Modified by: Jon Dahl.
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
    #     B_pixel_count[i_Ex,i_Eg,Jpi_idx] += 1
    #     Ex_already_seen[i_Eg].append(Ex)

    NOTE: Ex_final or Ex_initial?

    Parameters
    ----------
    levels:
        Nx3 matrix containing [Ei, 2*J_initial, parity] in each row.

    transitions:
        Mx8 array containing [2J_final, parity_initial, Ex_final,
        2J_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)]

    Jpi_list:
        Set a spin window by defining a list of allowed initial
        [[spins, parities], ...]. NOTE: I think the purpose of the list
        is to give a unique index to each pair of [spin, parity].

    bin_width:
        The width of the energy bins. A bin width of 0.2 contains 20
        states of uniform spacing of 0.01.

    Ex_min:
        Lower limit for emitted gamma energy [MeV].

    Ex_max:
        Upper limit for emitted gamma energy [MeV].
        NOTE: If there are transitions with larger gamma energy than
        this, the program will crash (IndexError from i_Eg).
        TODO: Implement a way to check and skip these cases.

    multipole_type:
        Choose whether to calculate for 'M1' or 'E2'.

    Returns
    -------
    gSF_ExJpiavg:
        The gamma strength function.
    """
    n_bins = int(np.ceil(Ex_max/bin_width)) # Make sure the number of bins cover the whole Ex region.
    # bin_array = np.linspace(0, bin_width*n_bins, n_bins + 1) # Array of lower bin edge energy values
    # bin_array_middle = (bin_array[0: -1] + bin_array[1:])/2 # Array of middle bin values
    
    # Find index of first and last bin (lower bin edge) where we put counts.
    # It's important to not include the other Ex bins in the averaging later, because they contain zeros which will pull the average down.
    Ex_min_idx = int(np.floor(Ex_min/bin_width)) 
    Ex_max_idx = int(np.floor(Ex_max/bin_width))    

    prefactors = {   # Factor from the def. of the GSF.
        "M1": 11.5473e-9, # [1/(mu_N**2*MeV**2)].
        "E1": 1.047e-6
    }

    E_ground_state = levels[0, 0] # Read out the absolute ground state energy so we can get relative energies later.
    Ex_final = np.copy(transitions[:, 2])   # To avoid altering the raw data.
    
    if Ex_final[0] != 0:
        """
        Adjust energies relative to the ground state energy if they have
        not been adjusted already.
        """
        Ex_final -= E_ground_state

    """
    B_pixel_sum[Ex_final_idx, E_gamma_idx, Jpi_idx] contains the summed
    reduced transition probabilities for all transitions contained
    within the Ex_final_idx bin, E_gamma_idx bin, and Jpi_idx bin.
    B_pixel_counts counts the number of transitions within the same
    bins.
    """
    B_pixel_sum = np.zeros((n_bins, n_bins, len(Jpi_list)))     # Summed B(..) values for each pixel.
    B_pixel_count = np.zeros((n_bins, n_bins, len(Jpi_list)))   # The number of transitions.

    for transition_idx in range(len(transitions[:, 0])):
        """
        Iterate over all transitions in the transitions matrix and put
        in the correct pixel.
        """
        # Ex = transitions[transition_idx, 2]# - E_ground_state # Calculate energy relative to ground state.
        # print(f"{transition_idx=}")
        # print(f"{transitions[transition_idx, 2]=}")
        # print(f"{E_ground_state=}")
        # print(f"{Ex=}")
        # print(f"{Ex_min=}")
        # print(f"{Ex_max=}")
        # sys.exit()
        if (Ex_final[transition_idx] < Ex_min) or (Ex_final[transition_idx] >= Ex_max):
            """
            Check if transition is within min max limits, skip if not.
            NOTE: Is it correct to check the energy of the final state
            here? Why not the initial?
            """
            continue

        # Get bin index for Eg and Ex (initial). Indices are defined with respect to the lower bin edge.
        E_gamma_idx = int(np.floor(transitions[transition_idx, 6]/bin_width))
        Ex_final_idx = int(np.floor(Ex_final[transition_idx]/bin_width))

        # Read initial spin and parity of level: NOTE: I think the name / index is wrong. Or do I...?
        J_initial = int(transitions[transition_idx, 0])
        parity_initial = int(transitions[transition_idx, 1])
        try:
            """
            Get index for current [J_initial, parity_initial] combination in Jpi_list.
            """
            Jpi_idx = Jpi_list.index([J_initial, parity_initial])
        except ValueError:
            print("Transition skipped due to lack of Jpi.")
            continue

        try:
            """
            Add B(..) value and increment transition count,
            respectively. NOTE: Hope to remove this try-except by
            implementing suitable input checks to this function.
            """
            B_pixel_sum[Ex_final_idx, E_gamma_idx, Jpi_idx] += \
                transitions[transition_idx, 7]
            B_pixel_count[Ex_final_idx, E_gamma_idx, Jpi_idx] += 1
        except IndexError as err:
            print(err)
            print(f"{Ex_final_idx=}, {E_gamma_idx=}, {Jpi_idx=}, {transition_idx=}")
            print(f"{B_pixel_sum.shape}")
            print(f"{transitions.shape}")
            sys.exit()

    rho_ExJpi = np.zeros((n_bins, len(Jpi_list)))   # (Ex, Jpi) matrix to store level density
    for levels_idx in range(len(levels[:, 0])):
        """
        Count number of levels for each (Ex, J, parity_initial) pixel.
        """
        Ex, spin, parity_initial = levels[levels_idx]

        if (Ex - E_ground_state) >= Ex_max:
            """
            Skip if level is outside range.
            """
            continue

        Ex_idx = int(np.floor((Ex - E_ground_state)/bin_width))
        try:
            Jpi_idx = Jpi_list.index([spin, parity_initial])
        except ValueError:
            print("Transition skipped due to lack of Jpi.")
            continue
        
        rho_ExJpi[Ex_idx, Jpi_idx] += 1

    rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1.

    # Calculate gamma strength functions for each Ex, J, parity_initial individually, using the partial level density for each J, parity_initial.
    gSF = np.zeros((n_bins, n_bins, len(Jpi_list)))
    prefactor = prefactors[multipole_type] # mu_N^-2 MeV^-2, conversion constant
    for Jpi_idx in range(len(Jpi_list)):
        for Ex_idx in range(n_bins):
            gSF[Ex_idx, :, Jpi_idx] = prefactor*rho_ExJpi[Ex_idx, Jpi_idx]*div0(
                B_pixel_sum[Ex_idx, :, Jpi_idx],
                B_pixel_count[Ex_idx, :, Jpi_idx]
            )

    # Return the average gSF(Eg) over all (Ex,J,parity_initial)

    # return gSF[Ex_min_idx:Ex_max_idx+1,:,:].mean(axis=(0,2))
    # Update 20171009: Took proper care to only average over the non-zero f(Eg,Ex,J,parity_initial) pixels:
    gSF_currentExrange = gSF[Ex_min_idx:Ex_max_idx + 1, :, :]
    # print(f"{gSF=}")
    gSF_ExJpiavg = div0(
        gSF_currentExrange.sum(axis = (0, 2)),
        (gSF_currentExrange != 0).sum(axis = (0, 2))
    )
    return gSF_ExJpiavg

def level_plot(
    levels: np.ndarray,
    max_spin_states: int = 1_000,
    filter_spins: Union[None, list] = None
    ):
    """
    Generate a level plot for a single isotope. Spin on the x axis,
    energy on the y axis.

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