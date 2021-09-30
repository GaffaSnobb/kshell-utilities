import os, sys, multiprocessing
from fractions import Fraction
from typing import Union
import numpy as np
from .kshell_exceptions import DataStructureNotAccountedForError
from .general_utilities import level_plot, level_density

atomic_numbers = {
    "oxygen": 8, "fluorine": 9, "neon": 10, "sodium": 11, "magnesium": 12,
    "aluminium": 13, "silicon": 14, "phosphorus": 15, "sulfur": 16,
    "chlorine": 17, "argon": 18
}

atomic_numbers_reversed = {
    8: 'oxygen', 9: 'fluorine', 10: 'neon', 11: 'sodium', 12: 'magnesium',
    13: 'aluminium', 14: 'silicon', 15: 'phosphorus', 16: 'sulfur',
    17: 'chlorine', 18: 'argon'
}

def generate_states(
    start: int = 0,
    stop: int = 14,
    n_states: int = 100,
    parity: Union[str, int] = "both"
    ):
    """
    Generate correct string for input to kshell_ui.py when asked for
    which states to calculate.

    Examples
    --------
    For input start = 0, stop = 14, n_states = 200:
    0+200, 0.5+200, 1+200, 1.5+200, 2+200, 2.5+200, 3+200, 3.5+200, 4+200,
    4.5+200, 5+200, 5.5+200, 6+200, 6.5+200, 7+200, 7.5+200, 8+200, 8.5+200,
    9+200, 9.5+200, 10+200, 10.5+200, 11+200, 11.5+200, 12+200, 12.5+200,
    13+200, 13.5+200, 14+200, 14.5+200, 0-200, 0.5-200, 1-200, 1.5-200, 2-200,
    2.5-200, 3-200, 3.5-200, 4-200, 4.5-200, 5-200, 5.5-200, 6-200, 6.5-200,
    7-200, 7.5-200, 8-200, 8.5-200, 9-200, 9.5-200, 10-200, 10.5-200, 11-200,
    11.5-200, 12-200, 12.5-200, 13-200, 13.5-200, 14-200, 14.5-200,
    """
    allowed_positive_parity_inputs = ["positive", "pos", "+", "1", "+1", 1, "both"]
    allowed_negative_parity_inputs = ["negative", "neg", "-", "-1", -1, "both"]
    
    def correct_syntax(lst):
        for elem in lst:
            print(elem, end=", ")
    
    if parity in allowed_positive_parity_inputs:
        positive = [f"{i:g}{'+'}{n_states}" for i in np.arange(start, stop+0.5, 0.5)]
        correct_syntax(positive)
    
    if parity in allowed_negative_parity_inputs:
        negative = [f"{i:g}{'-'}{n_states}" for i in np.arange(start, stop+0.5, 0.5)]
        correct_syntax(negative)

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
        state. [[E, 2*spin, parity], ...].

    self.transitions:
        Mx8 array containing [2*spin_final, parity_initial, Ex_final,
        2*spin_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)]
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
                    spin_initial = float(Fraction(tmp[0][:parity_idx]))
                    
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
                        spin_final = float(Fraction(tmp[2].split(parity_symbol)[0]))
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
                        spin_final = float(Fraction(tmp[2][:-2]))
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
                        spin_final = float(Fraction(tmp[3].split(parity_symbol)[0]))
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
                        spin_final = float(Fraction(tmp[3][:-2]))
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
                        spin_final = float(Fraction(tmp[2].split(parity_symbol)[0]))
                        Ex_final = float(tmp[3])

                    else:
                        msg = "ERROR: Structure not accounted for!"
                        msg += f"\n{line=}"
                        raise DataStructureNotAccountedForError(msg)

                    if (spin_final == -1) or (spin_initial == -1):
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
                        2*spin_final, parity_initial, Ex_final, 2*spin_initial,
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
                        self.levels = [] # [Ei, 2*spin_initial, parity].
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

    def level_plot(self,
        max_spin_states: int = 1_000,
        filter_spins: Union[None, list] = None
        ):
        """
        Wrapper method to include level plot as an attribute to this
        class. Generate a level plot for a single isotope. Spin on the x
        axis, energy on the y axis.

        Parameters
        ----------        
        max_spin_states:
            The maximum amount of states to plot for each spin. Default
            set to a large number to indicate ≈ no limit.

        filter_spins:
            Which spins to include in the plot. If None, all spins are
            plotted.
        """
        level_plot(
            levels = self.levels,
            max_spin_states = max_spin_states,
            filter_spins = filter_spins
        )

    def level_density_plot(self,
            bin_size: Union[int, float]
        ):
        """
        Wrapper method to include level density plotting as
        an attribute to this class. Generate the level density with the
        input bin size.
        """
        level_density(
            energy_levels = self.levels[:, 0],
            bin_size = bin_size,
            plot = True,
            ax_input = None
        )

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
            if not elem.startswith("_"):   # Omit magic and private methods.
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
