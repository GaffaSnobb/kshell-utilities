from __future__ import annotations
import os, sys, multiprocessing, hashlib, ast, time, re
from fractions import Fraction
from typing import Union, Callable, Tuple, Iterable
from itertools import chain
import numpy as np
import numba
import matplotlib.pyplot as plt
import seaborn as sns
from .collect_logs import collect_logs
from .kshell_exceptions import KshellDataStructureError
from .parameters import atomic_numbers, flags
from .general_utilities import (
    level_plot, level_density, gamma_strength_function_average, porter_thomas,
    isotope
)
from .loaders import (
    _generic_loader, _load_energy_levels, _load_transition_probabilities,
    _load_transition_probabilities_old, _load_transition_probabilities_jem
)

def _generate_unique_identifier(path: str) -> str:
    """
    Generate a unique identifier based on the shell script and the
    save_input file from KSHELL.

    Parameters
    ----------
    path : str
        The path to a summary file or a directory with a summary file.
    """
    shell_file_content = ""
    save_input_content = ""
    msg = "Not able to generate unique identifier!"
    if os.path.isfile(path):
        """
        If a file is specified, extract the directory from the path.
        """
        directory = path.rsplit("/", 1)[0]
        if directory == path:
            """
            Example: path is 'summary.txt'
            """
            directory = "."
    
    elif os.path.isdir(path):
        directory = path

    for elem in os.listdir(directory):
        """
        Loop over all elements in the directory and find the shell
        script and save_input file.
        """
        try:
            if elem.endswith(".sh"):
                with open(f"{directory}/{elem}", "r") as infile:
                    shell_file_content += infile.read()
                    
            elif "save_input_ui.txt" in elem:
                with open(f"{directory}/{elem}", "r") as infile:
                    save_input_content += infile.read()
        except UnicodeDecodeError:
            msg = f"Skipping {elem} for tmp file unique identifier due to UnicodeDecodeError."
            msg += " Are you sure this file is supposed to be in this directory?"
            print(msg)
            continue

    if (shell_file_content == "") and (save_input_content == ""):
        print(msg)

    return hashlib.sha1((shell_file_content + save_input_content).encode()).hexdigest()

class ReadKshellOutput:
    """
    Read `KSHELL` data files and store the values as instance
    attributes.

    Attributes
    ----------
    levels : np.ndarray
        Array containing energy, spin, and parity for each excited
        state. [[E, 2*spin, parity, idx], ...]. idx counts how many
        times a state of that given spin and parity has occurred. The
        first 0+ state will have an idx of 1, the second 0+ will have an
        idx of 2, etc.

    transitions_BE1 : np.ndarray
        Transition data for BE1 transitions. Structure:
        NEW:
        [2*spin_initial, parity_initial, idx_initial, Ex_initial,
        2*spin_final, parity_final, idx_final, Ex_final, E_gamma,
        B(.., i->f), B(.., f<-i)]
        OLD NEW:
        [2*spin_initial, parity_initial, Ex_initial, 2*spin_final,
        parity_final, Ex_final, E_gamma, B(.., i->f), B(.., f<-i)]
        OLD:
        Mx8 array containing [2*spin_final, parity_initial, Ex_final,
        2*spin_initial, parity_initial, Ex_initial, E_gamma, B(.., i->f)].

    transitions_BM1 : np.ndarray
        Transition data for BM1 transitions. Same structure as BE1.

    transitions_BE2 : np.ndarray
        Transition data for BE2 transitions. Same structure as BE1.
    """
    def __init__(self, path: str, load_and_save_to_file: bool, old_or_new: str):
        """
        Parameters
        ----------
        path : string
            Path of `KSHELL` output file directory, or path to a
            specific `KSHELL` data file.

        load_and_save_to_file : bool
            Toggle saving data as `.npy` files on / off. If `overwrite`,
            saved `.npy` files are overwritten.

        old_or_new : str
            Choose between old and new summary file syntax. All summary
            files generated pre 2021-11-24 use old style.
            New:
            J_i  pi_i idx_i Ex_i    J_f  pi_f idx_f Ex_f      dE         B(E2)->         B(E2)->[wu]     B(E2)<-         B(E2)<-[wu]
            5    +    1     0.036   6    +    1     0.000     0.036     70.43477980      6.43689168     59.59865983      5.44660066
            Old:
            J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
            2+(11) 18.393 2+(10) 17.791 0.602 0.1(  0.0) 0.1( 0.0)
        """

        self.path = path.rstrip("/")    # Just in case, prob. not necessary.
        self.load_and_save_to_file = load_and_save_to_file
        self.old_or_new = old_or_new
        # Some attributes might not be altered, depending on the input file.
        # self.fname_ptn = None     # I'll fix .pnt reading when the functionality is actually needed.
        # self.fname_summary = None
        # self.proton_partition = None
        # self.neutron_partition = None
        # self.truncation = None
        self.nucleus = None
        self.interaction = None
        self.levels = None
        self.transitions_BM1 = None
        self.transitions_BE2 = None
        self.transitions_BE1 = None
        self.npy_path = "tmp"   # Directory for storing .npy files.
        # Debug.
        self.negative_spin_counts = np.array([0, 0, 0, 0])  # The number of skipped -1 spin states for [levels, BM1, BE2, BE1].

        if isinstance(self.load_and_save_to_file, str) and (self.load_and_save_to_file != "overwrite"):
            msg = "Allowed values for 'load_and_save_to_file' are: 'True', 'False', 'overwrite'."
            msg += f" Got '{self.load_and_save_to_file}'."
            raise ValueError(msg)

        if os.path.isdir(self.path):
            """
            If input 'path' is a directory. Look for summaries.
            """
            summaries = [i for i in os.listdir(self.path) if "summary" in i]

            if not summaries:
                """
                No summaries found, call collect_logs.
                """
                self.fname_summary = collect_logs(
                    path = self.path,
                    old_or_new = "both"
                )
                self.path_summary = f"{self.path}/{self.fname_summary}"

            elif len(summaries) > 1:
                msg = f"Several summaries found in {self.path}. Please specify path"
                msg += " directly to the summary file you want to load."
                raise RuntimeError(msg)

            else:
                self.fname_summary = summaries[0]   # Just filename.
                self.path_summary = f"{self.path}/{summaries[0]}"    # Complete path (maybe relative).

        elif os.path.isfile(self.path):
            """
            'path' is a single file, not a directory.
            """
            self.fname_summary = path.split("/")[-1]       # Just filename.
            self.path_summary = self.path    # Complete path (maybe relative).

        else:
            msg = f"{self.path} is not a file or a directory!"
            tmp_directory = self.path.rsplit('/', 1)[0]
            if os.path.isdir(tmp_directory):
                summaries = [i for i in os.listdir(tmp_directory) if "summary" in i]
                if summaries:
                    msg += f" {len(summaries)} summary files were found in {tmp_directory}."
                    msg += f" {summaries}"
                
                logs = [i for i in os.listdir(tmp_directory) if i.startswith("log_")]
                if logs and (not summaries):
                    msg += f" Logs found in '{tmp_directory}'. Set path to '{tmp_directory}'"
                    msg += " to collect logs and load the summary file."

            raise RuntimeError(msg)

        self.base_fname = self.fname_summary.split(".")[0] # Base filename for .npy tmp files.
        self.unique_id = _generate_unique_identifier(self.path) # Unique identifier for .npy files.
        self._extract_info_from_summary_fname()
        self._read_summary()
        self.mixing_pairs_BM1_BE2 = self._mixing_pairs(B_left="M1", B_right="E2")
        
        self.ground_state_energy = self.levels[0, 0]
        
        try:
            self.A = int("".join(filter(str.isdigit, self.nucleus)))
            self.Z, self.N = isotope(
                name = "".join(filter(str.isalpha, self.nucleus)).lower(),
                A = self.A
            )
        except ValueError:
            """
            Prob. because the summary filename does not contain the name
            of the isotope.
            """
            self.A = None
            self.Z = None
            self.N = None
        
        self.check_data()

    def _extract_info_from_ptn_fname(self):
        """
        Extract nucleus and model space name.
        """
        fname_split = self.fname_ptn.split("/")[-1]
        fname_split = fname_split.split("_")
        self.nucleus = fname_split[0]
        self.interaction = fname_split[1]

    def _read_ptn(self):
        """
        Read `KSHELL` partition file (.ptn) and extract proton
        partition, neutron partition, and particle-hole truncation data.
        Save as instance attributes.
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
        fname_split = fname_split.split(".")[0] # Remove .txt.
        fname_split = fname_split.split("_")
        self.nucleus = fname_split[1]
        self.interaction = fname_split[2]

    def _read_summary(self):
        """
        Read energy level data, transition probabilities and transition
        strengths from `KSHELL` output files.

        Raises
        ------
        KshellDataStructureError
            If the `KSHELL` file has unexpected structure / syntax.
        """
        # npy_path = "tmp"
        # base_fname = self.path.split("/")[-1][:-4]
        # unique_id = _generate_unique_identifier(self.path)

        if self.load_and_save_to_file:
            try:
                os.mkdir(self.npy_path)
            except FileExistsError:
                pass

            with open(f"{self.npy_path}/README.txt", "w") as outfile:
                msg = "This directory contains binary numpy data of KSHELL summary data."
                msg += " The purpose is to speed up subsequent runs which use the same summary data."
                msg += " It is safe to delete this entire directory if you have the original summary text file, "
                msg += "though at the cost of having to read the summary text file over again which may take some time."
                msg += " The ksutil.loadtxt parameter load_and_save_to_file = 'overwrite' will force a re-write of the binary numpy data."
                outfile.write(msg)
        
        levels_fname = f"{self.npy_path}/{self.base_fname}_levels_{self.unique_id}.npy"
        transitions_BM1_fname = f"{self.npy_path}/{self.base_fname}_transitions_BM1_{self.unique_id}.npy"
        transitions_BE2_fname = f"{self.npy_path}/{self.base_fname}_transitions_BE2_{self.unique_id}.npy"
        transitions_BE1_fname = f"{self.npy_path}/{self.base_fname}_transitions_BE1_{self.unique_id}.npy"
        debug_fname = f"{self.npy_path}/{self.base_fname}_debug_{self.unique_id}.npy"

        fnames = [
            levels_fname, transitions_BE2_fname, transitions_BM1_fname,
            transitions_BE1_fname, debug_fname
        ]

        if self.load_and_save_to_file != "overwrite":
            """
            Do not load files if overwrite parameter has been passed.
            """
            if all([os.path.isfile(fname) for fname in fnames]) and self.load_and_save_to_file:
                """
                If all files exist, load them. If any of the files do
                not exist, all will be generated.
                """
                self.levels = np.load(file=levels_fname, allow_pickle=True)
                self.transitions_BM1 = np.load(file=transitions_BM1_fname, allow_pickle=True)
                self.transitions_BE2 = np.load(file=transitions_BE2_fname, allow_pickle=True)
                self.transitions_BE1 = np.load(file=transitions_BE1_fname, allow_pickle=True)
                self.debug = np.load(file=debug_fname, allow_pickle=True)
                msg = "Summary data loaded from .npy!"
                msg += " Use loadtxt parameter load_and_save_to_file = 'overwrite'"
                msg += " to re-read data from the summary file."
                print(msg)
                return

        parallel_args = [
            [self.path_summary, "Energy", "replace_this_entry_with_loader", 0],
            [self.path_summary, "B(E1)", "replace_this_entry_with_loader", 1],
            [self.path_summary, "B(M1)", "replace_this_entry_with_loader", 2],
            [self.path_summary, "B(E2)", "replace_this_entry_with_loader", 3],
        ]

        if self.old_or_new == "new":
            parallel_args[0][2] = _load_energy_levels
            parallel_args[1][2] = _load_transition_probabilities
            parallel_args[2][2] = _load_transition_probabilities
            parallel_args[3][2] = _load_transition_probabilities

        elif self.old_or_new == "old":
            parallel_args[0][2] = _load_energy_levels
            parallel_args[1][2] = _load_transition_probabilities_old
            parallel_args[2][2] = _load_transition_probabilities_old
            parallel_args[3][2] = _load_transition_probabilities_old

        elif self.old_or_new == "jem":
            parallel_args[0][2] = _load_energy_levels
            parallel_args[1][2] = _load_transition_probabilities_jem
            parallel_args[2][2] = _load_transition_probabilities_jem
            parallel_args[3][2] = _load_transition_probabilities_jem

        if flags["parallel"]:
            with multiprocessing.Pool() as pool:
                pool_res = pool.map(_generic_loader, parallel_args)
                self.levels, self.negative_spin_counts[0] = pool_res[0]
                self.transitions_BE1, self.negative_spin_counts[1] = pool_res[1]
                self.transitions_BM1, self.negative_spin_counts[2] = pool_res[2]
                self.transitions_BE2, self.negative_spin_counts[3] = pool_res[3]
        else:
            self.levels, self.negative_spin_counts[0] = _generic_loader(parallel_args[0])
            self.transitions_BE1, self.negative_spin_counts[1] = _generic_loader(parallel_args[1])
            self.transitions_BM1, self.negative_spin_counts[2] = _generic_loader(parallel_args[2])
            self.transitions_BE2, self.negative_spin_counts[3] = _generic_loader(parallel_args[3])

        self.levels = np.array(self.levels)
        self.transitions_BE1 = np.array(self.transitions_BE1)
        self.transitions_BM1 = np.array(self.transitions_BM1)
        self.transitions_BE2 = np.array(self.transitions_BE2)
        self.debug = "DEBUG\n"
        self.debug += f"skipped -1 states in levels: {self.negative_spin_counts[0]}\n"
        self.debug += f"skipped -1 states in BE1: {self.negative_spin_counts[1]}\n"
        self.debug += f"skipped -1 states in BM1: {self.negative_spin_counts[2]}\n"
        self.debug += f"skipped -1 states in BE2: {self.negative_spin_counts[3]}\n"
        self.debug = np.array(self.debug)

        if self.old_or_new == "jem":
            """
            'jem style' summary syntax lists all initial and final
            excitation energies in transitions as absolute values.
            Subtract the ground state energy to get the relative
            energies to match the newer KSHELL summary file syntax.
            """
            msg = "The issue of E_final > E_initial must be figured out before"
            msg += " JEM style syntax can be used!"
            raise NotImplementedError(msg)
            E_gs = abs(self.levels[0, 0])   # Can prob. just use ... -= E_gs
            try:
                self.transitions_BM1[:, 3] = E_gs - np.abs(self.transitions_BM1[:, 3])
                self.transitions_BM1[:, 7] = E_gs - np.abs(self.transitions_BM1[:, 7])
            except IndexError:
                """
                No BM1 transitions.
                """
                pass
            try:
                self.transitions_BE1[:, 3] = E_gs - np.abs(self.transitions_BE1[:, 3])
                self.transitions_BE1[:, 7] = E_gs - np.abs(self.transitions_BE1[:, 7])
            except IndexError:
                """
                No BE1 transitions.
                """
                pass
            try:
                self.transitions_BE2[:, 3] = E_gs - np.abs(self.transitions_BE2[:, 3])
                self.transitions_BE2[:, 7] = E_gs - np.abs(self.transitions_BE2[:, 7])
            except IndexError:
                """
                No BE2 transitions.
                """
                pass

            self.levels[:, 1] /= 2  # JEM style syntax has 2*J already. Without this correction it would be 4*J.

        if self.load_and_save_to_file:
            np.save(file=levels_fname, arr=self.levels, allow_pickle=True)
            np.save(file=transitions_BM1_fname, arr=self.transitions_BM1, allow_pickle=True)
            np.save(file=transitions_BE2_fname, arr=self.transitions_BE2, allow_pickle=True)
            np.save(file=transitions_BE1_fname, arr=self.transitions_BE1, allow_pickle=True)
            np.save(file=debug_fname, arr=self.debug, allow_pickle=True)

    def level_plot(self,
        include_n_levels: int = 1000,
        filter_spins: Union[None, list] = None,
        filter_parity: Union[None, str] = None,
        color: Union[None, str] = "black"
        ):
        """
        Wrapper method to include level plot as an attribute to this
        class. Generate a level plot for a single isotope. Angular
        momentum on the x axis, energy on the y axis.

        Parameters
        ----------
        include_n_levels : int
            The maximum amount of states to plot for each spin. Default
            set to a large number to indicate ≈ no limit.

        filter_spins : Union[None, list]
            Which spins to include in the plot. If `None`, all spins are
            plotted. Defaults to `None`

        color : Union[None, str]
            Set the color of the level lines.
        """
        level_plot(
            levels = self.levels,
            include_n_levels = include_n_levels,
            filter_spins = filter_spins,
            filter_parity = filter_parity,
            color = color,
        )

    def level_density_plot(self,
            bin_width: Union[int, float] = 0.2,
            include_n_levels: Union[None, int] = None,
            filter_spins: Union[None, int, list] = None,
            filter_parity: Union[None, str, int] = None,
            return_counts: bool = False,
            E_min: Union[float, int] = 0,
            E_max: Union[float, int] = np.inf,
            plot: bool = True,
            save_plot: bool = False
        ):
        """
        Wrapper method to include level density plotting as
        an attribute to this class. Generate the level density with the
        input bin size.

        Parameters
        ----------
        See level_density in general_utilities.py for parameter
        information.
        """
        bins, density = level_density(
            levels = self.levels,
            bin_width = bin_width,
            include_n_levels = include_n_levels,
            filter_spins = filter_spins,
            filter_parity = filter_parity,
            return_counts = return_counts,
            E_min = E_min,
            E_max = E_max,
            plot = plot,
            save_plot = save_plot
        )

        return bins, density

    def nld(self,
        bin_width: Union[int, float] = 0.2,
        include_n_levels: Union[None, int] = None,
        filter_spins: Union[None, int, list] = None,
        filter_parity: Union[None, str, int] = None,
        E_min: Union[float, int] = 0,
        E_max: Union[float, int] = np.inf,
        return_counts: bool = False,
        plot: bool = True,
        save_plot: bool = False
        ):
        """
        Wrapper method to level_density_plot.
        """
        return self.level_density_plot(
            bin_width = bin_width,
            include_n_levels = include_n_levels,
            filter_spins = filter_spins,
            filter_parity = filter_parity,
            E_min = E_min,
            E_max = E_max,
            return_counts = return_counts,
            plot = plot,
            save_plot = save_plot
        )

    def gamma_strength_function_average_plot(self,
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
        plot: bool = True,
        save_plot: bool = False
        ):
        """
        Wrapper method to include gamma ray strength function
        calculations as an attribute to this class. Includes saving
        of GSF data to .npy files.

        Parameters
        ----------
        See gamma_strength_function_average in general_utilities.py
        for parameter descriptions.
        """
        transitions_dict = {
            "M1": self.transitions_BM1,
            "E2": self.transitions_BE2,
            "E1": self.transitions_BE1
        }
        is_loaded = False
        gsf_unique_string = f"{bin_width}{Ex_min}{Ex_max}{multipole_type}"
        gsf_unique_string += f"{prefactor_E1}{prefactor_M1}{prefactor_E2}"
        gsf_unique_string += f"{initial_or_final}{partial_or_total}{include_only_nonzero_in_average}"
        gsf_unique_string += f"{include_n_levels}{filter_spins}{filter_parities}"
        gsf_unique_id = hashlib.sha1((gsf_unique_string).encode()).hexdigest()
        gsf_fname = f"{self.npy_path}/{self.base_fname}_gsf_{gsf_unique_id}_{self.unique_id}.npy"
        bins_fname = f"{self.npy_path}/{self.base_fname}_gsfbins_{gsf_unique_id}_{self.unique_id}.npy"
        n_transitions_fname = f"{self.npy_path}/{self.base_fname}_gsfntransitions_{gsf_unique_id}_{self.unique_id}.npy"
        
        fnames = [gsf_fname, bins_fname]
        if return_n_transitions:
            fnames.append(n_transitions_fname)
        
        if all([os.path.isfile(fname) for fname in fnames]) and self.load_and_save_to_file and (self.load_and_save_to_file != "overwrite"):
            """
            If all these conditions are met, all arrays will be loaded
            from file. If any of these conditions are NOT met, all
            arrays will be re-calculated.
            """
            gsf = np.load(file=gsf_fname, allow_pickle=True)
            bins = np.load(file=bins_fname, allow_pickle=True)
            if return_n_transitions:
                n_transitions = np.load(file=n_transitions_fname, allow_pickle=True)
            
            msg = f"{self.nucleus} {multipole_type} GSF data loaded from .npy!"
            print(msg)
            is_loaded = True

        else:
            tmp = gamma_strength_function_average(
                levels = self.levels,
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
                return_n_transitions = return_n_transitions,
                # plot = plot,
                # save_plot = save_plot
            )
            if return_n_transitions:
                bins, gsf, n_transitions = tmp
            else:
                bins, gsf = tmp

        if self.load_and_save_to_file and not is_loaded:
            np.save(file=gsf_fname, arr=gsf, allow_pickle=True)
            np.save(file=bins_fname, arr=bins, allow_pickle=True)
            
            if return_n_transitions:
                np.save(file=n_transitions_fname, arr=n_transitions, allow_pickle=True)

        if plot:
            unit_exponent = 2*int(multipole_type[-1]) + 1
            fig, ax = plt.subplots()
            ax.plot(bins, gsf, label=multipole_type.upper(), color="black")
            ax.legend()
            ax.grid()
            ax.set_xlabel(r"E$_{\gamma}$ [MeV]")
            ax.set_ylabel(f"$\gamma$SF [MeV$^-$$^{unit_exponent}$]")
            if save_plot:
                fname = f"gsf_{multipole_type}.png"
                print(f"GSF saved as '{fname}'")
                fig.savefig(fname=fname, dpi=300)
            plt.show()

        if return_n_transitions:
            return bins, gsf, n_transitions
        else:
            return bins, gsf

    def gsf(self,
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
        plot: bool = True,
        save_plot: bool = False
        ):
        """
        Alias for gamma_strength_function_average_plot. See that
        docstring for details.
        """
        return self.gamma_strength_function_average_plot(
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
            return_n_transitions = return_n_transitions,
            plot = plot,
            save_plot = save_plot
        )

    def porter_thomas(self, multipole_type: str, **kwargs):
        """
        Wrapper for general_utilities.porter_thomas. See that docstring
        for details.

        Parameters
        ----------
        multipole_type : str
            Choose the multipolarity of the transitions. 'E1', 'M1',
            'E2'.
        """
        transitions_dict = {
            "E1": self.transitions_BE1,
            "M1": self.transitions_BM1,
            "E2": self.transitions_BE2,
        }
        
        return porter_thomas(transitions_dict[multipole_type], **kwargs)

    def porter_thomas_Ei_plot(self,
        Ei_range_min: float = 5,
        Ei_range_max: float = 9,
        Ei_values: Union[list, None] = None,
        Ei_bin_width: float = 0.2,
        BXL_bin_width: float = 0.1,
        multipole_type: Union[str, list] = "M1",
        set_title: bool = True
        ):
        """
        Porter-Thomas analysis of the reduced transition probabilities
        for different initial excitation energies. Produces a figure
        very similar to fig. 3.3 in JEM PhD thesis:
        http://urn.nb.no/URN:NBN:no-79895.

        Parameters
        ----------
        Ei_range_min : float
            Minimum value of the initial energy range. Three equally
            spaced intervals will be chosen from this range. Be sure to
            choose this value to be above the discrete region. MeV.
        
        Ei_range_max : float
            Maximum value of the initial energy range. Three equally
            spaced intervals will be chosen from this range. The neutron
            separation energy is a good choice. MeV.

        Ei_values : Union[list, None]
            List of initial energies to be used. If None, the
            initial energies will be chosen from the Ei_range_min
            and Ei_range_max. Values in a bin around Ei_values of size
            Ei_bin_width will be used. Max 3 values allowed. MeV.

        Ei_bin_width : float
            Width of the initial energy bins. MeV.

        BXL_bin_width : float
            Width of the BXL bins when the BXL/mean(BXL) values are
            counted. Unitless.

        multipole_type : str
            Choose the multipolarity of the transitions. 'E1', 'M1',
            'E2'.

        set_title: bool
            Toggle figure title on / off. Defaults to True (on).
        """

        if Ei_values is None:
            """
            Defaults to a range defined by Ei_range_min and Ei_range_max.
            """
            Ei_values = np.linspace(Ei_range_min, Ei_range_max, 3)
        
        if len(Ei_values) > 3:
            raise ValueError("Ei_values must be a list of length <= 3.")

        if isinstance(multipole_type, str):
            multipole_type = [multipole_type]
        
        elif isinstance(multipole_type, list):
            pass
        
        else:
            msg = f"multipole_type must be str or list. Got {type(multipole_type)}."
            raise TypeError(msg)

        colors = ["blue", "royalblue", "lightsteelblue"]
        Ei_range = np.linspace(Ei_range_min, Ei_range_max, 4)
        
        if len(multipole_type) == 1:
            fig, axd = plt.subplot_mosaic(
                [['upper'], ['middle'], ['lower']],
                gridspec_kw = dict(height_ratios=[1, 1, 0.7]),
                figsize = (6.4, 8),
                constrained_layout = True,
                sharex = True
            )

            for Ei, color in zip(Ei_values, colors):
                """
                Calculate in a bin size of 'Ei_bin_width' around given Ei
                values.
                """
                bins, counts, chi2 = self.porter_thomas(
                    multipole_type = multipole_type[0],
                    Ei = Ei,
                    BXL_bin_width = BXL_bin_width,
                    Ei_bin_width = Ei_bin_width,
                    return_chi2 = True
                )
                idx = np.argmin(np.abs(bins - 10))  # Slice the arrays at approx 10.
                bins = bins[:idx]
                counts = counts[:idx]
                chi2 = chi2[:idx]
                axd["upper"].step(
                    bins,
                    counts,
                    label = r"$E_i = $" + f"{Ei:.2f}" + r" $\pm$ " + f"{Ei_bin_width/2:.2f} MeV",
                    color = color
                )
        
            axd["upper"].plot(
                bins,
                chi2,
                color = "tab:green",
                label = r"$\chi_{\nu = 1}^2$"
            )
            axd["upper"].legend(loc="upper right")
            axd["upper"].set_ylabel(r"Normalised counts")

            for i, color in enumerate(colors):
                """
                Calculate in the specified range of Ei values.
                """
                bins, counts, chi2 = self.porter_thomas(
                    multipole_type = multipole_type[0],
                    Ei = [Ei_range[i], Ei_range[i+1]],
                    BXL_bin_width = BXL_bin_width,
                    return_chi2 = True
                )
                
                idx = np.argmin(np.abs(bins - 10))
                bins = bins[:idx]
                counts = counts[:idx]
                chi2 = chi2[:idx]
                
                axd["middle"].step(
                    bins,
                    counts,
                    color = color,
                    label = r"$E_i = $" + f"[{Ei_range[i]:.2f}, {Ei_range[i+1]:.2f}] MeV"
                )
                axd["lower"].step(
                    bins,
                    counts/chi2,
                    color = color,
                    label = r"($E_i = $" + f"[{Ei_range[i]:.2f}, {Ei_range[i+1]:.2f}] MeV)" + r"$/\chi_{\nu = 1}^2$",
                )

            axd["middle"].plot(bins, chi2, color="tab:green", label=r"$\chi_{\nu = 1}^2$")
            axd["middle"].legend(loc="upper right")
            axd["middle"].set_ylabel(r"Normalised counts")

            axd["lower"].hlines(y=1, xmin=bins[0], xmax=bins[-1], linestyle="--", color="black")
            axd["lower"].set_xlabel(r"$B(M1)/\langle B(M1) \rangle$")
            axd["lower"].legend(loc="upper left")
            axd["lower"].set_ylabel(r"Relative error")
            axd["lower"].set_xlabel(
                r"$B(" + f"{multipole_type[0]}" + r")/\langle B(" + f"{multipole_type[0]}" + r") \rangle$"
            )
            if set_title:
                axd["upper"].set_title(
                    f"{self.nucleus_latex}, {self.interaction}, " + r"$" + f"{multipole_type[0]}" + r"$"
                )
            fig.savefig(fname=f"{self.nucleus}_porter_thomas_Ei_{multipole_type[0]}.png", dpi=300)
        
        elif len(multipole_type) == 2:
            fig, axd = plt.subplot_mosaic([
                    ['upper left', 'upper right'],
                    ['middle left', 'middle right'],
                    ['lower left', 'lower right']
                ],
                gridspec_kw = dict(height_ratios=[1, 1, 0.7]),
                figsize = (10, 8),
                constrained_layout = True,
                sharex = True
            )

            for multipole_type_, loc in zip(multipole_type, ["left", "right"]):
                for Ei, color in zip(Ei_values, colors):
                    """
                    Calculate in a bin size of 'Ei_bin_width' around given Ei
                    values.
                    """
                    bins, counts, chi2 = self.porter_thomas(
                        multipole_type = multipole_type_,
                        Ei = Ei,
                        BXL_bin_width = BXL_bin_width,
                        Ei_bin_width = Ei_bin_width,
                        return_chi2 = True
                    )
                    idx = np.argmin(np.abs(bins - 10))  # Slice the arrays at approx 10.
                    bins = bins[:idx]
                    counts = counts[:idx]
                    chi2 = chi2[:idx]
                    axd["upper " + loc].step(
                        bins,
                        counts,
                        label = r"$E_i = $" + f"{Ei:.2f}" + r" $\pm$ " + f"{Ei_bin_width/2:.2f} MeV",
                        color = color
                    )
            
                axd["upper " + loc].plot(
                    bins,
                    chi2,
                    color = "tab:green",
                    label = r"$\chi_{\nu = 1}^2$"
                )

                for i, color in enumerate(colors):
                    """
                    Calculate in the specified range of Ei values.
                    """
                    bins, counts, chi2 = self.porter_thomas(
                        multipole_type = multipole_type_,
                        Ei = [Ei_range[i], Ei_range[i+1]],
                        BXL_bin_width = BXL_bin_width,
                        return_chi2 = True
                    )
                    
                    idx = np.argmin(np.abs(bins - 10))
                    bins = bins[:idx]
                    counts = counts[:idx]
                    chi2 = chi2[:idx]
                    
                    axd["middle " + loc].step(
                        bins,
                        counts,
                        color = color,
                        label = r"$E_i = $" + f"[{Ei_range[i]:.2f}, {Ei_range[i+1]:.2f}] MeV"
                    )
                    axd["lower " + loc].step(
                        bins,
                        counts/chi2,
                        color = color,
                        label = r"($E_i = $" + f"[{Ei_range[i]:.2f}, {Ei_range[i+1]:.2f}] MeV)" + r"$/\chi_{\nu = 1}^2$",
                    )

                axd["middle " + loc].plot(bins, chi2, color="tab:green", label=r"$\chi_{\nu = 1}^2$")

                axd["lower " + loc].hlines(y=1, xmin=bins[0], xmax=bins[-1], linestyle="--", color="black")
                # axd["lower " + loc].set_xlabel(r"$B(M1)/\langle B(M1) \rangle$")
                axd["lower " + loc].set_xlabel(
                    r"$B(" + f"{multipole_type_}" + r")/\langle B(" + f"{multipole_type_}" + r") \rangle$"
                )
                if set_title:
                    axd["upper " + loc].set_title(
                        f"{self.nucleus_latex}, {self.interaction}, " + r"$" + f"{multipole_type_}" + r"$"
                    )
            
            axd["lower left"].legend(loc="upper left")
            axd["middle left"].legend(loc="upper right")
            axd["middle left"].set_ylabel(r"Normalised counts")
            axd["lower left"].set_ylabel(r"Relative error")
            axd["upper left"].legend(loc="upper right")
            axd["upper left"].set_ylabel(r"Normalised counts")
            
            axd["upper right"].set_yticklabels([])
            axd["upper right"].set_ylim(axd["upper left"].get_ylim())

            axd["middle right"].set_yticklabels([])
            axd["middle right"].set_ylim(axd["middle left"].get_ylim())

            axd["lower right"].set_yticklabels([])
            axd["lower right"].set_ylim(axd["lower left"].get_ylim())
            fig.savefig(fname=f"{self.nucleus}_porter_thomas_Ei_{multipole_type[0]}_{multipole_type[1]}.png", dpi=300)

        else:
            msg = "Only 1 or 2 multipole types may be given at the same time!"
            msg += f" Got {len(multipole_type)}."
            raise ValueError(msg)
        
        plt.show()

    def _porter_thomas_j_plot_calculator(self,
        Ex_min: float,
        Ex_max: float,
        j_lists: Union[list, None],
        BXL_bin_width: float,
        multipole_type: str,
        ):
        """
        Really just a wrapper to self.porter_thomas with j checks.

        Parameters
        ----------
        Ex_min : float
            Minimum value of the initial energy. MeV.
        
        Ex_max : float
            Maximum value of the initial energy. MeV.

        j_lists : Union[list, None]
            Either a list of j values to compare, a list of lists of j
            values to compare, or None where all j values available
            will be used.

        BXL_bin_width : float
            Width of the BXL bins when the BXL/mean(BXL) values are
            counted. Unitless.

        multipole_type : str
            Choose the multipolarity of the transitions. 'E1', 'M1',
            'E2'.
        """        
        if isinstance(j_lists, list):
            if not j_lists:
                msg = "Please provide a list of j values or a list of lists of j values."
                raise ValueError(msg)

        else:
            msg = f"j_lists must be a list. Got {type(j_lists)}."
            raise TypeError(msg)
        
        if all(isinstance(j, list) for j in j_lists):
            """
            All entries in j_lists are lists.
            """
            pass
        
        elif any(isinstance(j, list) for j in j_lists):
            """
            Only some of the entries are lists. The case where all
            entries are lists will be captured by the previous check.
            """
            msg = "j_lists cant contain a mix of lists and numbers!"
            raise TypeError(msg)

        else:
            """
            None of the entries are lists. Combine all numbers as a
            single list inside j_lists.
            """
            if all(isinstance(j, (int, float)) for j in j_lists):
                j_lists = [j_lists]
            else:
                msg = "All entries in j_lists must either all be lists or all be numbers!"
                raise TypeError(msg)

        if (j_lists_len := len(j_lists)) > 3:
            msg = f"j_lists cannot contain more than 3 ranges of j values. Got {j_lists_len}."
            raise ValueError(msg)

        if Ex_min > Ex_max:
            msg = "Ex_min cannot be larger than Ex_max!"
            raise ValueError(msg)

        if (Ex_min < 0) or (Ex_max < 0):
            msg = "Ex_min and Ex_max cannot be negative!"
            raise ValueError(msg)
        
        binss = []
        countss = []
        chi2s = []
        for j_list in j_lists:
            """
            Calculate for the j values in j_list (note: not in j_lists).
            """
            bins, counts, chi2 = self.porter_thomas(
                multipole_type = multipole_type,
                j_list = j_list,
                Ei = [Ex_min, Ex_max],
                BXL_bin_width = BXL_bin_width,
                return_chi2 = True
            )
            idx = np.argmin(np.abs(bins - 10))  # Slice the arrays at approx 10.
            # bins = bins[:idx]
            # counts = counts[:idx]
            # chi2 = chi2[:idx]
            binss.append(bins[:idx])
            countss.append(counts[:idx])
            chi2s.append(chi2[:idx])

        return binss, countss, chi2s

    def porter_thomas_j_plot(self,
        Ex_min: float = 5,
        Ex_max: float = 9,
        j_lists: Union[list, None] = None,
        BXL_bin_width: float = 0.1,
        multipole_type: Union[str, list] = "M1",
        include_relative_difference: bool = True,
        set_title: bool = True
        ):
        """
        Porter-Thomas analysis of the reduced transition probabilities
        for different angular momenta.

        Parameter
        ---------
        multipole_type : Union[str, list]
            Choose the multipolarity of the transitions. 'E1', 'M1',
            'E2'. Accepts a list of max 2 multipolarities.

        set_title : bool
            Toggle plot title on / off.

        See the docstring of _porter_thomas_j_plot_calculator for the
        rest of the descriptions.
        """
        if j_lists is None:
            j_list_default = True
        else:
            j_list_default = False

        transitions_dict = {
            "E1": self.transitions_BE1,
            "M1": self.transitions_BM1,
            "E2": self.transitions_BE2,
        }

        colors = ["blue", "royalblue", "lightsteelblue"]
        if isinstance(multipole_type, str):
            multipole_type = [multipole_type]
        
        elif isinstance(multipole_type, list):
            pass
        
        else:
            msg = f"multipole_type must be str or list. Got {type(multipole_type)}."
            raise TypeError(msg)

        if len(multipole_type) == 1:
            if j_list_default:
                """
                Default j_lists values.
                """
                j_lists = []
                for elem in np.unique(transitions_dict[multipole_type[0]][:, 0]):
                    j_lists.append([elem/2])

                j_lists = j_lists[:3]   # _porter_thomas_j_plot_calculator supports max. 3 lists of j values.

            if include_relative_difference:
                fig, axd = plt.subplot_mosaic(
                    [['upper'], ['lower']],
                    gridspec_kw = dict(height_ratios=[1, 0.5]),
                    figsize = (6.4, 8),
                    constrained_layout = True,
                    sharex = True
                )
            else:
                fig, axd = plt.subplot_mosaic(
                    [['upper']],
                    gridspec_kw = dict(height_ratios=[1]),
                    # figsize = (6.4, 8),
                    constrained_layout = True,
                    sharex = True
                )

            binss, countss, chi2s = self._porter_thomas_j_plot_calculator(
                Ex_min = Ex_min,
                Ex_max = Ex_max,
                j_lists = j_lists,
                BXL_bin_width = BXL_bin_width,
                multipole_type = multipole_type[0],
            )

            for bins, counts, chi2, j_list, color in zip(binss, countss, chi2s, j_lists, colors):
                axd["upper"].step(
                    bins,
                    counts,
                    label = r"$j_i = $" + f"{j_list}",
                    color = color
                )
                if include_relative_difference:
                    axd["lower"].step(
                        bins,
                        counts/chi2,
                        color = color,
                        label = r"($j_i = $" + f"{j_list})" + r"$/\chi_{\nu = 1}^2$",
                    )
        
            axd["upper"].plot(
                bins,
                chi2,
                color = "tab:green",
                label = r"$\chi_{\nu = 1}^2$"
            )
            axd["upper"].legend(loc="upper right")
            axd["upper"].set_ylabel(r"Normalised counts")
            if set_title:
                axd["upper"].set_title(
                    f"{self.nucleus_latex}, {self.interaction}, " + r"$" + f"{multipole_type[0]}" + r"$"
                )

            if include_relative_difference:
                axd["lower"].hlines(y=1, xmin=bins[0], xmax=bins[-1], linestyle="--", color="black")
                axd["lower"].legend(loc="upper left")
                axd["lower"].set_ylabel(r"Relative error")
                axd["lower"].set_xlabel(
                    r"$B(" + f"{multipole_type[0]}" + r")/\langle B(" + f"{multipole_type[0]}" + r") \rangle$"
                )
            else:
                axd["upper"].set_xlabel(
                    r"$B(" + f"{multipole_type[0]}" + r")/\langle B(" + f"{multipole_type[0]}" + r") \rangle$"
                )

            fig.savefig(fname=f"{self.nucleus}_porter_thomas_j_{multipole_type[0]}.png", dpi=300)

        elif len(multipole_type) == 2:
            if j_list_default:
                """
                Default j_lists values.
                """
                j_lists = []
                for elem in np.unique(transitions_dict[multipole_type[0]][:, 0]):
                    j_lists.append([elem/2])

                j_lists = j_lists[:3]   # _porter_thomas_j_plot_calculator supports max. 3 lists of j values.

            fig, axd = plt.subplot_mosaic(
                [['upper left', 'upper right'], ['lower left', 'lower right']],
                gridspec_kw = dict(height_ratios=[1, 0.5]),
                figsize = (10, 8),
                constrained_layout = True,
                sharex = True
            )
            binss, countss, chi2s = self._porter_thomas_j_plot_calculator(
                Ex_min = Ex_min,
                Ex_max = Ex_max,
                j_lists = j_lists,
                BXL_bin_width = BXL_bin_width,
                multipole_type = multipole_type[0],
            )

            for bins, counts, chi2, j_list, color in zip(binss, countss, chi2s, j_lists, colors):
                axd["upper left"].step(
                    bins,
                    counts,
                    label = r"$j_i = $" + f"{j_list}",
                    color = color
                )
                axd["lower left"].step(
                    bins,
                    counts/chi2,
                    color = color,
                    label = r"($j_i = $" + f"{j_list})" + r"$/\chi_{\nu = 1}^2$",
                )
        
            axd["upper left"].plot(
                bins,
                chi2,
                color = "tab:green",
                label = r"$\chi_{\nu = 1}^2$"
            )
            axd["upper left"].legend(loc="upper right")
            axd["upper left"].set_ylabel(r"Normalised counts")

            axd["lower left"].hlines(y=1, xmin=bins[0], xmax=bins[-1], linestyle="--", color="black")
            axd["lower left"].set_xlabel(
                r"$B(" + f"{multipole_type[0]}" + r")/\langle B(" + f"{multipole_type[0]}" + r") \rangle$"
            )
            axd["lower left"].legend(loc="upper left")
            axd["lower left"].set_ylabel(r"Relative error")
            if set_title:
                axd["upper left"].set_title(
                    f"{self.nucleus_latex}, {self.interaction}, " + r"$" + f"{multipole_type[0]}" + r"$"
                )

            if j_list_default:
                """
                Default j_lists values.
                """
                j_lists = []
                for elem in np.unique(transitions_dict[multipole_type[1]][:, 0]):
                    j_lists.append([elem/2])

                j_lists = j_lists[:3]   # _porter_thomas_j_plot_calculator supports max. 3 lists of j values.

            binss, countss, chi2s = self._porter_thomas_j_plot_calculator(
                Ex_min = Ex_min,
                Ex_max = Ex_max,
                j_lists = j_lists,
                BXL_bin_width = BXL_bin_width,
                multipole_type = multipole_type[1],
            )

            for bins, counts, chi2, j_list, color in zip(binss, countss, chi2s, j_lists, colors):
                axd["upper right"].step(
                    bins,
                    counts,
                    label = r"$j_i = $" + f"{j_list}",
                    color = color
                )
                axd["lower right"].step(
                    bins,
                    counts/chi2,
                    color = color,
                    label = r"($j_i = $" + f"{j_list})" + r"$/\chi_{\nu = 1}^2$",
                )
        
            axd["upper right"].plot(
                bins,
                chi2,
                color = "tab:green",
                label = r"$\chi_{\nu = 1}^2$"
            )
            # axd["upper right"].legend(loc="upper right")
            axd["upper right"].set_yticklabels([])
            axd["upper right"].set_ylim(axd["upper left"].get_ylim())

            axd["lower right"].hlines(y=1, xmin=bins[0], xmax=bins[-1], linestyle="--", color="black")
            axd["lower right"].set_xlabel(
                r"$B(" + f"{multipole_type[1]}" + r")/\langle B(" + f"{multipole_type[1]}" + r") \rangle$"
            )
            # axd["lower right"].legend(loc="upper left")
            axd["lower right"].set_yticklabels([])
            axd["lower right"].set_ylim(axd["lower left"].get_ylim())
            if set_title:
                axd["upper right"].set_title(
                    f"{self.nucleus_latex}, {self.interaction}, " + r"$" + f"{multipole_type[1]}" + r"$"
                )
            fig.savefig(fname=f"{self.nucleus}_porter_thomas_j_{multipole_type[0]}_{multipole_type[1]}.png", dpi=300)
        else:
            msg = "Only 1 or 2 multipole types may be given at the same time!"
            msg += f" Got {len(multipole_type)}."
            raise ValueError(msg)

        plt.show()

    def angular_momentum_distribution_plot(self,
        bin_width: float = 0.2,
        E_min: float = 5,
        E_max: float = 10,
        j_list: Union[None, int, float, Iterable] = None,
        filter_parity: Union[None, int, str] = None,
        plot: bool = True,
        # single_spin_plot: Union[None, int, float, Iterable] = None,
        save_plot: bool = True,
        set_title: bool = True
    ):
        """
        Plot the angular momentum distribution of the levels.

        Parameters
        ----------
        bin_width : float
            Width of the energy bins. MeV.

        E_min : float
            Minimum value of the energy range.

        E_max : float
            Maximum value of the energy range.

        j_list : Union[None, int, float, Iterable]
            Filter the levels by their angular momentum. If None,
            all levels are plotted.

        filter_parity : Union[None, int, str]
            Filter the levels by their parity. If None, all levels
            are plotted.

        plot : bool
            If True, the plot will be shown.
        
        """
        if not isinstance(filter_parity, (type(None), int, str)):
            msg = f"'filter_parity' must be of type: None, int, str. Got {type(j_list)}."
            raise TypeError(msg)

        if isinstance(filter_parity, str):
            valid_filter_parity = ["+", "-"]
            if filter_parity not in valid_filter_parity:
                msg = f"Valid parity filters are: {valid_filter_parity}."
                raise ValueError(msg)
            
            filter_parity = 1 if (filter_parity == "+") else -1

        if j_list is None:
            """
            If no angular momentum filter, then include all angular
            momenta in the data set.
            """
            angular_momenta = np.unique(self.levels[:, 1])/2
        else:
            if isinstance(j_list, (float, int)):
                angular_momenta = [j_list]
            
            elif isinstance(j_list, Iterable) and not isinstance(j_list, str):
                angular_momenta = j_list
            
            else:
                msg = f"'j_list' must be of type: None, Iterable, int, float. Got {type(j_list)}."
                raise TypeError(msg)
        
        bins = np.arange(E_min, E_max, bin_width)
        n_bins = len(bins)
        n_angular_momenta = len(angular_momenta)
        densities = np.zeros((n_bins, n_angular_momenta))

        for i in range(n_angular_momenta):
            """
            Calculate the nuclear level density for each angular
            momentum.

            NOTE: Each column is the density for a unique angular
            momentum. Consequently, rows are the energy axis.
            """
            bins_tmp, densities_tmp, counts_tmp = level_density(
                levels = self.levels,
                bin_width = bin_width,
                filter_spins = angular_momenta[i],
                filter_parity = filter_parity,
                E_min = E_min,
                E_max = E_max,
                return_counts = True,
                plot = False,
                save_plot = False
            )
            assert len(bins_tmp) <= n_bins, "There are more NLD bins than bins in the expected range!"
            for b1, b2 in zip(bins_tmp, bins):
                """
                Check that the returned bins match the expected bin
                range. Note that bins_tmp might be shorter than
                bins, but never longer.
                """
                assert b1 == b2, "NLD bins do not match the expected bins!"

            """
            Check that the total number of levels returned by the NLD
            function is actually the correct number of levels as counted
            with the following stright-forward counter:
            """
            E_tmp = self.levels[self.levels[:, 1] == 2*angular_momenta[i]]  # Extract levels of correct angular momentum.
            if filter_parity is not None:
                E_tmp = E_tmp[E_tmp[:, 2] == filter_parity]
            mask_lower = (E_tmp[:, 0] - self.ground_state_energy) >= E_min  # Keep only levels larger or equal to E_min.
            mask_upper = (E_tmp[:, 0] - self.ground_state_energy) < E_max   # Keep only levels lower than E_max.
            mask_total = np.logical_and(mask_lower, mask_upper)             # Combine the two masks with and.
            n_levels_in_range = sum(mask_total) # The number of Trues is the number of levels within the energy range.
            n_counts_tmp = sum(counts_tmp)
            msg = f"The number of levels from the NLD in the range [{E_min, E_max})"
            msg += " does not match the expected value!"
            msg += f" {n_levels_in_range = }, {n_counts_tmp = }"
            success = n_levels_in_range == n_counts_tmp
            assert success, msg

            """
            If E_max exceeds the max energy in the data set,
            level_density will adjust E_max to the max energy value. In
            this function, the bin range should be the same for all
            angular momenta, meaning that we might have to add some
            extra zeros to the end of the densities array.
            """
            n_missing_values = n_bins - len(bins_tmp)
            densities[:, i] = np.concatenate((densities_tmp, np.zeros(n_missing_values)))

            if flags["debug"]:
                print("-----------------------------------")
                print("angular_momentum_distribution debug")
                if all(densities_tmp == 0):
                    print("No levels in this energy range!")
                print(f"{angular_momenta[i] = }")
                print(f"{filter_parity = }")
                print(f"{bin_width = }")
                print(f"{E_min = }")
                print(f"{E_max = }")
                print(f"densities extended with {n_missing_values} zeros")
                print("-----------------------------------")

        if filter_parity is None:
            exponent = r"$^{\pm}$"  # For exponent in yticklabels.
            parity_str = "+-"       # For filename in savefig.
        elif filter_parity == 1:
            exponent = r"$^{+}$"
            parity_str = "+"
        elif filter_parity == -1:
            exponent = r"$^{-}$"
            parity_str = "-"

        if plot:
            xticklabels = []
            for i in bins:
                """
                Loop for generating x tick labels.
                
                Loop over the bins of the first angular momentum entry.
                Note that the bins are equal for all the different
                angular momenta (all columns in 'bins' are equal). Round
                all energies to 1 decimal when that decimal is != 0. If
                that decimal is zero, round to 0 decimals.
                """
                if (tmp := int(i)) == i:
                    """
                    Eg.
                    >>> 1 == 1.0
                    True
                    
                    (round to 0 decimals)
                    """
                    xticklabels.append(tmp)
                else:
                    """
                    Round to 1 decimal.
                    """
                    xticklabels.append(round(i, 1))
            
            fig, ax = plt.subplots(figsize=(7, 6.4))
            sns.heatmap(
                data = densities.T[-1::-1],
                linewidth = 0.5,
                annot = True,
                cmap = 'gray',
                xticklabels = xticklabels,
                fmt = ".0f",
                ax = ax
            )

            ax.set_yticklabels(np.flip([f"{Fraction(i)}" + exponent for i in angular_momenta]), rotation=0)
            ax.set_xlabel(r"$E$ [MeV]")
            ax.set_ylabel(r"$j$ [$\hbar$]")
            if set_title:
                ax.set_title(f"{self.nucleus_latex}, {self.interaction}")
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel(r"NLD [MeV$^{-1}$]", rotation=90)

            if save_plot:
                fig.savefig(f"{self.nucleus}_j{parity_str}_distribution_heatmap.png", dpi=300)
        
        if plot:
            plt.show()

        return bins, densities

    def B_distribution(self,
        partial_or_total: str,
        multipole_type: str = "M1",
        filter_spins: Union[None, list] = None,
        filter_parity: Union[None, int] = None,
        filter_indices: Union[None, int, list] = None,
        plot: bool = True,
    ) -> np.ndarray:
        """
        Plot a histogram of the distribution of B values.

        Parameters
        ----------
        partial_or_total : str
            If total, then all partial B values will be summed per
            level. If partial, then the distribution of all partial B
            values will be generated.

        multipole_type : str
            Choose the multipolarity of the transitions. 'E1', 'M1',
            'E2'.
        
        filter_spins : Union[None, list]
            Filter the levels by their angular momentum. If None,
            all levels are included.

        filter_parity : Union[None, int]
            Filter the levels by their parity. If None, both parities
            are included.

        plot : bool
            If True, the plot will be shown.

        Returns
        -------
        total_B : np.ndarray
            The sum over every partial B value for each level.
        """
        total_time = time.perf_counter()

        is_loaded = False
        B_unique_string = f"{multipole_type}{filter_spins}{filter_parity}{filter_indices}{partial_or_total}"
        B_unique_id = hashlib.sha1((B_unique_string).encode()).hexdigest()
        B_fname = f"{self.npy_path}/{self.base_fname}_Bdist_{B_unique_id}_{self.unique_id}.npy"

        if os.path.isfile(B_fname) and self.load_and_save_to_file:
            total_B = np.load(file=B_fname, allow_pickle=True)
            msg = f"{self.nucleus} {multipole_type} B distribution data loaded from .npy!"
            print(msg)
            is_loaded = True

        else:
            transitions_dict = {
                "E1": self.transitions_BE1,
                "M1": self.transitions_BM1,
                "E2": self.transitions_BE2,
            }
            transitions = transitions_dict[multipole_type]

            if filter_spins is None:
                initial_j = np.unique(transitions[:, 0])
            else:
                initial_j = [2*j for j in filter_spins]
            
            if filter_parity is None:
                initial_pi = [-1, 1]
            else:
                initial_pi = [filter_parity]

            if filter_indices is None:
                initial_indices = np.unique(transitions[:, 2]).astype(int)
            elif isinstance(filter_indices, list):
                initial_indices = [int(i) for i in filter_indices]
            elif isinstance(filter_indices, int):
                initial_indices = [filter_indices]
            
            total_B = []    # The sum of every partial B value for each level.
            idxi_masks = []
            pii_masks = []
            ji_masks = []

            mask_time = time.perf_counter()
            for idxi in initial_indices:
                idxi_masks.append(transitions[:, 2] == idxi)

            for pii in initial_pi:
                pii_masks.append(transitions[:, 1] == pii)

            for ji in initial_j:
                ji_masks.append(transitions[:, 0] == ji)
            mask_time = time.perf_counter() - mask_time

            for pii in pii_masks:
                for idxi in idxi_masks:
                    for ji in ji_masks:
                        mask = np.logical_and(ji, np.logical_and(pii, idxi))
                        # total_B.append(np.sum(transitions[mask][:, 9]))   # 9 is B decay
                        total_B.append(transitions[mask][:, 9])   # 9 is B decay

            if partial_or_total == "total":
                """
                Sum partial B values to get total B values.
                """
                for i in range(len(total_B)):
                    total_B[i] = sum(total_B[i])
            
            elif partial_or_total == "partial":
                """
                Keep a 1D list of partial B values.
                """
                total_B = list(chain.from_iterable(total_B))

        total_B = np.asarray(total_B)
        
        if self.load_and_save_to_file and not is_loaded:
            np.save(file=B_fname, arr=total_B, allow_pickle=True)
        
        total_time = time.perf_counter() - total_time

        if flags["debug"]:
            if not is_loaded: print(f"B_distribution {mask_time = :.4f}")
            print(f"B_distribution {total_time = :.4f}")

        if plot:
            plt.hist(total_B, bins=100, color="black")
            plt.xlabel(r"$B(" + f"{multipole_type}" + r")$")
            plt.show()

        return total_B

    def _brink_axel_j_calculator(self,
        bin_width: Union[float, int],
        Ex_min: Union[float, int],
        Ex_max: Union[float, int],
        multipole_type: str,
        j_list: list,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the GSF as a function of Eg for all Ei and ji, as well
        as the GSF individually for all ji in j_list.
        """
        bins_all_j, gsf_all_j = self.gsf(
            bin_width = bin_width,
            Ex_min = Ex_min,
            Ex_max = Ex_max,
            multipole_type = multipole_type,
            return_n_transitions = False,
            plot = False,
        )
        n_j = len(j_list)
        n_gsf_all_j = len(gsf_all_j)
        gsf = np.zeros((n_gsf_all_j, n_j))
        bins = np.zeros((n_gsf_all_j, n_j))   # Eg rows, j cols.

        for i in range(n_j):
            """
            Calculate the GSF individually for each ji.
            """
            bins_one_j, gsf_one_j = self.gsf(
                bin_width = bin_width,
                Ex_min = Ex_min,
                Ex_max = Ex_max,
                multipole_type = multipole_type,
                filter_spins = [j_list[i]],
                return_n_transitions = False,
                plot = False,
            )
            gsf[:, i] = gsf_one_j
            bins[:, i] = bins_one_j

        return bins, gsf, bins_all_j, gsf_all_j

    def brink_axel_j(self,
        bin_width: Union[float, int] = 0.2,
        Ex_min: Union[float, int] = 5,
        Ex_max: Union[float, int] = 50,
        multipole_type: Union[list, str] = "M1",
        j_list: Union[list, None] = None,
        set_title: bool = True
    ):
        """
        Plot a comparison of the GSF averaged over all ji and GSFs for
        each individual ji.
        """
        if j_list is None:
            j_list_default = True
        else:
            j_list_default = False
            
        transitions_dict = {
            "E1": self.transitions_BE1,
            "M1": self.transitions_BM1,
            "E2": self.transitions_BE2,
        }

        if isinstance(multipole_type, str):
            multipole_type = [multipole_type]
        
        n_multipole_types = len(multipole_type)

        fig, ax = plt.subplots(
            nrows = n_multipole_types,
            ncols = 1,
            figsize = (6.4, 4.8*n_multipole_types)
        )
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        upper_ylim = -np.inf
        lower_ylim = np.inf
        for i in range(n_multipole_types):
            if j_list_default:
                """
                Choose all available angular momenta as default.
                """
                j_list = np.unique(transitions_dict[multipole_type[i]][:, 0])/2
                j_list.sort()   # Just in case.

            bins, gsf, bins_all_j, gsf_all_j = self._brink_axel_j_calculator(
                    bin_width = bin_width,
                    Ex_min = Ex_min,
                    Ex_max = Ex_max,
                    multipole_type = multipole_type[i],
                    j_list = j_list,
                )

            ax[i].plot(bins_all_j, gsf_all_j, color="black", label=r"All $j_i$")
            ax[i].plot(bins, gsf, color="black", alpha=0.2)
            ax[i].plot([0], [0], color="black", alpha=0.2, label=r"Single $j_i$")  # Dummy for legend.
            ax[i].set_yscale('log')
            ax[i].set_ylabel(r"GSF [MeV$^{-3}$]")
            ax[i].legend()
            if set_title:
                ax[i].set_title(
                    f"{self.nucleus_latex}, {self.interaction}, " + r"$" + f"{multipole_type[i]}" + r"$"
                )
            else:
                ax[i].set_title(
                    r"$" + f"{multipole_type[i]}" + r"$"
                )

            lower_ylim = min(ax[i].get_ylim()[0], lower_ylim)
            upper_ylim = max(ax[i].get_ylim()[1], upper_ylim)

        for i in range(n_multipole_types):
            """
            Set both lower ylim to the lowest of the two and both upper
            ylims to the largest of the two.
            """
            ax[i].set_ylim((lower_ylim, upper_ylim))
        
        ax[-1].set_xlabel(r"E$_{\gamma}$ [MeV]")
        fig.savefig(
            fname = f"{self.nucleus}_brink-axel_ji_{'_'.join(multipole_type)}.png",
            dpi = 300
        )
        plt.show()

    def primary_matrix(self,
        bin_width: Union[float, int] = 0.2,
        Ex_min: Union[float, int] = 0,
        Ex_max: Union[float, int] = 50,
        multipole_type: str = "M1",
        plot: bool = True,
    ):
        """
        Create a Ex Eg primary matrix like the ones which are calculated
        with the Oslo method.

        Parameters
        ----------
        bin_width : float, optional
            Width of the bins in MeV, by default 0.2
        
        Ex_min : float, optional
            Minimum excitation energy of initial level in a transition,
            by default 0 MeV.

        Ex_max : float, optional
            Maximum excitation energy of initial level in a transition,
            by default 50 MeV.
        
        multipole_type : str
            Multipolarity of the transitions to be considered, by
            default "M1".

        plot : bool, optional
            Whether to plot the matrix, by default True.

        Returns
        -------
        Ex_range : np.ndarray
            The range of the initial level energies.

        Eg_range : np.ndarray
            The range of the gamma energies.

        B_matrix : np.ndarray
            The primary matrix.
        """
        transitions_dict = {
            "E1": self.transitions_BE1,
            "M1": self.transitions_BM1,
            "E2": self.transitions_BE2,
        }

        n_bins = int(np.ceil((Ex_max - Ex_min)/bin_width))
        B_matrix = np.zeros((n_bins, n_bins), dtype=float)    # rows: Ei, cols: Eg
        Ex_range = np.linspace(Ex_min, Ex_max, n_bins)
        Eg_range = np.linspace(Ex_min, Ex_max, n_bins)
        
        transitions = transitions_dict[multipole_type]
        Ex_masks = []
        Eg_masks = []
        for i in range(n_bins-1):
            """
            Generate the masks once. It is not necessary to generate the
            Eg masks over and over again while iterating over the Ex
            masks.
            """
            Ex_masks.append(np.logical_and(
                transitions[:, 3] >= Ex_range[i],
                transitions[:, 3] < Ex_range[i+1],
            ))
        for i in range(n_bins-1):
            """
            Using separate loops allows using different bin widths for
            Ex and Eg, however, that has not yet been implemented.
            """
            Eg_masks.append(np.logical_and(
                transitions[:, 8] >= Eg_range[i],
                transitions[:, 8] < Eg_range[i+1],
            ))
        for i in range(n_bins-1):
            """
            Loop over each Ex Eg mask pair and create a combined mask.
            Take the mean of the B values for each combined mask.
            """
            for j in range(i+1):
                mask = np.logical_and(
                    Ex_masks[i],
                    Eg_masks[j],
                )
                B_slice = transitions[mask][:, 9]
                if B_slice.size == 0: continue  # Skip if there are no B values in the mask.
                
                B_matrix[i, j] = np.mean(B_slice)

        if plot:
            fig, ax = plt.subplots()
            im = ax.pcolormesh(Ex_range, Eg_range, B_matrix, cmap="jet", norm="log")
            ax.set_title(r"$\langle B($" + f"{multipole_type}" + r"$) \rangle$")
            ax.set_xlabel(r"$E_{\gamma}$")
            ax.set_ylabel(r"$E_{x}$")
            fig.colorbar(im)
            plt.show()

        return Ex_range, Eg_range, B_matrix

    def _mixing_pairs(self,
        B_left: str = "M1",
        B_right: str = "E2",
    ):
        """
        NOTE: Currently hard-coded for BM1_BE2!
        """
        if B_left != "M1":#not in ["M1", "E2"]:
            # msg = f"'B_left' must be 'M1' or 'E2', got: {B_left}"
            # raise ValueError(msg)
            raise ValueError
        
        if B_right != "E2":# not in ["M1", "E2"]:
            # msg = f"'B_right' must be 'M1' or 'E2', got: {B_right}"
            # raise ValueError(msg)
            raise ValueError
        
        transitions_dict = {
            "M1": self.transitions_BM1,
            "E2": self.transitions_BE2,
            "E1": self.transitions_BE1
        }
        transitions_left = transitions_dict[B_left]
        transitions_right = transitions_dict[B_right]
        
        mixing_pairs_fname = f"{self.npy_path}/{self.base_fname}_mixing_pairs_B{B_left}_B{B_right}_{self.unique_id}.npy"

        if os.path.isfile(mixing_pairs_fname) and self.load_and_save_to_file and (self.load_and_save_to_file != "overwrite"):
            print(f"Mixing pairs loaded from .npy!")
            mixing_pairs = np.load(file=mixing_pairs_fname, allow_pickle=True)
            return mixing_pairs        
        
        @numba.njit
        def calculate_mixing_pairs(
            possible_j: list[int],
            possible_indices: list[int],
            transitions_BM1: np.ndarray,
            transitions_BE2: np.ndarray,
        ):
            mixing_pairs = []

            for ji in possible_j:
                ji_slice_BM1 = transitions_BM1[transitions_BM1[:, 0] == ji]
                ji_slice_BE2 = transitions_BE2[transitions_BE2[:, 0] == ji]

                if (not ji_slice_BM1.size) or (not ji_slice_BE2.size): return

                for jf in possible_j:
                    # if not j_allowed(ji=ji, jf=jf): continue
                    jf_slice_BM1 = ji_slice_BM1[ji_slice_BM1[:, 4] == jf]
                    jf_slice_BE2 = ji_slice_BE2[ji_slice_BE2[:, 4] == jf]

                    if (not jf_slice_BM1.size) or (not jf_slice_BE2.size): continue

                    for pi in [-1, 1]:
                        """
                        [2*spin_initial, parity_initial, idx_initial, Ex_initial, 2*spin_final,
                        parity_final, idx_final, Ex_final, E_gamma, B(.., i->f), B(.., f<-i)]
                        """
                        pi_i_slice_BM1 = jf_slice_BM1[jf_slice_BM1[:, 1] == pi]
                        pi_i_slice_BE2 = jf_slice_BE2[jf_slice_BE2[:, 1] == pi]

                        if (not pi_i_slice_BM1.size) or (not pi_i_slice_BE2.size): continue
                        
                        pi_f_slice_BM1 = pi_i_slice_BM1[pi_i_slice_BM1[:, 5] == pi]
                        pi_f_slice_BE2 = pi_i_slice_BE2[pi_i_slice_BE2[:, 5] == pi]

                        if (not pi_f_slice_BM1.size) or (not pi_f_slice_BE2.size): continue

                        for idx_i in possible_indices:
                            idx_i_slice_BM1 = pi_f_slice_BM1[pi_f_slice_BM1[:, 2] == idx_i]
                            idx_i_slice_BE2 = pi_f_slice_BE2[pi_f_slice_BE2[:, 2] == idx_i]

                            if (not idx_i_slice_BM1.size) or (not idx_i_slice_BE2.size): continue

                            for idx_f in possible_indices:
                                idx_f_slice_BM1 = idx_i_slice_BM1[idx_i_slice_BM1[:, 2] == idx_f]
                                idx_f_slice_BE2 = idx_i_slice_BE2[idx_i_slice_BE2[:, 2] == idx_f]

                                if (not idx_f_slice_BM1.size) or (not idx_f_slice_BE2.size): continue

                                for transition_BM1 in idx_f_slice_BM1:
                                    for transition_BE2 in idx_f_slice_BE2:
                                        if (transition_BM1[3] == transition_BE2[3]) and (transition_BM1[7] == transition_BE2[7]):
                                            mixing_pairs.append([transition_BM1, transition_BE2])
                
            return mixing_pairs

        mixing_pairs_time = time.perf_counter()
        possible_j = np.unique(transitions_left[:, 0])
        possible_indices = np.unique(transitions_left[:, 2])
        assert np.all(possible_indices == np.unique(transitions_left[:, 6]))    # Check that indices of the initial and final levels are the same range.
        
        mixing_pairs = calculate_mixing_pairs(
            transitions_BM1 = transitions_left,
            transitions_BE2 = transitions_right,
            possible_j = possible_j,
            possible_indices = possible_indices,
        )
        mixing_pairs.sort(key=lambda tup: tup[0][8])    # Sort wrt. gamma energy.
        mixing_pairs = np.array(mixing_pairs)
        np.save(file=mixing_pairs_fname, arr=mixing_pairs, allow_pickle=True)
        mixing_pairs_time = time.perf_counter() - mixing_pairs_time
        print(f"{mixing_pairs_time = :.3f} s")
        return mixing_pairs

    def mixing_ratio(self,
        bin_width: float = 0.2,
        plot: bool = True,
        save_plot: bool = False,
    ):
        """
        Calculate the ratio of T(E2)/(T(E2) + T(M1)), aka. how large the
        E2 contribution is. Currently hard-coded for this specific
        ratio.

        Parameters
        ----------
        bin_width : float
            The ratios are sorted by gamma energy and averaged over
            the ratio values in a gamma energy bin of bin_with.
        """
        E_min = self.mixing_pairs_BM1_BE2[0, 0, 8]  # BM1 and BE2 has to be at the exact same gamma energies so it doesnt matter which one we take E_min and E_max from.
        E_max = self.mixing_pairs_BM1_BE2[-1, 0, 8]

        bins = np.arange(E_min, E_max + bin_width, bin_width)
        n_bins = len(bins)
        ratios = np.zeros(n_bins - 1)

        for i in range(n_bins - 1):
            mask_1 = self.mixing_pairs_BM1_BE2[:, 0, 8] >= bins[i]
            mask_2 = self.mixing_pairs_BM1_BE2[:, 0, 8] < bins[i + 1]
            mask_3 = np.logical_and(mask_1, mask_2)

            # M1_mean = self.mixing_pairs_BM1_BE2[:, 0, 9][mask_3].mean()
            # E2_mean = self.mixing_pairs_BM1_BE2[:, 1, 9][mask_3].mean()

            M1_mean = (1.76e13*(self.mixing_pairs_BM1_BE2[:, 0, 8][mask_3]**3)*self.mixing_pairs_BM1_BE2[:, 0, 9][mask_3]).mean()
            E2_mean = (1.22e09*(self.mixing_pairs_BM1_BE2[:, 1, 8][mask_3]**5)*self.mixing_pairs_BM1_BE2[:, 1, 9][mask_3]).mean()

            ratios[i] = E2_mean/(E2_mean + M1_mean)
        
        if plot:
            fig, ax = plt.subplots()
            ax.step(bins[:-1], ratios, color="black")
            # ax.legend()
            ax.grid()
            ax.set_ylabel(r"TE2/(TE2 + TM1)")
            ax.set_xlabel(r"$E_{\gamma}$ [MeV]")
            if save_plot:
                fname = f"mixing_ratio_E2_M1.png"
                print(f"Mixing ratio plot saved as '{fname}'")
                fig.savefig(fname=fname, dpi=300)
            plt.show()

        return bins[:-1], ratios

    @property
    def help(self):
        """
        Generate a list of instance attributes without magic and private
        methods.

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

    @property
    def parameters(self) -> dict:
        """
        Get the KSHELL parameters from the shell file.

        Returns
        -------
        : dict
            A dictionary of KSHELL parameters.
        """
        path = self.path
        if os.path.isfile(path):
            path = path.rsplit("/", 1)[0]
        return get_parameters(path)

    @property
    def nucleus_latex(self):
        m = re.search(r"\d+$", self.nucleus)
        A = m.group()
        X = self.nucleus[:m.span()[0]]
        return r"$^{" + f"{A}" + r"}$" + f"{X}"

    def check_data(self):
        """
        Check that the 'levels' data is sorted in order of increasing
        energy.
        """
        msg = "'levels' should be sorted in order of increasing energy,"
        msg += " but it isnt! Check the data!"
        energies = self.levels[:, 0]
        success = np.all(energies[:-1] <= energies[1:])
        assert success, msg

        """
        Check that the ground state energy is correct.
        """
        msg = "'ground_state_energy' is not the lowest energy in 'levels'"
        msg += " as it should be! Check the data!"
        success = self.ground_state_energy == np.min(self.levels[:, 0])
        assert success, msg

        """
        Insert more checks under...
        """

def loadtxt(
    path: str,
    load_and_save_to_file: Union[bool, str] = True,
    old_or_new = "new"
    ) -> ReadKshellOutput:
    """
    Wrapper for using ReadKshellOutput class as a function.

    Parameters
    ----------
    path : str
        Path to summary file or path to directory with summary / log
        files.

    load_and_save_to_file : Union[bool, str]
        Toggle saving data as `.npy` files on / off. If 'overwrite',
        saved `.npy` files are overwritten.

    old_or_new : str
        Choose between old and new summary file syntax. All summary
        files generated pre 2021-11-24 use old style.
        New:
        J_i  pi_i idx_i Ex_i    J_f  pi_f idx_f Ex_f      dE         B(E2)->         B(E2)->[wu]     B(E2)<-         B(E2)<-[wu]
        5    +    1     0.036   6    +    1     0.000     0.036     70.43477980      6.43689168     59.59865983      5.44660066
        Old:
        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
        2+(11) 18.393 2+(10) 17.791 0.602 0.1(  0.0) 0.1( 0.0)

    Returns
    -------
    res : ReadKshellOutput
        Instances with data from `KSHELL` data file as attributes.
    """
    loadtxt_time = time.perf_counter()  # Debug.

    if old_or_new not in (old_or_new_allowed := ["old", "new", "jem"]):
        msg = f"'old_or_new' argument must be in {old_or_new_allowed}!"
        msg += f" Got '{old_or_new}'."
        raise ValueError(msg)

    res = ReadKshellOutput(path, load_and_save_to_file, old_or_new)

    loadtxt_time = time.perf_counter() - loadtxt_time
    if flags["debug"]:
        print(f"{loadtxt_time = } s")

    return res

def _get_timing_data(path: str):
    """
    Get timing data from KSHELL log files.

    Parameters
    ----------
    path : str
        Path to log file.

    Examples
    --------
    Last 10 lines of log_Ar30_usda_m0p.txt:
    ```
          total      20.899         2    10.44928   1.0000
    pre-process       0.029         1     0.02866   0.0014
        operate       3.202      1007     0.00318   0.1532
     re-orthog.      11.354       707     0.01606   0.5433
  thick-restart       0.214        12     0.01781   0.0102
   diag tri-mat       3.880       707     0.00549   0.1857
           misc       2.220                         0.1062

           tmp        0.002       101     0.00002   0.0001
    ```
    """
    if "log" not in path:
        msg = f"Unknown log file name! Got '{path}'"
        raise KshellDataStructureError(msg)

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    res = os.popen(f'tail -n 20 {path}').read()    # Get the final 10 lines.
    res = res.split("\n")
    total = None
    
    if "_tr_" not in path:
        """
        KSHELL log.
        """
        for elem in res:
            tmp = elem.split()
            try:
                if tmp[0] == "total":
                    total = float(tmp[1])
                    break
            except IndexError:
                continue
        
    elif "_tr_" in path:
        """
        Transit log.
        """
        for elem in res:
            tmp = elem.split()
            try:
                if tmp[0] == "total":
                    total = float(tmp[3])
                    break
            except IndexError:
                continue

    if total is None:
        msg = f"Not able to extract timing data from '{path}'!"
        raise KshellDataStructureError(msg)
    
    return total

def _get_memory_usage(path: str) -> Union[float, None]:
    """
    Get memory usage from KSHELL log files.

    Parameters
    ----------
    path : str
        Path to a single log file.

    Returns
    -------
    total : float, None
        Memory usage in GB or None if memory usage could not be read.
    """
    total = None
    
    if "tr" not in path:
        """
        KSHELL log.
        """
        with open(path, "r") as infile:
            for line in infile:
                if line.startswith("Total Memory for Lanczos vectors:"):
                    try:
                        total = float(line.split()[-2])
                    except ValueError:
                        msg = f"Error reading memory usage from '{path}'."
                        msg += f" Got '{line.split()[-2]}'."
                        raise KshellDataStructureError(msg)
                    break
        
    elif "tr" in path:
        """
        Transit log. NOTE: Not yet implemented.
        """
        return 0

    if total is None:
        msg = f"Not able to extract memory data from '{path.split('/')[-1]}'!"
        raise KshellDataStructureError(msg)
    
    return total

def _sortkey(filename):
    """
    Key for sorting filenames based on angular momentum and parity.
    Example filename: 'log_Sc44_GCLSTsdpfsdgix5pn_j0n.txt'
    (angular momentum  = 0). 
    """
    tmp = filename.split("_")[-1]
    tmp = tmp.split(".")[0]
    # parity = tmp[-1]
    spin = int(tmp[1:-1])
    # return f"{spin:03d}{parity}"    # Examples: 000p, 000n, 016p, 016n
    return spin

def _get_data_general(
    path: str,
    func: Callable,
    plot: bool
    ):
    """
    General input handling for timing data and memory data.

    Parameters
    ----------
    path : str
        Path to a single log file or path to a directory of log files.

    func : Callable
        _get_timing_data or _get_memory_usage.
    """
    total_negative = []
    total_positive = []
    filenames_negative = []
    filenames_positive = []
    if os.path.isfile(path):
        return func(path)
    
    elif os.path.isdir(path):
        for elem in os.listdir(path):
            """
            Select only log files in path.
            """
            tmp = elem.split("_")
            try:
                if ((tmp[0] == "log") or (tmp[1] == "log")) and elem.endswith(".txt"):
                    tmp = tmp[-1].split(".")
                    parity = tmp[0][-1]
                    if parity == "n":
                        filenames_negative.append(elem)
                    elif parity == "p":
                        filenames_positive.append(elem)
            except IndexError:
                continue
        
        filenames_negative.sort(key=_sortkey)
        filenames_positive.sort(key=_sortkey)

        for elem in filenames_negative:
            total_negative.append(func(f"{path}/{elem}"))
        for elem in filenames_positive:
            total_positive.append(func(f"{path}/{elem}"))
        
        if plot:
            xticks_negative = ["sum"] + [str(Fraction(_sortkey(i)/2)) for i in filenames_negative]
            xticks_positive = ["sum"] + [str(Fraction(_sortkey(i)/2)) for i in filenames_positive]
            sum_total_negative = sum(total_negative)
            sum_total_positive = sum(total_positive)
            
            fig0, ax0 = plt.subplots(ncols=1, nrows=2)
            fig1, ax1 = plt.subplots(ncols=1, nrows=2)

            bars = ax0[0].bar(
                xticks_negative,
                [sum_total_negative/60/60] + [i/60/60 for i in total_negative],
                color = "black",
            )
            ax0[0].set_title("negative")
            for rect in bars:
                height = rect.get_height()
                ax0[0].text(
                    x = rect.get_x() + rect.get_width() / 2.0,
                    y = height,
                    s = f'{height:.3f}',
                    ha = 'center',
                    va = 'bottom'
                )
            
            bars = ax1[0].bar(
                xticks_negative,
                [sum_total_negative/sum_total_negative] + [i/sum_total_negative for i in total_negative],
                color = "black",
            )
            ax1[0].set_title("negative")
            for rect in bars:
                height = rect.get_height()
                ax1[0].text(
                    x = rect.get_x() + rect.get_width() / 2.0,
                    y = height,
                    s = f'{height:.3f}',
                    ha = 'center',
                    va = 'bottom'
                )
            
            bars = ax0[1].bar(
                xticks_positive,
                [sum_total_positive/60/60] + [i/60/60 for i in total_positive],
                color = "black",
            )
            ax0[1].set_title("positive")
            for rect in bars:
                height = rect.get_height()
                ax0[1].text(
                    x = rect.get_x() + rect.get_width() / 2.0,
                    y = height,
                    s = f'{height:.3f}',
                    ha = 'center',
                    va = 'bottom'
                )

            bars = ax1[1].bar(
                xticks_positive,
                [sum_total_positive/sum_total_positive] + [i/sum_total_positive for i in total_positive],
                color = "black",
            )
            ax1[1].set_title("positive")
            for rect in bars:
                height = rect.get_height()
                ax1[1].text(
                    x = rect.get_x() + rect.get_width() / 2.0,
                    y = height,
                    s = f'{height:.3f}',
                    ha = 'center',
                    va = 'bottom'
                )

            fig0.text(x=0.02, y=0.5, s="Time [h]", rotation="vertical")
            fig0.text(x=0.5, y=0.02, s="Angular momentum")
            fig1.text(x=0.02, y=0.5, s="Norm. time", rotation="vertical")
            fig1.text(x=0.5, y=0.02, s="Angular momentum")
            plt.show()

        return sum(total_negative) + sum(total_positive)

    else:
        msg = f"'{path}' is neither a file nor a directory!"
        raise FileNotFoundError(msg)

def get_timing_data(path: str, plot: bool = False) -> float:
    """
    Wrapper for _get_timing_data. Input a single log filename and get
    the timing data. Input a path to a directory several log files and
    get the summed timing data. In units of seconds.

    Parameters
    ----------
    path : str
        Path to a single log file or path to a directory of log files.

    Returns
    -------
    : float
        The summed times for all input log files.
    """
    return _get_data_general(path, _get_timing_data, plot)

def get_memory_usage(path: str) -> float:
    """
    Wrapper for _get_memory_usage. Input a single log filename and get
    the memory data. Input a path to a directory several log files and
    get the summed memory data. In units of GB.

    Parameters
    ----------
    path : str
        Path to a single log file or path to a directory of log files.

    Returns
    -------
    : float
        The summed memory usage for all input log files.
    """
    return _get_data_general(path, _get_memory_usage, False)

def get_parameters(path: str, verbose: bool = True) -> dict:
    """
    Extract the parameters which are fed to KSHELL throught the shell
    script.

    Parameters
    ----------
    path : str
        Path to a KSHELL work directory.

    Returns
    -------
    res : dict
        A dictionary where the keys are the parameter names and the
        values are the corresponding values.
    """
    res = {}
    shell_filename = None
    if os.path.isdir(path):
        for elem in os.listdir(path):
            if elem.endswith(".sh"):
                shell_filename = f"{path}/{elem}"
                break
    else:
        print("Directly specifying path to .sh file not yet implemented!")

    if shell_filename is None:
        if verbose:
            msg = f"No .sh file found in path '{path}'!"
            print(msg)

        return res
    
    with open(shell_filename, "r") as infile:
        for line in infile:
            if line.startswith(r"&input"):
                break
        
        for line in infile:
            if line.startswith(r"&end"):
                """
                End of parameters.
                """
                break
            
            tmp = line.split("=")
            key = tmp[0].strip()
            value = tmp[1].strip()

            try:
                value = ast.literal_eval(value)
            except ValueError:
                """
                Cant convert strings. Keep them as strings.
                """
                pass
            except SyntaxError:
                """
                Cant convert Fortran booleans (.true., .false.). Keep
                them as strings.
                """
                pass
            
            res[key] = value

    return res

def inspect_log(
    path: str
):
    """
    Load log file(s) for inspection.

    Parameters
    ----------
    path : str
        Path to directory of log files or directly to a log file.
    """
    allowed_multipolarities_list = []
    if os.path.isdir(path):
        """
        List all log files in 'path' and let the user decide which one
        to inspect.
        """
        content = [i for i in os.listdir(path) if "log_" in i]
        content.sort()
        content = np.array(content)
        
        if not content.size:
            msg = f"No log files found in '{path}'!"
            raise FileNotFoundError(msg)

        for i, elem in enumerate(content):
            """
            Make log file names more readable and calculate allowed
            multipolarities (E1, M1, E2) based on angular momentum and
            parity (if transition log file).
            """
            if "_tr_" in elem:
                """
                Transition log file.
                """
                elem = elem.rstrip(".txt")
                elem = elem.split("_")[-2:]
                elem[0] = elem[0].lstrip("j")
                elem[1] = elem[1].lstrip("j")
                
                delta_parity = elem[0][-1] != elem[1][-1]   # Change in parity.

                j_i = int(elem[0][:-1])/2   # Initial total angular momentum.
                j_f = int(elem[1][:-1])/2   # Final.

                if delta_parity:
                    j_max = 1   # Because KSHELL doesnt support M2.
                else:
                    j_max = 2   # Because KSHELL supports max. E2.
                
                allowed_j = np.arange(max(np.abs(j_i - j_f), 1), min(j_i + j_f, j_max) + 1, 1, dtype=int)   # Possible j couplings.

                tmp_multipolarity = []
                if delta_parity:
                    """
                    If there is a change in parity: Even numbered
                    magnetic and odd numbered electric.
                    """
                    for j in allowed_j:
                        if j%2 == 0:
                            tmp_multipolarity.append(f"M{j}")
                        else:
                            tmp_multipolarity.append(f"E{j}")

                else:
                    """
                    If there is not a change in parity: Even numbered
                    electric and odd numbered magnetic.
                    """
                    for j in allowed_j:
                        if j%2 == 0:
                            tmp_multipolarity.append(f"E{j}")
                        else:
                            tmp_multipolarity.append(f"M{j}")

                allowed_multipolarities_list.append(tmp_multipolarity)
                
                elem = " <-> ".join(elem)
            else:
                elem = elem.rstrip(".txt")
                elem = elem.split("_")[-1]
                elem = elem.lstrip("j")
                allowed_multipolarities_list.append([])

            print(f"{i}: {elem}")

        allowed_multipolarities_list = np.array(allowed_multipolarities_list, dtype=object) # Numpy arrays support indexing with lists of indices.

        while True:
            """
            Prompt user for a single index or indices separated by
            comma. The latter is converted to a list and used as indices
            to the correct path and accompanying allowed multipolarity.
            """
            choice = input("Choose a log file for inspection: ")
            try:
                choice = ast.literal_eval(choice)
            except (SyntaxError, ValueError):
                print("Input must be an index or comma separated indices!")
                continue
            try:
                if isinstance(choice, int):
                    path_log = [content[choice]]
                    allowed_multipolarities = [allowed_multipolarities_list[choice]]
                elif isinstance(choice, Iterable):
                    choice = list(choice)
                    path_log = content[choice]
                    allowed_multipolarities = allowed_multipolarities_list[choice]
                else:
                    continue
                
                break

            except IndexError:
                print(f"Input indices cannot be larger than {len(content) - 1}.")
                continue

        print(f"{path_log} chosen")
        path_log = [f"{path}/{log}" for log in path_log]    # Prepend entire path (relative or absolute depending on user input).

    elif os.path.isfile(path):
        """
        If path to a single log file is given.
        """
        path_log = [path]
        elem = path_log[0].split("/")[-1]
        elem = elem.rstrip(".txt")
        elem = elem.split("_")[-2:]
        elem[0] = elem[0].lstrip("j")
        elem[1] = elem[1].lstrip("j")
        
        delta_parity = elem[0][-1] != elem[1][-1]

        j_i = int(elem[0][:-1])/2   # Initial total angular momentum.
        j_f = int(elem[1][:-1])/2   # Final.

        if delta_parity:
            j_max = 1   # Because KSHELL doesnt support M2.
        else:
            j_max = 2   # Because KSHELL supports max. E2.
        
        allowed_j = np.arange(max(np.abs(j_i - j_f), 1), min(j_i + j_f, j_max) + 1, 1, dtype=int)   # Possible j couplings.

        tmp_multipolarity = []
        if delta_parity:
            """
            If there is a change in parity: Even numbered
            magnetic and odd numbered electric.
            """
            for j in allowed_j:
                if j%2 == 0:
                    tmp_multipolarity.append(f"M{j}")
                else:
                    tmp_multipolarity.append(f"E{j}")

        else:
            """
            If there is not a change in parity: Even numbered
            electric and odd numbered magnetic.
            """
            for j in allowed_j:
                if j%2 == 0:
                    tmp_multipolarity.append(f"E{j}")
                else:
                    tmp_multipolarity.append(f"M{j}")

        allowed_multipolarities = [tmp_multipolarity]
    
    else:
        msg = "Input 'path' must be a file or directory!"
        raise FileNotFoundError(msg)

    for i, log in enumerate(path_log):
        if "_tr_" not in log:
            msg = "Inspection implemented only for transition log files (for now)."
            msg += f"\n{log = }"
            raise NotImplementedError(msg)
        
        for multipolarity in allowed_multipolarities[i]:
            j_f = []
            idx_f = []
            E_f = []
            j_i = []
            idx_i = []
            E_i = []
            E_x = []
            mred = []
            B_decay = []
            B_excite = []
            mom = []

            with open(log, "r") as infile:
                for line in infile:
                    """
                    Find where 'multipolarity' starts in the log file.
                    """
                    if f"{multipolarity} transition" not in line:
                        continue
                    else:
                        break
                
                infile.readline()   # Skip a line.

                for line in infile:
                    try:
                        tmp_j_f, tmp_idx_f, tmp_E_f, tmp_j_i, tmp_idx_i, tmp_E_i, \
                        tmp_E_x, tmp_mred, tmp_B_decay, tmp_B_excite, tmp_mom = \
                            [ast.literal_eval(i) for i in line.split()]

                        j_f.append(tmp_j_f)
                        idx_f.append(tmp_idx_f)
                        E_f.append(tmp_E_f)
                        j_i.append(tmp_j_i)
                        idx_i.append(tmp_idx_i)
                        E_i.append(tmp_E_i)
                        E_x.append(tmp_E_x)
                        mred.append(tmp_mred)
                        B_decay.append(tmp_B_decay)
                        B_excite.append(tmp_B_excite)
                        mom.append(tmp_mom)
                    except ValueError:
                        break
            
            j_f = np.array(j_f)
            idx_f = np.array(idx_f)
            E_f = np.array(E_f)
            j_i = np.array(j_i)
            idx_i = np.array(idx_i)
            E_i = np.array(E_i)
            E_x = np.array(E_x)
            mred = np.array(mred)
            B_decay = np.array(B_decay)
            B_excite = np.array(B_excite)
            mom = np.array(mom)

            print(f"{multipolarity} inspection summary of {log.split('/')[-1]}")
            print(f"{np.mean(B_decay) = }")
            print(f"{np.min(B_decay) = }")
            print(f"{np.max(B_decay) = }")
            print(f"{sum(B_decay) = }")
            print(f"{sum(B_decay == 0) = }")
            print(f"{sum(B_decay != 0) = }")
            print(f"{len(B_decay) = }")
            print()
