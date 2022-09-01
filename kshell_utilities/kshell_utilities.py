import os, sys, multiprocessing, hashlib, ast, time, re
from fractions import Fraction
from typing import Union, Callable
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .kshell_exceptions import KshellDataStructureError
from .parameters import atomic_numbers, flags
from .general_utilities import (
    level_plot, level_density, gamma_strength_function_average, porter_thomas
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

        for elem in os.listdir(directory):
            """
            Loop over all elements in the directory and find the shell
            script and save_input file.
            """
            try:
                if elem.endswith(".sh"):
                    with open(f"{directory}/{elem}", "r") as infile:
                        shell_file_content += infile.read()
                # elif elem.endswith(".input"):
                elif "save_input_ui.txt" in elem:
                    with open(f"{directory}/{elem}", "r") as infile:
                        save_input_content += infile.read()
            except UnicodeDecodeError:
                msg = f"Skipping {elem} for tmp file unique identifier due to UnicodeDecodeError."
                msg += " Are you sure this file is supposed to be in this directory?"
                print(msg)
                continue
    else:
        print(msg)

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

        self.path = path
        self.load_and_save_to_file = load_and_save_to_file
        self.old_or_new = old_or_new
        # Some attributes might not be altered, depending on the input file.
        self.fname_summary = None
        self.fname_ptn = None
        self.nucleus = None
        self.interaction = None
        self.proton_partition = None
        self.neutron_partition = None
        self.levels = None
        self.transitions_BM1 = None
        self.transitions_BE2 = None
        self.transitions_BE1 = None
        self.truncation = None
        self.npy_path = "tmp"   # Directory for storing .npy files.
        self.base_fname = self.path.split("/")[-1][:-4] # Base filename for .npy files.
        self.unique_id = _generate_unique_identifier(self.path) # Unique identifier for .npy files.
        # Debug.
        self.negative_spin_counts = np.array([0, 0, 0, 0])  # The number of skipped -1 spin states for [levels, BM1, BE2, BE1].

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
                raise KshellDataStructureError(msg)

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
            [self.fname_summary, "Energy", "replace_this_entry_with_loader", 0],
            [self.fname_summary, "B(E1)", "replace_this_entry_with_loader", 1],
            [self.fname_summary, "B(M1)", "replace_this_entry_with_loader", 2],
            [self.fname_summary, "B(E2)", "replace_this_entry_with_loader", 3],
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
        filter_spins: Union[None, list] = None
        ):
        """
        Wrapper method to include level plot as an attribute to this
        class. Generate a level plot for a single isotope. Spin on the x
        axis, energy on the y axis.

        Parameters
        ----------
        include_n_levels : int
            The maximum amount of states to plot for each spin. Default
            set to a large number to indicate ≈ no limit.

        filter_spins : Union[None, list]
            Which spins to include in the plot. If `None`, all spins are
            plotted. Defaults to `None`
        """
        level_plot(
            levels = self.levels,
            include_n_levels = include_n_levels,
            filter_spins = filter_spins
        )

    def level_density_plot(self,
            bin_width: Union[int, float] = 0.2,
            include_n_levels: Union[None, int] = None,
            filter_spins: Union[None, int, list] = None,
            filter_parity: Union[None, str, int] = None,
            E_min: Union[None, float, int] = None,
            E_max: Union[None, float, int] = None,
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
        E_min: Union[None, float, int] = None,
        E_max: Union[None, float, int] = None,
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
        multipole_type: str = "M1",
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

        colors = ["blue", "royalblue", "lightsteelblue"]
        Ei_range = np.linspace(Ei_range_min, Ei_range_max, 4)
        
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
                multipole_type = multipole_type,
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
                multipole_type = multipole_type,
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
        if set_title:
            axd["upper"].set_title(
                f"{self.nucleus_latex}, {self.interaction}, " + r"$" + f"{multipole_type}" + r"$"
            )
        fig.savefig(fname=f"{self.nucleus}_porter_thomas_Ei_{multipole_type}.png", dpi=300)
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
        # transitions_dict = {
        #     "E1": self.transitions_BE1,
        #     "M1": self.transitions_BM1,
        #     "E2": self.transitions_BE2,
        # }
        # if j_lists is None:
        #     j_lists = list(np.unique(transitions_dict[multipole_type][:, 0]))
        
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
        
        # colors = ["blue", "royalblue", "lightsteelblue"]
        
        # fig, axd = plt.subplot_mosaic(
        #     [['upper'], ['lower']],
        #     gridspec_kw = dict(height_ratios=[1, 0.5]),
        #     figsize = (6.4, 8),
        #     constrained_layout = True,
        #     sharex = True
        # )
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

        # self._porter_thomas_j_plot_plotter(
        #     binss = binss,
        #     countss = countss,
        #     chi2s = chi2s,
        #     j_lists = j_lists,
        #     colors = colors,
        #     set_title = set_title,
        #     multipole_type = multipole_type
        # )

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
        transitions_dict = {
            "E1": self.transitions_BE1,
            "M1": self.transitions_BM1,
            "E2": self.transitions_BE2,
        }
        if j_lists is None:
            """
            Default j_lists values.
            """
            j_lists = []
            for elem in np.unique(transitions_dict[multipole_type][:, 0]):
                j_lists.append([int(elem/2)])

            j_lists = j_lists[:3]   # _porter_thomas_j_plot_calculator supports max. 3 lists of j values.

        colors = ["blue", "royalblue", "lightsteelblue"]
        if isinstance(multipole_type, str):
            multipole_type = [multipole_type]
        
        elif isinstance(multipole_type, list):
            pass
        
        else:
            msg = f"multipole_type must be str or list. Got {type(multipole_type)}."
            raise TypeError(msg)

        if len(multipole_type) == 1:
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

        elif len(multipole_type) == 2:
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
        else:
            msg = "Only 1 and 2 multipole types may be given at the same time!"
            msg += f" Got {len(multipole_type)}."
            raise ValueError(msg)

        fig.savefig(fname=f"{self.nucleus}_porter_thomas_j_{multipole_type}.png", dpi=300)
        plt.show()

    def angular_momentum_distribution_plot(self,
        bin_width: float = 0.2,
        E_min: float = 5,
        E_max: float = 10,
        filter_spins: Union[None, int, float, list, tuple, np.ndarray] = None,
        filter_parity: Union[None, int, str] = None,
        plot: bool = True,
        single_spin_plot: Union[None, list, tuple, np.ndarray, int, float] = None,
        save_plot: bool = False,
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

        filter_spins : Union[None, int, float, list, tuple, np.ndarray]
            Filter the levels by their angular momentum. If None,
            all levels are plotted.

        filter_parity : Union[None, int, str]
            Filter the levels by their parity. If None, all levels
            are plotted.

        plot : bool
            If True, the plot will be shown.
        
        single_spin_plot : Union[None, list, tuple, np.ndarray, int, float]
            If not None, a single plot for each of the input angular
            momenta will be shown. If an integer or float is given,
            the plot will be shown for that angular momentum. If a
            list is given, the plot will be shown for each
            of the input angular momenta. If None, no plot will be
            shown.
        """
        if not isinstance(single_spin_plot, (type(None), list, tuple, np.ndarray, int, float)):
            msg = f"'single_spin_plot' must be of type: None, list, tuple, np.ndarray, int, float. Got {type(single_spin_plot)}."
            raise TypeError(msg)

        if isinstance(single_spin_plot, (int, float)):
            single_spin_plot = [single_spin_plot]

        if not isinstance(filter_parity, (type(None), int, str)):
            msg = f"'filter_parity' must be of type: None, int, str. Got {type(filter_spins)}."
            raise TypeError(msg)

        if isinstance(filter_parity, str):
            valid_filter_parity = ["+", "-"]
            if filter_parity not in valid_filter_parity:
                msg = f"Valid parity filters are: {valid_filter_parity}."
                raise ValueError(msg)
            
            filter_parity = 1 if (filter_parity == "+") else -1

        if filter_spins is None:
            """
            If no angular momentum filter, then include all angular
            momenta in the data set.
            """
            angular_momenta = np.unique(self.levels[:, 1])/2
        else:
            if isinstance(filter_spins, (float, int)):
                angular_momenta = [filter_spins]
            
            elif isinstance(filter_spins, (list, tuple, np.ndarray)):
                angular_momenta = filter_spins
            
            else:
                msg = f"'filter_spins' must be of type: None, list, int, float. Got {type(filter_spins)}."
                raise TypeError(msg)
        
        n_bins = int((self.levels[-1, 0] - self.levels[0, 0] + bin_width)/bin_width)
        n_angular_momenta = len(angular_momenta)
        bins = np.zeros((n_bins, n_angular_momenta))
        densities = np.zeros((n_bins, n_angular_momenta))

        for i in range(n_angular_momenta):
            """
            Calculate the nuclear level density for each angular
            momentum.
            """
            bins[:, i], densities[:, i] = level_density(
                levels = self.levels,
                bin_width = bin_width,
                filter_spins = angular_momenta[i],
                filter_parity = filter_parity,
                E_min = E_min,
                E_max = E_max,
                plot = False,
                save_plot = False
            )
        try:
            idx = np.where(bins[:, 0] > E_max)[0][0]
        except IndexError:
            idx = -1
        
        bins = bins[:idx]   # Remove bins of zero density.
        densities = densities[:idx]

        if filter_parity is None:
            exponent = r"$^{\pm}$"
        elif filter_parity == 1:
            exponent = r"$^{+}$"
        elif filter_parity == -1:
            exponent = r"$^{-}$"

        parity_str = "+" if (filter_parity == 1) else "-"

        if single_spin_plot:
            for j in single_spin_plot:
                if j not in angular_momenta:
                    msg = "Requested angular momentum is not present in the data."
                    msg += f" Allowed values are: {angular_momenta}, got {j}."
                    raise ValueError(msg)
            
            figax = []
            for i in range(len(single_spin_plot)):
                idx = np.where(angular_momenta == single_spin_plot[i])[0][0]  # Find the index of the angular momentum.

                figax.append(plt.subplots())
                label = r"$j^{\pi} =$" + f" {single_spin_plot[i]}" + exponent
                figax[i][1].step(bins[:, 0], densities[:, idx], label=label, color="black")
                figax[i][1].legend()
                figax[i][1].set_xlabel(r"$E$ [MeV]")
                figax[i][1].set_ylabel(r"NLD [MeV$^{-1}$]")

                if save_plot:
                    figax[i][0].savefig(
                        f"{self.nucleus}_j={single_spin_plot[i]}{parity_str}_distribution.png",
                        dpi = 300
                    )

        if plot:
            fig, ax = plt.subplots()
            ax = sns.heatmap(
                data = densities.T[-1::-1],
                linewidth = 0.5,
                annot = True,
                cmap = 'gray',
                ax = ax,
            )
            xticklabels = []
            for i in bins[:, 0]:
                if (tmp := int(i)) == i:
                    xticklabels.append(tmp)
                else:
                    xticklabels.append(round(i, 1))

            ax.set_xticklabels(xticklabels)
            ax.set_yticklabels(np.flip([f"{int(i)}" + exponent for i in angular_momenta]), rotation=0)
            ax.set_xlabel(r"$E$ [MeV]")
            ax.set_ylabel(r"$j$ [$\hbar$]")
            ax.set_title(f"{self.nucleus_latex}, {self.interaction}")
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel(r"NLD [MeV$^{-1}$]", rotation=90)

            if save_plot:
                fig.savefig(f"{self.nucleus}_j{parity_str}_distribution_heatmap.png", dpi=300)
        
        if plot or single_spin_plot:
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

def _process_kshell_output_in_parallel(args):
    """
    Simple wrapper for parallelizing loading of KSHELL files.
    """
    filepath, load_and_save_to_file, old_or_new = args
    print(filepath)
    return ReadKshellOutput(filepath, load_and_save_to_file, old_or_new)

def loadtxt(
    path: str,
    is_directory: bool = False,
    filter_: Union[None, str] = None,
    load_and_save_to_file: Union[bool, str] = True,
    old_or_new = "new"
    ) -> list:
    """
    Wrapper for using ReadKshellOutput class as a function.
    TODO: Consider changing 'path' to 'fname' to be the same as
    np.loadtxt.

    Parameters
    ----------
    path : str
        Filename (and path) of `KSHELL` output data file, or path to
        directory containing sub-directories with `KSHELL` output data.
    
    is_directory : bool
        If True, and 'path' is a directory containing sub-directories
        with `KSHELL` data files, the contents of 'path' will be scanned
        for `KSHELL` data files. Currently supports only summary files.

    filter_ : Union[None, str]
        NOTE: Shouldnt the type be list, not str?

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
    data : list
        List of instances with data from `KSHELL` data file as
        attributes.
    """
    loadtxt_time = time.perf_counter()  # Debug.
    all_fnames = None
    data = []
    if old_or_new not in (old_or_new_allowed := ["old", "new", "jem"]):
        msg = f"'old_or_new' argument must be in {old_or_new_allowed}!"
        msg += f" Got '{old_or_new}'."
        raise ValueError(msg)

    if (is_directory) and (not os.path.isdir(path)):
        msg = f"{path} is not a directory"
        raise NotADirectoryError(msg)

    elif (not is_directory) and (not os.path.isfile(path)):
        msg = f"{path} is not a file"
        raise FileNotFoundError(msg)

    elif (is_directory) and (os.path.isdir(path)):
        msg = "The 'is_directory' option is not properly tested and is"
        msg += " deprecated at the moment. Might return in the future."
        raise DeprecationWarning(msg)
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
            arg_list = [(path + i[0], load_and_save_to_file, old_or_new) for i in sub_fnames]
            data += pool.map(_process_kshell_output_in_parallel, arg_list)

    else:
        """
        Only a single KSHELL data file.
        """
        data.append(ReadKshellOutput(path, load_and_save_to_file, old_or_new))

    if not data:
        msg = "No KSHELL data loaded. Most likely error is that the given"
        msg += f" directory has no KSHELL data files. {path=}"
        raise RuntimeError(msg)

    loadtxt_time = time.perf_counter() - loadtxt_time
    if flags["debug"]:
        print(f"{loadtxt_time = } s")

    return data

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