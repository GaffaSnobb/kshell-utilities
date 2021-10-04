import os
import numpy as np
import matplotlib.pyplot as plt
from .kshell_utilities import atomic_numbers, loadtxt
from .general_utilities import create_spin_parity_list, gamma_strength_function_average

class LEE:
    def __init__(self, directory):
        self.bin_width = 0.2
        self.E_max = 30
        self.Ex_min = 0 # Lower limit for emitted gamma energy [MeV].
        self.Ex_max = 30 # Upper limit for emitted gamma energy [MeV].
        n_bins = int(np.ceil(self.E_max/self.bin_width))
        E_max_adjusted = self.bin_width*n_bins
        bins = np.linspace(0, E_max_adjusted, n_bins + 1)
        self.bins_middle = (bins[0: -1] + bins[1:])/2

        self.all_fnames = {}

        self.directory = directory

        for element in sorted(os.listdir(self.directory)):
            """
            List all content in self.directory.
            """
            if os.path.isdir(f"{self.directory}/{element}"):
                """
                If element is a directory, enter it to find data files.
                """
                self.all_fnames[element] = []    # Create blank entry in dict for current element.
                for isotope in os.listdir(f"{self.directory}/{element}"):
                    """
                    List all content in the element directory.
                    """
                    if isotope.startswith("summary"):
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
                        
                        self.all_fnames[element].append([f"{element}/{isotope}", n_neutrons])
        
        for key in self.all_fnames:
            """
            Sort each list in the dict by the number of neutrons.
            """
            self.all_fnames[key].sort(key=lambda tup: tup[1])   # Why not do this when directory is listed?

    # def plot_gsf(self, isotope_name):
    #     """
    #     Plot the gamma strength function for a single isotope.

    #     isotope_name : string
    #         Examples: S24, Ne30.

    #     Raises
    #     ------
    #     ValueError
    #         If isotope_name cannot be found in the calculated data
    #         files.
    #     """
    #     fname = None
        
    #     for fnames in self.fnames_combined:
    #         for i in range(len(fnames)):
    #             if isotope_name in fnames[i][0]:
    #                 fname = fnames[i][0]

    #     if fname is None:
    #         msg = f"Isotope name '{isotope_name}' is not a valid name."
    #         raise ValueError(msg)

    #     res = loadtxt(self.directory + fname)

    #     _, ax = plt.subplots()

    #     Jpi_list = create_jpi_list(res.levels[:, 1], None)
    #     E_gs = res.levels[0, 0]
    #     res.transitions[:, 2] += E_gs   # Add ground state energy for compatibility with Jørgen.

    #     gsf = strength_function_average(
    #         levels = res.levels,
    #         transitions = res.transitions,
    #         Jpi_list = Jpi_list,
    #         bin_width = self.bin_width,
    #         Ex_min = self.Ex_min,    # [MeV].
    #         Ex_max = self.Ex_max,    # [MeV].
    #         multipole_type = "M1"
    #     )

    #     bin_slice = self.bins_middle[0:len(gsf)]
    #     ax.plot(bin_slice, gsf, label=fname)
    #     ax.legend()
    #     ax.set_xlabel(r"$E_{\gamma}$ [MeV]")
    #     ax.set_ylabel(r"gsf [MeV$^{-3}$]")
    #     plt.show()


    def calculate_low_energy_enhancement(self, filter=None):
        """
        Recreate the figure from Jørgens article.
        """
        self.labels = []    # Suggested labels for plotting.
        self.ratios = []
        self.n_neutrons = []

        for key in self.all_fnames:
            """
            Loop over all elements (grunnstoff).
            """
            fnames = self.all_fnames[key]   # For compatibility with old code.
            if filter is not None:
                if key.split("_")[1] not in filter:
                    """
                    Skip elements not in filter.
                    """
                    continue
            
            ratios = [] # Reset ratio for every new element.
            for i in range(len(fnames)):
                """
                Loop over all isotopes per element.
                """
                try:
                    res = loadtxt(f"{self.directory}/{fnames[i][0]}")
                except FileNotFoundError:
                    print(f"File {fnames[i][0]} skipped! File not found.")
                    ratios.append(None) # Maintain correct list length for plotting.
                    continue

                Jpi_list = create_spin_parity_list(
                    spins = res.levels[:, 1],
                    parities = res.levels[:, 2]
                )
                E_gs = res.levels[0, 0]

                try:
                    res.transitions[:, 2] += E_gs   # Add ground state energy for compatibility with Jørgen.
                except IndexError:
                    print(f"File {fnames[i][0]} skipped! Too few / no energy levels are present in this data file.")
                    ratios.append(None) # Maintain correct list length for plotting.
                    continue
                
                try:
                    gsf = strength_function_average(
                        levels = res.levels,
                        transitions = res.transitions,
                        Jpi_list = Jpi_list,
                        bin_width = self.bin_width,
                        Ex_min = self.Ex_min,    # [MeV].
                        Ex_max = self.Ex_max,    # [MeV].
                        multipole_type = "M1"
                    )
                except IndexError:
                    print(f"File {fnames[i][0]} skipped! That unknown index out of bounds error in ksutil.")
                    ratios.append(None)
                    continue

                # Sum gsf for low and high energy range and take the ratio.
                bin_slice = self.bins_middle[0:len(gsf)]
                low_idx = (bin_slice <= 2)
                high_idx = (bin_slice <= 6) == (2 <= bin_slice)
                low = np.sum(gsf[low_idx])
                high = np.sum(gsf[high_idx])
                low_high_ratio = low/high
                ratios.append(low_high_ratio)

                print(f"{fnames[i][0]} loaded")

            if all(elem is None for elem in ratios):
                """
                Skip current element if no ratios are calculated.
                """
                continue
            
            self.labels.append(fnames[0][0][:fnames[0][0].index("/")])
            self.n_neutrons.append([n_neutrons for _, n_neutrons in fnames])
            self.ratios.append(ratios)


    def quick_plot(self):
        fig, ax = plt.subplots()
        for i in range(len(self.n_neutrons)):
            ax.plot(self.n_neutrons[i], self.ratios[i], ".--", label=self.labels[i])
            ax.set_yscale("log")
            ax.set_xlabel("N")
            ax.set_ylabel("Rel. amount of low-energy strength")
            ax.legend()

        plt.show()


def low_energy_enhancement(directory):
    """
    Wrapper for easier usage.

    Parameters
    ----------
    directory : string
        Directory containing subfolders with KSHELL data.

    Returns
    -------
    res : kshell_utilities.low_energy_enhancement.LEE
        Class instance containing LEE data.
    """
    res = LEE(directory)
    res.calculate_low_energy_enhancement()

    return res