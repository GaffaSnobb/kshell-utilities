from fractions import Fraction
import numpy as np

atomic_numbers = {
    "oxygen": 8, "fluorine": 9, "neon": 10, "sodium": 11, "magnesium": 12,
    "aluminium": 13, "silicon": 14, "phosphorus": 15, "sulfur": 16,
    "chlorine": 17, "argon": 18
}

def create_jpi_list(spins, parities):
    """
    Example list:
    [[1, +1], [3, +1], [5, +1], [7, +1], [9, +1], [11, +1], [13, +1]].
    Currently hard-coded for only positive parity states.

    Parameters
    ----------
    spins : numpy.ndarray
        Array of spins for each energy level.

    Returns
    -------
    spins_new : list
        A nested list of spins and parities [[spin, parity], ...] sorted
        with respect to the spin.
    """
    spins[spins < 0] = 0    # Discard negative entries.
    spins_new = []
    for elem in spins:
        if elem not in spins_new:
            """
            Extract each distinct value only once.
            """
            spins_new.append([elem])

    for i in range(len(spins_new)):
        """
        Add parity for each spin. [spin, parity].
        """
        spins_new[i].append(+1)
    
    return sorted(spins_new, key=lambda tup: tup[0])

class ReadKshellOutput:
    """
    Implemented as class just to avoid returning a bunch of values.
    Access instance attributes instead.
    """
    def read_kshell_output(self, fname):
        """
        Read energy level data, transition probabilities and transition
        strengths from KSHELL output files.

        Parameters
        ----------
        fname : string
            Filename of KSHELL output file.

        Returns
        -------
        E_x : numpy.ndarray
            1D array of energy levels.

        B_M1 : numpy.ndarray
            Nx3 matrix with rows [E_x, b_m1, E_gamma].

        Raises
        ------
        RuntimeError
            If the KSHELL file has unexpected structure / syntax.
        """

        self.E_x = []    
        self.B_M1 = []
        self.levels = [] # [Ei, 2*Ji, parity].
        self.transitions = []   # [2J_f, p_i, E_f, 2J_i, p_i, E_i, E_gamma, B(.., i->f)].

        def load_energy_levels(infile):
            for _ in range(3): infile.readline()
            for line in infile:
                try:
                    tmp = line.split()
                    self.E_x.append(float(tmp[6]))
                    parity = 1 if tmp[2] == "+" else -1
                    self.levels.append([float(tmp[5]), 2*float(Fraction(tmp[1])), parity])
                except IndexError:
                    """
                    End of energies.
                    """
                    break

        def load_m1_probabilities(infile):
            """
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
                    """
                    tmp = line.split()
                    
                    # Location of initial parity is common for all cases.
                    parity_idx = tmp[0].index("(") - 1 # Find index of initial parity.
                    p_i = 1 if tmp[0][parity_idx] == "+" else -1
                    
                    # Location of initial spin is common for all cases.
                    J_i = float(Fraction(tmp[0][:parity_idx]))
                    
                    if (tmp[1][-1] != ")") and (tmp[3][-1] != ")"):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        2+(11) 18.393 2+(10) 17.791 0.602 0.1(  0.0) 0.1( 0.0)
                        """
                        E_gamma = float(tmp[4])
                        E_i = float(tmp[1])
                        b_m1 = float(tmp[5][:-1])
                        J_f = float(Fraction(tmp[2][:-5]))
                        E_f = float(tmp[3])
                        self.B_M1.append([E_i, b_m1, E_gamma])
                        self.transitions.append([2*J_f, p_i, E_f, 2*J_i, p_i, E_i, E_gamma, b_m1])

                    elif (tmp[1][-1] != ")") and (tmp[3][-1] == ")"):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        2+(10) 17.791 2+( 1) 5.172 12.619 0.006( 0.00) 0.006( 0.00)
                        """
                        E_gamma = float(tmp[5])
                        E_i = float(tmp[1])
                        b_m1 = float(tmp[6][:-1])
                        J_f = float(Fraction(tmp[2][:-2]))
                        E_f = float(tmp[4])
                        self.B_M1.append([E_i, b_m1, E_gamma])
                        self.transitions.append([2*J_f, p_i, E_f, 2*J_i, p_i, E_i, E_gamma, b_m1])
                    
                    elif (tmp[1][-1] == ")") and (tmp[4][-1] != ")"):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        3+( 8) 19.503 2+(11) 18.393 1.111 0.000( 0.00) 0.000( 0.00)
                        """
                        E_gamma = float(tmp[5])
                        E_i = float(tmp[2])
                        b_m1 = float(tmp[6][:-1])
                        J_f = float(Fraction(tmp[3][:-5]))
                        E_f = float(tmp[4])
                        self.B_M1.append([E_i, b_m1, E_gamma])
                        self.transitions.append([2*J_f, p_i, E_f, 2*J_i, p_i, E_i, E_gamma, b_m1])

                    elif (tmp[1][-1] == ")") and (tmp[4][-1] == ")"):
                        """
                        Example:
                        J_i    Ex_i     J_f    Ex_f   dE        B(M1)->         B(M1)<- 
                        1+( 7) 19.408 2+( 9) 16.111 3.297 0.005( 0.00) 0.003( 0.00)
                        """
                        E_gamma = float(tmp[6])
                        E_i = float(tmp[2])
                        b_m1 = float(tmp[7][:-1])
                        J_f = float(Fraction(tmp[3][:-2]))
                        E_f = float(tmp[5])
                        self.B_M1.append([E_i, b_m1, E_gamma])
                        self.transitions.append([2*J_f, p_i, E_f, 2*J_i, p_i, E_i, E_gamma, b_m1])
                    else:
                        msg = "WARNING: Structure not accounted for!"
                        msg += f"\n{line=}"
                        raise RuntimeError(msg)

                except IndexError:
                    """
                    End of probabilities.
                    """
                    break

        with open(fname, "r") as infile:
            for line in infile:
                tmp = line.split()
                try:
                    if tmp[0] == "Energy":
                        load_energy_levels(infile)
                    
                    elif tmp[0] == "B(E2)":
                        continue
                    
                    elif tmp[0] == "B(M1)":
                        load_m1_probabilities(infile)
                
                except IndexError:
                    """
                    Skip blank lines.
                    """
                    continue

        self.levels = np.array(self.levels)
        self.transitions = np.array(self.transitions)
        return np.array(self.E_x), np.array(self.B_M1)

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

def loadtxt(fname):
    """
    Wrapper for using ReadKshellOutput class as a function.

    Parameters
    ----------
    fname : string
        Filename (and path) of KSHELL output data file.

    Returns
    -------
    data : kshell_utilities.ReadKshellOutput
        Class object with data from KSHELL data file as attributes.
    """
    data = ReadKshellOutput()
    data.read_kshell_output(fname)
    return data

def div0(a, b):
    """
    Author: Jørgen Midtbø.
    Division function designed to ignore / 0, i.e.
    div0( [-1, 0, 1], 0 ) -> [0, 0, 0].
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0    # -inf inf NaN
    return c

def strength_function_average(
        levels,
        transitions,
        Jpi_list,
        bin_width,
        Ex_min,
        Ex_max,
        multipole_type = "M1"
    ):
    """
    Author: Jørgen Midtbø.
    Modified by: Jon Dahl.
    20171009: Updated the way we average over Ex, J, pi to only count pixels with non-zero gSF.
    20170815: This function returns the strength function the way we now think is the correct way:
    By taking only the partial level density corresponding to the specific (Ex, J, pi) pixel in the
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
    #     B_pixel_count[i_Ex,i_Eg,i_Jpi] += 1
    #     Ex_already_seen[i_Eg].append(Ex)

    Parameters
    ----------
    levels : numpy.ndarray
        Nx3 matrix containing [Ei, 2*Ji, parity] in each row.

    transitions : numpy.ndarray
        Mx8 matrix containing [2Jf, pi, Ef, 2Ji, pi, Ei, Ex, B(.., i->f)]
        in each row.

    Jpi_list : list
        Set a spin window by defining a list of allowed initial
        [[spins, parities], ...]. 

    Ex_min : int, float
        Lower limit for emitted gamma energy [MeV].

    Ex_max : int, float
        Upper limit for emitted gamma energy [MeV].

    multipole_type : string
        Choose whether to calculate for 'M1' or 'E2'.
    """
    Nbins = int(np.ceil(Ex_max/bin_width)) # Make sure the number of bins cover the whole Ex region.
    # bin_array = np.linspace(0, bin_width*Nbins, Nbins + 1) # Array of lower bin edge energy values
    # bin_array_middle = (bin_array[0: -1] + bin_array[1:])/2 # Array of middle bin values
    
    # Find index of first and last bin (lower bin edge) where we put counts.
    # It's important to not include the other Ex bins in the averaging later, because they contain zeros which will pull the average down.
    i_Ex_min = int(np.floor(Ex_min/bin_width)) 
    i_Ex_max = int(np.floor(Ex_max/bin_width))    

    prefactor = {"M1": 11.5473e-9, "E1": 1.047e-6}

    Egs = levels[0, 0] # Read out the absolute ground state energy, so we can get relative energies later.

    # Allocate matrices to store the summed B(M1) values for each pixel,
    # and the number of transitions counted.
    B_pixel_sum = np.zeros((Nbins, Nbins, len(Jpi_list)))
    B_pixel_count = np.zeros((Nbins, Nbins, len(Jpi_list)))

    for i_tr in range(len(transitions[:, 0])):
        """
        Iterate over all transitions in the transitions matrix and put
        in the correct pixel.
        """
        Ex = transitions[i_tr, 2] - Egs # Calculate relative energy.
        if (Ex < Ex_min) or (Ex >= Ex_max):
            """
            Check if transition is within min max limits, skip if not.
            """
            continue

        # Get bin index for Eg and Ex (initial). Indices are defined with respect to the lower bin edge.
        i_Eg = int(np.floor(transitions[i_tr, 6]/bin_width))
        i_Ex = int(np.floor(Ex/bin_width))

        # Read initial spin and parity of level: NOTE: I think the name / index is wrong.
        Ji = int(transitions[i_tr, 0])
        pi = int(transitions[i_tr, 1])
        try:
            """
            Get index for current [Ji, pi] combination in Jpi_list.
            """
            i_Jpi = Jpi_list.index([Ji, pi])
        except ValueError:
            continue

        # Add B(M1) value and increment count to pixel, respectively
        B_pixel_sum[i_Ex, i_Eg, i_Jpi] += transitions[i_tr, 7]
        B_pixel_count[i_Ex, i_Eg, i_Jpi] += 1 # Original.



    # Allocate (Ex, Jpi) matrix to store level density
    rho_ExJpi = np.zeros((Nbins, len(Jpi_list)))
    # Count number of levels for each (Ex, J, pi) pixel.
    for i_l in range(len(levels[:, 0])):
        E, J, pi = levels[i_l]
        
        if (E - Egs) >= Ex_max:
            """
            Skip if level is outside range.
            """
            continue

        i_Ex = int(np.floor((E - Egs)/bin_width))
        try:
            i_Jpi = Jpi_list.index([J, pi])
        except:
            continue
        rho_ExJpi[i_Ex, i_Jpi] += 1

    rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1.


    # Calculate gamma strength functions for each Ex, J, pi individually, using the partial level density for each J, pi.
    gSF = np.zeros((Nbins, Nbins, len(Jpi_list)))
    a = prefactor[multipole_type] # mu_N^-2 MeV^-2, conversion constant
    for i_Jpi in range(len(Jpi_list)):
        for i_Ex in range(Nbins):
            gSF[i_Ex, :, i_Jpi] = a*rho_ExJpi[i_Ex, i_Jpi]*div0(
                B_pixel_sum[i_Ex, :, i_Jpi],
                B_pixel_count[i_Ex, :, i_Jpi]
            )

    # Return the average gSF(Eg) over all (Ex,J,pi)

    # return gSF[i_Ex_min:i_Ex_max+1,:,:].mean(axis=(0,2))
    # Update 20171009: Took proper care to only average over the non-zero f(Eg,Ex,J,pi) pixels:
    gSF_currentExrange = gSF[i_Ex_min:i_Ex_max + 1, :, :]
    gSF_ExJpiavg = div0(
        gSF_currentExrange.sum(axis = (0, 2)),
        (gSF_currentExrange != 0).sum(axis = (0, 2))
    )
    return gSF_ExJpiavg
