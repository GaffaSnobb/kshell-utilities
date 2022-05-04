"""
Author: Noritaka Shimizu (?)
Modified by: Jon Dahl
"""
import sys, os, warnings
from typing import List, Tuple
from fractions import Fraction
from math import pi

#weisskopf_threshold = 1.0 # threshold to show in W.u.
weisskopf_threshold = -0.001

n_jnp = {}

E_gs = 0.0

def parity_integer_to_string(i: int) -> str:
    """
    Convert 1 to '+' and -1 to '-'.

    Parameters
    ----------
    i : int
        Parity in integer representation.

    Returns
    -------
    : str
        The string representation of the parity.
    """
    if i == 1: return '+'
    else: return '-'

def weisskopf_unit(multipole_type: str, mass: int) -> Tuple[float, str]:
    """
    Generate the Weisskopf unit_weisskopf for input multipolarity and mass.
    Ref. Bohr and Mottelson, Vol.1, p. 389.

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

    unit_weisskopf : str
        The accompanying unit.
    """
    l = int(multipole_type[1:])
    if multipole_type[0].upper() == "E":
        B_weisskopf = 1.2**(2*l)/(4*pi)*(3/(l + 3))**2*mass**(2*l/3)  
        unit_weisskopf = f"e^2 fm^{str(2*l)}"
    
    elif multipole_type[0].upper() == "M":
        B_weisskopf = 10/pi*1.2**(2*l - 2)*(3/(l + 3))**2*mass**((2*l - 2)/3) 
        if l == 1: 
            unit_weisskopf = 'mu_N^2  '
        else:
            unit_weisskopf = f"mu_N^2 fm^{str(2*l - 2)}"
    else:
        msg = f"Got invalid multipole type: '{multipole_type}'."
        raise ValueError(msg)
    
    return B_weisskopf, unit_weisskopf
    
def read_energy_logfile(filename: str, E_data: dict):
    """
    Extract the energy, spin, and parity for each eigenstate and arrange
    the data in a dictionary, E_data, where the keys are the energies
    and the values are tuples of
    (log filename, spin, parity, eigenstate number, tt).

    The transition logs will not be read by this function as they do not
    contain the energy level information. Only the KSHELL logs will be
    read.

    Parameters
    ----------
    filename : str
        The log filename.
    """
    with open(filename, "r") as infile:
        while True:
            line = infile.readline()
            if not line: break
            if len(line) >= 11 and line[:11] != "H converged": continue
            if len(line) >= 14 and line[:14] != "H bn converged": continue
            while True:
                line = infile.readline()
                if not line: break
                if line[6:10] == '<H>:':
                    """
                    Example:
                    -------------------------------------------------
                    1  <H>:  -391.71288  <JJ>:    -0.00000  J:  0/2  prty -1     <-- this line will be read
                        <Hcm>:     0.00022  <TT>:     6.00000  T:  4/2              <-- this line will be read
                    <p Nj>  5.944  3.678  1.489  3.267  0.123  0.456  0.043        <-- this line will be skipped
                    <n Nj>  5.993  3.902  1.994  5.355  0.546  0.896  0.314        <-- this line will be skipped
                    hw:  1:1.000                                                   <-- this line will be skipped
                    -------------------------------------------------
                    NOTE: All this substring indexing seems to be easily
                    replaceable with a simple split!
                    """
                    n_eig = int(line[:5])       # Eigenvalue number. 1, 2, 3, ...
                    energy = float(line[11:22])
                    spin = int(line[45:48])     # 2*spin actually.
                    parity = int(line[57:59])
                    parity = parity_integer_to_string(parity)
                    while energy in E_data: energy += 0.000001  # NOTE: To separate energies close together? Else keys may be identical!
                    while True:
                        line = infile.readline()
                        if line[42:45] != ' T:': continue
                        tt = int(line[45:48])
                        E_data[ energy ] = (filename, spin, parity, n_eig, tt)
                        break

def spin_to_string(spin: int) -> str:
    """
    Divide spin by 2 and represent integer results as integers, and
    fraction results as fractions.

    NOTE: Tempting to just do str(Fraction(spin/2)) on the negatives too
    and just let them be negative fractions.

    Parameters
    ----------
    spin : int
        Two times the actual spin (for easier integer representation).

    Returns
    -------
    res : str
        The actual spin (input divided by 2) represented correctly as
        an integer or as a fraction.
    """
    if spin < 0:
        """
        For the invalid -1 cases. Spin cannot actually be < 0.
        """
        res = str(spin)
    else:
        res = str(Fraction(spin/2))

    return res

def read_transit_logfile_old(
    filename: str,
    multipole_type: str
    ):
    """
    Extract transit information from transit logfile. Old syntax style,
    pre 2021-11-24.

    Parameters
    ----------
    filename : str
        Filename of the log file.

    multipole_type : str
        The electromagnetic character and angular momentum of the gamma
        radiation. Examples: 'E1', 'M1', 'E2'.

    Raises
    ------
    Exception:
        If mass != mass_save. Unsure why mass_save is here at all.

    ValueError:
        If the conversion of log file values to int or float cannot be
        done. This indicates wrong log syntax.
    """
    out_e = {}
    mass_save = 0           # NOTE: Unclear what this is used for.
    with open(filename, "r") as infile:
        for line in infile:
            """
            Fetch mass number, wave function filenames and parities.
            Loop breaks when the line with parity information is found.
            """
            line_split = line.split()
            if len(line_split) == 0: continue   # Skip empty lines.
            if 'mass=' in line:
                """
                Fetch mass number. Example:
                N. of valence protons and neutrons =  15 19   mass= 50   n,z-core     8    8
                """
                n = line.index('mass=')
                mass = int(line[n + 5:n + 8])
                if not mass_save:
                    mass_save = mass
                
                if mass_save != mass: 
                    msg = f"ERROR  mass: {mass=}, {mass_save=}"
                    raise RuntimeError(msg)
                
                B_weisskopf, unit_weisskopf = weisskopf_unit(multipole_type, mass)
                continue

            if line_split[0] == 'fn_load_wave_l': 
                filename_wavefunction_left = line_split[2]
                continue
            
            if line_split[0] == 'fn_load_wave_r': 
                filename_wavefunction_right = line_split[2]
                continue
            
            if f"{multipole_type} transition" in line:
                """
                Example:
                 E2 transition  e^2 fm^4  eff_charge=  1.3500  0.3500 parity -1 -1
                   2Jf   idx  Ef        2Ji   idx  Ei          Ex        Mred.           B(EM )->        B(EM)<-         Mom.
                   4     1   -387.729   4     1   -387.729     0.000    -20.43244863     83.49699137     83.49699137    -15.48643011
                ...
                """
                parity_final = parity_integer_to_string(int(line_split[-2]))
                parity_initial = parity_integer_to_string(int(line_split[-1]))
                
                if filename_wavefunction_left == filename_wavefunction_right:
                    is_diag = True
                else:
                    is_diag = False
                break

            continue    # Included for readability.

        infile.readline()    # Skip table header.
        for line in infile:
            """
            Extract transition data from log_*_tr_*.txt. Example:

            NEW (higher decimal precision and only whitespace between values):
            E2 transition  e^2 fm^4  eff_charge=  1.3500  0.3500 parity -1 -1
            2Jf   idx  Ef        2Ji   idx  Ei          Ex        Mred.           B(EM )->        B(EM)<-         Mom.
            4     1   -387.729   4     1   -387.729     0.000    -20.43244863     83.49699137     83.49699137    -15.48643011
            4     1   -387.729   6     2   -387.461     0.267     10.55639593     22.28749899     15.91964213      0.00000000
            4     1   -387.729   8     3   -387.196     0.532     14.53838975     42.27295529     23.48497516      0.00000000
            4     1   -387.729   6     4   -387.080     0.649    -17.34631937     60.17895916     42.98497083      0.00000000
            4     1   -387.729   8     5   -386.686     1.042    -11.24379628     25.28459094     14.04699497      0.00000000
            ...

            OLD:
            ...
            4(   2) -200.706 6(   4) -198.842   1.864    0.0147    0.0000    0.0000    0.0000
            4(   2) -200.706 6(  10) -197.559   3.147   -0.0289    0.0002    0.0001    0.0000
            ...
            """
            line_split = line.split()
            if not line_split: break    # End file read when blank lines are encountered.
            if line.startswith("pn="):
                """
                jonkd: I had to add this because 'pn' showed up in the
                middle of the 'log_Ni56_gxpf1a_tr_m0p_m0p.txt' file
                after trying (unsuccessfully) to run on Fram. Example:

                ...
                4(   2) -200.706 6(   4) -198.842   1.864    0.0147    0.0000    0.0000    0.0000
                pn= 1   # of mbits=            286
                4(   2) -200.706 6(  10) -197.559   3.147   -0.0289    0.0002    0.0001    0.0000
                ...
                """
                continue
            
            try:
                spin_final = int(line[:2])
                idx_1 = int(line[3:7])
                spin_initial = int(line[17:19])
                idx_2 = int(line[20:24])
                dE = float(line[34:42]) # Gamma energy.
                E_final = float(line[8:17]) - E_gs
                E_initial = float(line[25:34]) - E_gs
                B_decay = float(line[52:62])
                B_excite = float(line[62:72])
            except ValueError as err:
                msg = f"\n{err.__str__()}"
                msg += "\nThis might be due to wrong log file syntax."
                msg += " Try using old_or_new='new' or 'both' as argument"
                msg += " to collect_logs."
                msg += f"\n{filename = }"
                msg += f"\n{line = }"
                raise ValueError(msg)

            B_weisskopf_decay  = B_decay/B_weisskopf
            B_weisskopf_excite = B_excite/B_weisskopf
            if (spin_final == spin_initial) and (idx_1 == idx_2): continue
            if is_diag and (dE < 0.0): continue
            if (B_weisskopf_decay < weisskopf_threshold): continue
            if (B_weisskopf_excite < weisskopf_threshold): continue
            if abs(E_final) < 1e-3: E_final = 0.
            if abs(E_initial) < 1e-3: E_initial = 0.
            
            idx_1 = n_jnp[ (spin_final, parity_final, idx_1) ]
            idx_2 = n_jnp[ (spin_initial, parity_initial, idx_2) ]
                
            if dE > 0:
                out = f"{spin_to_string(spin_initial):4s} "
                out += f"{parity_initial:1s} "
                out += f"{idx_2:4d} "
                out += f"{E_initial:9.3f}   "
                out += f"{spin_to_string(spin_final):4s} "
                out += f"{parity_final:1s} "
                out += f"{idx_1:4d} "
                out += f"{E_final:9.3f} "
                out += f"{dE:9.3f} "
                out += f"{B_excite:15.8f} "
                out += f"{B_weisskopf_excite:15.8f} "
                out += f"{B_decay:15.8f} "
                out += f"{B_weisskopf_decay:15.8f}\n"
                key = E_initial + E_final * 1e-5 + spin_initial *1e-10 + idx_2*1e-11 + spin_final*1e-13 + idx_1*1e-14
            else:
                """
                NOTE: What is this option used for? In what case is the
                excitation energy negative?
                """
                out = f"{spin_to_string(spin_final):4s} "
                out += f"{parity_final:1s} "
                out += f"{idx_1:4d} "
                out += f"{E_final:9.3f}   "
                out += f"{spin_to_string(spin_initial):4s} "
                out += f"{parity_initial:1s} "
                out += f"{idx_2:4d} "
                out += f"{E_initial:9.3f} "
                out += f"{-dE:9.3f} "
                out += f"{B_decay:15.8f} "
                out += f"{B_weisskopf_decay:15.8f} "
                out += f"{B_excite:15.8f} "
                out += f"{B_weisskopf_excite:15.8f}\n"
                key = E_final + E_initial * 1e-5 + spin_final *1.e-10 + idx_1*1.e-11 + spin_initial*1.e-12 + idx_2*1.e-14
            out_e[key] = out

    return unit_weisskopf, out_e, mass_save   

def read_transit_logfile(filename: str, multipole_type: str):
    """
    Extract transit information from transit logfile.

    Parameters
    ----------
    filename : str
        Filename of the log file.

    multipole_type : str
        The electromagnetic character and angular momentum of the gamma
        radiation. Examples: 'E1', 'M1', 'E2'.

    Raises
    ------
    Exception:
        If mass != mass_save. Unsure why mass_save is here at all.

    ValueError:
        If the conversion of log file values to int or float cannot be
        done. This indicates wrong log syntax.
    """
    out_e = {}
    mass_save = 0           # NOTE: Unclear what this is used for.
    unit_weisskopf = None   # Stays None if the logfile is invalid.
    with open(filename, "r") as infile:
        for line in infile:
            """
            Fetch mass number, wave function filenames and parities.
            Loop breaks when the line with parity information is found.
            """
            line_split = line.split()
            if len(line_split) == 0: continue   # Skip empty lines.
            if 'mass=' in line:
                """
                Fetch mass number. Example:
                N. of valence protons and neutrons =  15 19   mass= 50   n,z-core     8    8
                """
                n = line.index('mass=')
                mass = int(line[n + 5:n + 8])
                if not mass_save:
                    mass_save = mass
                
                if mass_save != mass: 
                    msg = f"ERROR  mass: {mass=}, {mass_save=}"
                    raise RuntimeError(msg)
                
                B_weisskopf, unit_weisskopf = weisskopf_unit(multipole_type, mass)
                continue

            if line_split[0] == 'fn_load_wave_l': 
                filename_wavefunction_left = line_split[2]
                continue
            
            if line_split[0] == 'fn_load_wave_r': 
                filename_wavefunction_right = line_split[2]
                continue
            
            if f"{multipole_type} transition" in line:
                """
                Example:
                 E2 transition  e^2 fm^4  eff_charge=  1.3500  0.3500 parity -1 -1
                   2Jf   idx  Ef        2Ji   idx  Ei          Ex        Mred.           B(EM )->        B(EM)<-         Mom.
                   4     1   -387.729   4     1   -387.729     0.000    -20.43244863     83.49699137     83.49699137    -15.48643011
                ...
                """
                parity_final = parity_integer_to_string(int(line_split[-2]))
                parity_initial = parity_integer_to_string(int(line_split[-1]))
                
                if filename_wavefunction_left == filename_wavefunction_right:
                    is_diag = True
                else:
                    is_diag = False
                break

            continue    # Included for readability.

        infile.readline()    # Skip table header.
        for line in infile:
            """
            Extract transition data from log_*_tr_*.txt. Example:

            NEW (higher decimal precision and only whitespace between values):
            E2 transition  e^2 fm^4  eff_charge=  1.3500  0.3500 parity -1 -1
            2Jf   idx  Ef        2Ji   idx  Ei          Ex        Mred.           B(EM )->        B(EM)<-         Mom.
            4     1   -387.729   4     1   -387.729     0.000    -20.43244863     83.49699137     83.49699137    -15.48643011
            4     1   -387.729   6     2   -387.461     0.267     10.55639593     22.28749899     15.91964213      0.00000000
            4     1   -387.729   8     3   -387.196     0.532     14.53838975     42.27295529     23.48497516      0.00000000
            4     1   -387.729   6     4   -387.080     0.649    -17.34631937     60.17895916     42.98497083      0.00000000
            4     1   -387.729   8     5   -386.686     1.042    -11.24379628     25.28459094     14.04699497      0.00000000
            ...

            OLD:
            ...
            4(   2) -200.706 6(   4) -198.842   1.864    0.0147    0.0000    0.0000    0.0000
            4(   2) -200.706 6(  10) -197.559   3.147   -0.0289    0.0002    0.0001    0.0000
            ...
            """
            line_split = line.split()
            if not line_split: break    # End file read when blank lines are encountered.
            if line.startswith("pn="):
                """
                jonkd: I had to add this because 'pn' showed up in the
                middle of the 'log_Ni56_gxpf1a_tr_m0p_m0p.txt' file
                after trying (unsuccessfully) to run on Fram. Example:

                ...
                4(   2) -200.706 6(   4) -198.842   1.864    0.0147    0.0000    0.0000    0.0000
                pn= 1   # of mbits=            286
                4(   2) -200.706 6(  10) -197.559   3.147   -0.0289    0.0002    0.0001    0.0000
                ...
                """
                continue
            try:
                spin_final   = int(line_split[0])
                idx_1        = int(line_split[1])
                E_final      = float(line_split[2]) - E_gs
                spin_initial = int(line_split[3])
                idx_2        = int(line_split[4])
                E_initial    = float(line_split[5]) - E_gs
                dE           = float(line_split[6])
                Mred         = float(line_split[7])
                B_decay      = float(line_split[8])
                B_excite     = float(line_split[9])
                Mom          = float(line_split[10])
            except ValueError as err:
                msg = f"\n{err.__str__()}"
                msg += "\nThis might be due to wrong log file syntax."
                msg += " Try using old_or_new='old' or 'both' as argument to"
                msg += " collect_logs."
                msg += f"\n{filename = }"
                msg += f"\n{line = }"
                raise ValueError(msg)

            B_weisskopf_decay  = B_decay/B_weisskopf
            B_weisskopf_excite = B_excite/B_weisskopf

            if (spin_final == spin_initial) and (idx_1 == idx_2): continue
            if is_diag and (dE < 0.0): continue
            if (B_weisskopf_decay < weisskopf_threshold): continue
            if (B_weisskopf_excite < weisskopf_threshold): continue
            if abs(E_final) < 1e-3: E_final = 0.
            if abs(E_initial) < 1e-3: E_initial = 0.
            
            idx_1 = n_jnp[ (spin_final, parity_final, idx_1) ]
            idx_2 = n_jnp[ (spin_initial, parity_initial, idx_2) ]
                
            if dE > 0.0:
                out = f"{spin_to_string(spin_initial):4s} "
                out += f"{parity_initial:1s} "
                out += f"{idx_2:4d} "
                out += f"{E_initial:9.3f}   "
                out += f"{spin_to_string(spin_final):4s} "
                out += f"{parity_final:1s} "
                out += f"{idx_1:4d} "
                out += f"{E_final:9.3f} "
                out += f"{dE:9.3f} "
                out += f"{B_excite:15.8f} "
                out += f"{B_weisskopf_excite:15.8f} "
                out += f"{B_decay:15.8f} "
                out += f"{B_weisskopf_decay:15.8f}\n"
                key = E_initial + E_final * 1e-5 + spin_initial *1e-10 + idx_2*1e-11 + spin_final*1e-13 + idx_1*1e-14
            else:
                """
                NOTE: What is this option used for? In what case is the
                excitation energy negative?
                """
                out = f"{spin_to_string(spin_final):4s} "
                out += f"{parity_final:1s} "
                out += f"{idx_1:4d} "
                out += f"{E_final:9.3f}   "
                out += f"{spin_to_string(spin_initial):4s} "
                out += f"{parity_initial:1s} "
                out += f"{idx_2:4d} "
                out += f"{E_initial:9.3f} "
                out += f"{-dE:9.3f} "
                out += f"{B_decay:15.8f} "
                out += f"{B_weisskopf_decay:15.8f} "
                out += f"{B_excite:15.8f} "
                out += f"{B_weisskopf_excite:15.8f}\n"
                key = E_final + E_initial * 1e-5 + spin_final *1.e-10 + idx_1*1.e-11 + spin_initial*1.e-12 + idx_2*1.e-14
            out_e[key] = out

    return unit_weisskopf, out_e, mass_save

def check_multipolarities(path: str="."):
    """
    Check whether E1, M1, and E2 values are present in transit logfiles.
    Currently only checks that the correct headers are there, not the
    actual values.

    Parameters
    ----------
    path : str
        Path to directory with transit logfiles.
    """
    print("logfile                                 E1        M1        E2        Is it correct?")
    print("------------------------------------------------------------------------------------")
    # print("log_V50_sdpf-mu_tr_j12n_j16n.txt     True   False  True")
    for elem in os.listdir(path):
        E1 = False
        should_it_have_E1 = False
        M1 = False
        should_it_have_M1 = False
        E2 = False
        should_it_have_E2 = False

        if elem.startswith("log_") and elem.endswith(".txt") and ("_tr_" in elem):
            tmp = elem[:-4]
            tmp = tmp.split("_")
            orbital_1_parity = tmp[-2][-1]
            orbital_2_parity = tmp[-1][-1]
            orbital_1_spin = int(tmp[-2][1:-1])/2
            orbital_2_spin = int(tmp[-1][1:-1])/2
            # spin_diff = abs(orbital_1_spin - orbital_2_spin)
            spin_range = list(range(
                int(abs(orbital_1_spin - orbital_2_spin)),
                int(orbital_1_spin + orbital_2_spin + 1),
                1
            ))

            if orbital_1_parity == orbital_2_parity:
                if (1 in spin_range) and (2 not in spin_range):
                    should_it_have_E1 = False
                    should_it_have_M1 = True
                    should_it_have_E2 = False
                elif (1 in spin_range) and (2 in spin_range):
                    should_it_have_E1 = False
                    should_it_have_M1 = True
                    should_it_have_E2 = True
                elif (1 not in spin_range) and (2 in spin_range):
                    should_it_have_E1 = False
                    should_it_have_M1 = False
                    should_it_have_E2 = True
                else:
                    msg = "No conditions met! (parity change)"
                    raise RuntimeError(msg)

            elif orbital_1_parity != orbital_2_parity:
                if (1 in spin_range) and (2 not in spin_range):
                    should_it_have_E1 = True
                    should_it_have_M1 = False
                    should_it_have_E2 = False
                elif (1 in spin_range) and (2 in spin_range):
                    should_it_have_E1 = True
                    should_it_have_M1 = False
                    should_it_have_E2 = False
                elif (1 not in spin_range) and (2 in spin_range):
                    """
                    This condition should never be met since KSHELL
                    will never calculate these transitions.
                    """
                    should_it_have_E1 = False
                    should_it_have_M1 = False
                    should_it_have_E2 = False
                    msg = "This condition should not ever be met!"
                    raise RuntimeError(msg)
                else:
                    msg = "No conditions met! (no parity change)"
                    raise RuntimeError(msg)
            else:
                msg = "No conditions met! (could not decide parity change)"
                raise RuntimeError(msg)

            with open(f"{path}/{elem}", "r") as infile:
                for line in infile:
                    if "E1 transition" in line:
                        E1 = True
                    elif "M1 transition" in line:
                        M1 = True
                    elif "E2 transition" in line:
                        E2 = True
                
                if (E1 == should_it_have_E1) and (M1 == should_it_have_M1) and (E2 == should_it_have_E2):
                    correct = True
                else:
                    correct = False
                print(f"{elem:40s}{str(E1):10s}{str(M1):10s}{str(E2):10s}{correct}")

def collect_logs(path: str=".", old_or_new: str="new"):
    """
    Collect energy and transition data from all log files in 'path'.

    Parameters
    ----------
    path : str
        Path to directory with log files, or parth to single energy log
        file, if you wanna do that for some reason.

    old_or_new : str
        Choose between old or new log file syntax. Pre or post
        2021-11-24. Summary file syntax is of new type, regardless.
        The option "both" is for the rare case of log files of mixed
        syntax in the same directory.

    Raises
    ------
    RuntimeError:
        If no energy log files are found in 'path'.

    ValueError:
        If 'old_or_new' is of invalid value.
    """
    allowed_old_or_new = ["new", "old", "both"]
    old_or_new = old_or_new.lower()
    if old_or_new not in allowed_old_or_new:
        msg = f"old_or_new must be in {allowed_old_or_new}. Got {old_or_new}."
        raise ValueError(msg)

    energy_log_files = []
    transit_log_files = []
    isotopes = []
    for elem in os.listdir(path):
        # if elem.startswith("log_") and elem.endswith(".txt"):
        if ("log_" in elem) and elem.endswith(".txt"):
            tmp = elem.split("_")
            isotope_index = tmp.index("log") + 1    # Isotope name is always after 'log'.
            # if (tmp := elem.split("_")[1].lower()) not in isotopes:
            if tmp[isotope_index] not in isotopes:
                isotopes.append(tmp[isotope_index])
            if not "_tr_" in elem:
                energy_log_files.append(f"{path}/{elem}")
            elif "_tr_" in elem:
                transit_log_files.append(f"{path}/{elem}")
    
    if len(isotopes) > 1:
        print(f"Log files for different isotopes have been found in {path}")
        msg = f"Found: {isotopes}. Your choice: "
        while True:
            choice = input(msg)
            if choice in isotopes:
                break
        
        # Remove all log files not of type 'choice'.
        energy_log_files = [i for i in energy_log_files if choice in i.split("/")[-1].lower()]
        transit_log_files = [i for i in transit_log_files if choice in i.split("/")[-1].lower()]

    if len(energy_log_files) == 0:
        msg = f"No energy log files in path '{path}'."
        raise FileNotFoundError(msg)

    if len(transit_log_files) == 0:
        msg = f"No transit log files in path '{path}', only energy log files."
        warnings.warn(msg, RuntimeWarning)

    E_data = {} # E_data[energy] = (log filename, spin, parity, eigenstate number, tt).
    spin_parity_occurrences = {}    # Count the occurrences of each (spin, parity) pair.
    multipole_types = ["E1", "M1", "E2"]
    
    print("Loading energy log files...")
    for i, log_file in enumerate(energy_log_files):
        progress_message = f"Reading file {i + 1:3d} of {len(energy_log_files)},"
        progress_message += f" '{log_file.split('/')[-1]}'"
        print(progress_message)
        read_energy_logfile(log_file, E_data)

    energies = E_data.keys()
    if len(energies) == 0:
        msg = "No energy data has been read from energy logs!"
        raise RuntimeError(msg)

    energies = sorted(energies)
    for energy in energies:
        """
        What does this loop actually do...?
        """
        filename, spin, parity, n_eig, tt = E_data[energy]
        spin_parity = (spin, parity)
        try:
            """
            Count the number of each (spin, parity) occurrence.
            """
            spin_parity_occurrences[spin_parity] += 1
        except KeyError:
            """
            Create initial value if key has not yet occurred.
            """
            spin_parity_occurrences[spin_parity] = 1
        n_jnp[ (spin, parity, n_eig) ] = spin_parity_occurrences[spin_parity]
        E_data[energy] = filename, spin, parity, spin_parity_occurrences[spin_parity], tt
    
    global E_gs
    E_gs = energies[0]

    counter = 0
    filename_without_path = energy_log_files[0].split("/")[-1]
    isotope = filename_without_path.split("_")[1]
    model_space = filename_without_path.split("_")[2]
    while True:
        """
        Create unique summary filename.
        """
        summary_filename = f"summary_{isotope}_{model_space}_{counter:03d}"
        if os.path.isfile(f"{path}/{summary_filename}.txt"):
            counter += 1
        else:
            summary_filename += ".txt"
            break
    
    with open(f"{path}/{summary_filename}", "w") as outfile:
        outfile.write("\n Energy levels\n")
        outfile.write('\n    N   J     prty N_Jp T        E(MeV)    Ex(MeV)  log-file\n\n')
        for i, energy in enumerate(energies):
            filename, spin, parity, n_eig, tt = E_data[energy]
            out = f"{i + 1:5d}   "
            out += f"{spin_to_string(spin):5s} "
            out += f"{parity:1s} "
            out += f"{n_eig:5d}   "
            out += f"{spin_to_string(tt):3s} "
            out += f"{energy:10.3f} "
            out += f"{energy - E_gs:10.3f}   "
            out += f"{filename}\n"
            outfile.write(out)
        outfile.write("\n")

        if len(transit_log_files) > 0:
            print("\nLoading transit log files...")
            for multipole_type in multipole_types:
                output_e = {}
                
                for i, filename in enumerate(transit_log_files):
                    progress_message = f"Reading file {i + 1:3d} of {len(transit_log_files)},"
                    progress_message += f" multipole: {multipole_type},"
                    progress_message += f" '{filename.split('/')[-1]}'"
                    print(progress_message)
                    if old_or_new == "new":
                        unit_weisskopf, out_e, mass = read_transit_logfile(filename, multipole_type)
                    elif old_or_new == "old":
                        unit_weisskopf, out_e, mass = read_transit_logfile_old(filename, multipole_type)
                    elif old_or_new == "both":
                        try:
                            unit_weisskopf, out_e, mass = read_transit_logfile_old(filename, multipole_type)
                        except ValueError:
                            """
                            If a ValueError is raised inside
                            read_transit_logfile_old, it is due to the
                            log file being new syntax styled.
                            """
                            unit_weisskopf, out_e, mass = read_transit_logfile(filename, multipole_type)
                    
                    if unit_weisskopf is None:
                        """
                        Incomplete and invalid logfile.
                        """
                        print(f"Incomplete logfile '{filename}'. Skipping...")
                        continue
                    
                    output_e.update(out_e)
                
                B_weisskopf, unit_weisskopf = weisskopf_unit(multipole_type, mass)
                outfile.write(f"B({multipole_type})  ( > {weisskopf_threshold:.1f} W.u.)  mass = {mass}    1 W.u. = {B_weisskopf:.1f} {unit_weisskopf}")
                outfile.write(f"\n{unit_weisskopf} (W.u.)")
                outfile.write(f"\nJ_i  pi_i idx_i Ex_i    J_f  pi_f idx_f Ex_f      dE         B({multipole_type})->         B({multipole_type})->[wu]     B({multipole_type})<-         B({multipole_type})<-[wu]\n")

                for _, out in sorted(output_e.items()):
                    outfile.write(out)
                outfile.write("\n\n")