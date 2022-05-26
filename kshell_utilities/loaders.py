import time
from fractions import Fraction
import numpy as np
from .kshell_exceptions import KshellDataStructureError

def _parity_string_to_integer(parity: str):
    if parity == "+":
        res = 1
    elif parity == "-":
        res = -1
    else:
        msg = f"Invalid parity read from file. Got: '{parity}'."
        raise KshellDataStructureError(msg)

    return res

def _load_energy_levels(infile):
    """
    Load excitation energy, spin and parity into a list of structure:
    levels = [[energy, spin, parity], ...].
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

def _load_transition_probabilities_old(infile):
        """
        For summary files with old syntax (pre 2021-11-24).
        Parameters
        ----------
        infile:
            The KSHELL summary file.
        """
        reduced_transition_prob_decay_list = []
        negative_spin_counts = 0
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
                reduced_transition_prob_decay_list.append([
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
    
        return reduced_transition_prob_decay_list, negative_spin_counts
    
def _load_transition_probabilities(infile):
        """
        Example structure:
        B(E2)  ( > -0.0 W.u.)  mass = 50    1 W.u. = 10.9 e^2 fm^4
        e^2 fm^4 (W.u.)
        J_i  pi_i idx_i Ex_i    J_f  pi_f idx_f Ex_f      dE         B(E2)->         B(E2)->[wu]     B(E2)<-         B(E2)<-[wu]
        5    +    1     0.036   6    +    1     0.000     0.036     70.43477980      6.43689168     59.59865983      5.44660066
        4    +    1     0.074   6    +    1     0.000     0.074     47.20641983      4.31409897     32.68136758      2.98668391
        Parameters
        ----------
        infile:
            The KSHELL summary file.
        """
        reduced_transition_prob_decay_list = []
        negative_spin_counts = 0
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
            idx_final = int(line_split[2])
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
            reduced_transition_prob_decay_list.append([
                2*spin_initial, parity_initial, idx_initial, Ex_initial,
                2*spin_final, parity_final, idx_final, Ex_final, E_gamma,
                reduced_transition_prob_decay, reduced_transition_prob_excite
            ])
    
        return reduced_transition_prob_decay_list, negative_spin_counts

def _load_parallel(arg_list):
    """
    For parallel data loads.
    [self.fname_summary, "Energy", self._load_energy_levels, None]
    """
    fname, condition, loader, thread_idx = arg_list
    print(f"Thread {thread_idx} loading {condition} values...")
    load_time = time.perf_counter()
    
    with open(fname, "r") as infile:
        for line in infile:
            if condition in line:
                ans = loader(infile)
    
    load_time = time.perf_counter() - load_time
    print(f"Thread {thread_idx} finished loading {condition} values in {load_time:.2f} s")

    return ans

def _load_transition_probabilities_jem(infile, multipole_type="M1"):
    """
    Author : JEM
    Modified : GaffaSnobb
    https://github.com/jorgenem/kshell_public/blob/759d16dc44802be307a98b23dd984aaae2807b8d/bin/shellmodelutilities.py#L42
    JEM has modified the summary files from KSHELL with a slightly
    different syntax. This function reads that syntax. The output array
    has been modified to match kshell-utilities requirements, and the
    way infile is treated is changed to match the other loaders.
    """
    transitions = []
    # with open(infile, 'r') as f:
    #     lines = f.readlines()
    #     i_start = -1
    #     for i in range(len(lines)):
    #         if len(lines[i].split())<1: continue
    #         if lines[i].split()[0] == "B({:s})".format(multipole_type):
    #             # print "hello"
    #             i_start = i+2
    #             break
    for _ in range(2): infile.readline()
    for line in infile:
        line_split = line.split()
        if not line_split: break
        
        pi_str = line[25:27].strip()
        # print pi_str
        if pi_str == "+":
            pi = +1
        elif pi_str == "-":
            pi = -1
        else:
            raise Exception("From function read_transition_strengths: Could not assign initial parity. Read value: "+pi_str)
        pf_str = line[4:5].strip()
        if pf_str == "+":
            pf = +1
        elif pf_str == "-":
            pf = -1
        else:
            raise Exception("From function read_transition_strengths: Could not assign final parity Read value: "+pf_str)
        transitions.append([float(line[22:25]), pi, float(line[34:43]), float(line[0:3]), pi, float(line[12:22]), float(line[43:51]), float(line[67:83])])
    
    return np.asarray(transitions)