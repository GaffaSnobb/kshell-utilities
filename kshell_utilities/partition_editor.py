import time, os, curses
from typing import Callable
from .data_structures import OrbitalParameters
from .parameters import spectroscopic_conversion
from .vum import Vum

shell_model_order: dict[str, int] = {   # Standard shell order for spherical nuclei.
    "0s1": 0,
    "0p3": 1,
    "0p1": 2,
    "0d5": 3,
    "1s1": 4,
    "0d3": 5,
    "0f7": 6,
    "1p3": 7,
    "0f5": 8,
    "1p1": 9,
    "0g9": 10,
    "1d5": 11,
    "0g7": 12,
    "1d3": 13,
    "2s1": 14,
}

def draw_shell_map(
    vum: Vum,
    model_space: list[OrbitalParameters],
    is_proton: bool,
    is_neutron: bool,
    occupation: tuple[int, int] | None = None,
):
    """
    Draw a simple map of the model space orbitals of the current
    interaction file. Sort the orbitals based on the shell_model_order
    dict.
    """
    y_offset: int = 6
    model_space_copy = sorted(  # Sort the orbitals based on the shell_model_order dict.
        model_space,
        key = lambda orbital: shell_model_order[f"{orbital.n}{spectroscopic_conversion[orbital.l]}{orbital.j}"],
        reverse = True
    )
    if is_proton:
        model_space_proton = [orbital for orbital in model_space_copy if orbital.tz == -1]
        max_proton_j: int = max([orbital.j for orbital in model_space_proton])
        
        if occupation is None:
            """
            Draw the entire map with no occupation.
            """
            for i in range(len(model_space_proton)):
                string = (
                    f"{model_space_proton[i].idx + 1:2d}"
                    f" {model_space_proton[i].name} " +
                    " "*(max_proton_j - model_space_proton[i].j) + "-" + " -"*(model_space_proton[i].j + 1)
                )
                vum.addstr(i + y_offset, 0, string)
        else:
            """
            Re-draw a single orbital with the user inputted occupation.
            """
            location: int = 0
            for orbital in model_space_proton:
                if orbital.idx == occupation[0]: break
                location += 1
            else:
                msg = (
                    f"Orbital index {occupation[0]} not found in the"
                    " current model space!"
                )
                raise RuntimeError(msg)
            
            string = (
                f"{orbital.idx + 1:2d}"
                f" {orbital.name} " +
                " "*(max_proton_j - orbital.j) + "-"
            )
            string += "o-"*occupation[1] + " -"*(orbital.j + 1 - occupation[1])
            vum.addstr(location + y_offset, 0, string)

    else:
        max_proton_j: int = 0    # Used for placing the neutron map so it must be defined.

    if is_neutron:
        model_space_neutron = [orbital for orbital in model_space_copy if orbital.tz == 1]
        max_neutron_j: int = max([orbital.j for orbital in model_space_neutron])

        for i in range(len(model_space_neutron)):        
            string = (
                f"{model_space_neutron[i].idx + 1:2d}"
                f" {model_space_neutron[i].name} " +
                " "*(max_neutron_j - model_space_neutron[i].j) + "-" + " -"*(model_space_neutron[i].j + 1)
            )
            vum.addstr(i + y_offset, max_proton_j + 26, string, is_blank_line=False)

def partition_editor(
    filename_interaction: str | None = None,
    filename_partition: str | None = None,
    filename_partition_edited: str | None = None,
    input_wrapper: Callable | None = None,
    is_interactive: bool = True,
):  
    """
    Wrapper for error handling.
    """
    try:
        msg = _partition_editor(
            filename_interaction = filename_interaction,
            filename_partition = filename_partition,
            filename_partition_edited = filename_partition_edited,
            input_wrapper = input_wrapper,
            is_interactive = is_interactive
        )
    except KeyboardInterrupt:
        curses.endwin()

    except curses.error as e:
        curses.endwin()
        raise(e)
    
    except Exception as e:
        curses.endwin()
        raise(e)

    else:
        curses.endwin()
        print(msg)

def _partition_editor(
    filename_interaction: str | None = None,
    filename_partition: str | None = None,
    filename_partition_edited: str | None = None,
    input_wrapper: Callable | None = None,
    is_interactive: bool = True,
):
    """
    Extract the model space orbitals from an interaction file. Extract
    proton and neutron partitions from an accompanying partition file.
    Prompt the user for new proton and neutron configurations and 
    regenerate the proton-neutron partition based on the existing and
    new proton and neutron configurations.

    Here, 'configuration' refers to a specific occupation of protons or
    neutrons in the model space orbitals while 'partition' refers to the
    set of all the configurations. Hence, a 'proton configuration' is
    one specific configuration, like
    ```
    1     6  4  2  0  2  4  2  0  0  0  0  0
    ```
    while the 'proton partition' is the set of all the proton
    configurations, like
    ```
    1     6  4  2  0  2  4  2  0  0  0  0  0
    2     6  4  2  0  3  3  2  0  0  0  0  0
    3     6  4  2  0  3  4  1  0  0  0  0  0
    ...
    ```

    Parameters
    ----------
    input_wrapper : Callable
        Defaults to `input` which asks the user for input. This wrapper
        exists so that unit tests can be performed in which case
        `input_wrapper = input_wrapper_test`.
    """
    vum = Vum()
    screen = vum.screen
    if input_wrapper is None:
        input_wrapper = vum.input

    if is_interactive:

        filenames_interaction = sorted([i for i in os.listdir() if i.endswith(".snt")])
        filenames_partition = sorted([i for i in os.listdir() if i.endswith(".ptn")])
        
        if not filenames_interaction:
            return f"No interaction file present in {os.getcwd()}. Exiting..."
        if not filenames_partition:
            return f"No partition file present in {os.getcwd()}. Exiting..."

        if len(filenames_interaction) == 1:
            filename_interaction = filenames_interaction[0]
            screen.addstr(0, 0, f"{filename_interaction} chosen")
            screen.refresh()

        elif len(filenames_interaction) > 1:
            interaction_choices: str = ""
            for i in range(len(filenames_interaction)):
                interaction_choices += f"{filenames_interaction[i]} ({i}), "
            
            screen.addstr(0, 0, "Several interaction files detected.")
            screen.addstr(1, 0, interaction_choices)
            screen.refresh()
            
            while True:
                ans = vum.input("Several interaction files detected. Please make a choice")
                try:
                    ans = int(ans)
                except ValueError:
                    continue
                
                try:
                    filename_interaction = filenames_interaction[ans]
                except IndexError:
                    continue

                break

        screen.addstr(0, 0, vum.blank_line)
        screen.addstr(1, 0, vum.blank_line)
        screen.refresh()
        
        if len(filenames_partition) == 1:
            filename_partition = filenames_partition[0]
            screen.addstr(0, 0, f"{filename_partition} chosen")
            screen.refresh()

        elif len(filenames_partition) > 1:
            partition_choices: str = ""
            for i in range(len(filenames_partition)):
                partition_choices += f"{filenames_partition[i]} ({i}), "
            
            screen.addstr(0, 0, "Several partition files detected.")
            screen.addstr(1, 0, partition_choices)
            screen.refresh()
            
            while True:
                ans = vum.input("Several partition files detected. Please make a choice")
                try:
                    ans = int(ans)
                except ValueError:
                    continue
                
                try:
                    filename_partition = filenames_partition[ans]
                except IndexError:
                    continue

                break
        
        screen.addstr(0, 0, vum.blank_line)
        screen.addstr(1, 0, vum.blank_line)
        screen.refresh()

    header: str = ""
    proton_configurations: list[str] = []
    neutron_configurations: list[str] = []
    model_space: list[OrbitalParameters] = []

    if filename_partition_edited is None:
        filename_partition_edited = f"{filename_partition.split('.')[0]}_edited.ptn"

    with open(filename_interaction, "r") as infile:
        """
        Extract information from the interaction file about the orbitals
        in the model space.
        """
        for line in infile:
            """
            Example:
            ! 2015/07/13
            ! from GCLSTsdpfsdgix5pn.snt 
            !  8 0g9/2  -0.8MeV
            !  9 0g7/2  -3.2MeV
            ! 10 1d5/2  -2.5MeV
            ! 11 1d3/2  -5.7MeV
            ! 12 2s1/2  -4.0MeV
            !
            ! sdpfsdg_64.sps
            ! sdpfsdg-shell
            ! index,    n,  l,  j, tz
            12  12   8   8
            ...
            """
            if line[0] != "!":
                tmp = line.split()
                n_proton_orbitals = int(tmp[0])
                n_neutron_orbitals = int(tmp[1])
                n_core_protons = int(tmp[2])
                n_core_neutrons = int(tmp[3])
                break

        for line in infile:
            if line[0] == "!": break
            idx, n, l, j, tz = [int(i) for i in line.split("!")[0].split()]
            idx -= 1
            nucleon = "p" if tz == -1 else "n"
            model_space.append(OrbitalParameters(
                idx = idx,
                n = n,
                l = l,
                j = j,
                tz = tz,
                nucleon = nucleon,
                name = f"{nucleon} {n}{spectroscopic_conversion[l]}{j}/2",
            ))

    assert all(orb.idx == i for i, orb in enumerate(model_space))   # Make sure that the list indices are the same as the orbit indices.

    draw_shell_map(vum=vum, model_space=model_space, is_proton=True, is_neutron=False)

    with open(filename_partition, "r") as infile:
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
                    n_protons, n_neutrons, parity = tmp
                except ValueError:
                    """
                    For example:
                    86 4
                    """
                    n_proton_configurations, n_neutron_configurations = tmp
                    infile.readline()   # Skip header.
                    break

            header += line

        for line in infile:
            """
            Extract proton partitions.
            """
            if "# neutron partition" in line: break

            proton_configurations.append(line)

        for line in infile:
            """
            Extract neutron partitions.
            """
            if "# partition of proton and neutron" in line: break

            neutron_configurations.append(line)

    new_proton_configurations: list[list[int]] = []
    new_neutron_configurations: list[list[int]] = []

    if input_wrapper("Add new proton configuration? (y/n): ") == "y":
        while True:
            occupation = _prompt_user_for_occupation(
                vum = vum,
                nucleon = "proton",
                model_space = model_space[:n_proton_orbitals],
                n_valence_nucleons = n_protons,
                input_wrapper = input_wrapper,
            )
            if occupation:
                new_proton_configurations.append(occupation)
                
                if input_wrapper("Add another proton configuration? (y/n): ") == "y":
                    continue
                else:
                    break

            elif occupation is None:
                """
                Quit signal. Do not keep the current configuration,
                but keep earlier defined new configurations and quit
                the prompt.
                """
                break

    if input_wrapper("Add new neutron configuration? (y/n): ") == "y":
        while True:
            occupation = _prompt_user_for_occupation(
                vum = vum,
                nucleon = "neutron",
                model_space = model_space[n_neutron_orbitals:],
                n_valence_nucleons = n_neutrons,
                input_wrapper = input_wrapper,
            )
            if occupation:
                new_neutron_configurations.append(occupation)
                if input_wrapper("Add another neutron configuration? (y/n): ") == "y":
                    continue
                else:
                    break

            elif occupation is None:
                """
                Quit signal. Do not keep the current configuration,
                but keep earlier defined new configurations and quit
                the prompt.
                """
                break

    n_new_proton_configurations = len(new_proton_configurations)
    n_new_neutron_configurations = len(new_neutron_configurations)
    n_total_proton_configurations = n_new_proton_configurations + n_proton_configurations
    n_total_neutron_configurations = n_new_neutron_configurations + n_neutron_configurations

    with open(filename_partition_edited, "w") as outfile:
        """
        Write edited data to new partition file.
        """
        outfile.write(header)
        outfile.write(f" {n_total_proton_configurations} {n_total_neutron_configurations}\n")
        outfile.write("# proton partition\n")
        
        for configuration in proton_configurations:
            outfile.write(configuration)

        for i in range(n_new_proton_configurations):
            outfile.write(
                f"{i + n_proton_configurations + 1:6d}     "    # +1 because .ptn indices start at 1.
                f"{str(new_proton_configurations[i]).strip('[]').replace(',', ' ')}"
                "\n"
            )
        outfile.write("# neutron partition\n")

        for configuration in neutron_configurations:
            outfile.write(configuration)

        for i in range(n_new_neutron_configurations):
            outfile.write(
                f"{i + n_neutron_configurations + 1:6d}     "    # +1 because .ptn indices start at 1.
                f"{str(new_neutron_configurations[i]).strip('[]').replace(',', ' ')}"
                "\n"
            )
        outfile.write("# partition of proton and neutron\n")
        outfile.write(f"{n_total_proton_configurations*n_total_neutron_configurations}\n")

        for p_idx in range(n_total_proton_configurations):
            """
            Write the proton-neutron configurations to file. This is
            really just pairing up all proton configuration indices
            with all neutron configuration indices.
            """
            for n_idx in range(n_total_neutron_configurations):
                outfile.write(f"{p_idx + 1:5d}{n_idx + 1:6d}\n")    # The .ptn indices start at 1.

def _prompt_user_for_occupation(
    vum: Vum,
    nucleon: str,
    model_space: list[OrbitalParameters],
    n_valence_nucleons: int,
    input_wrapper: Callable,
) -> list | None:
        
        if nucleon == "proton":
            is_proton = True
            is_neutron = False

        elif nucleon == "neutron":
            is_proton = False
            is_neutron = True

        else:
            msg = f"Invalid nucleon parameter '{nucleon}'!"
            raise ValueError(msg)

        model_space_copy = sorted(  # Sort the orbitals based on the shell_model_order dict.
            model_space,
            key = lambda orbital: shell_model_order[f"{orbital.n}{spectroscopic_conversion[orbital.l]}{orbital.j}"],
        )
        n_remaining_nucleons: int = n_valence_nucleons
        vum.addstr(0, 0, f"Please enter {nucleon} orbital occupation (f to fill, q to quit):")
        occupation: list[tuple[int, int]] = []
        
        for orbital in model_space_copy:
            if n_remaining_nucleons == 0:
                """
                If there are no more valence nucleons to use, set the
                remaining occupations to 0.
                """
                occupation.append((orbital.idx, 0))
                vum.addstr(1, 0, "Occupation of remaining orbitals set to 0.")
                continue

            while True:
                ans = input_wrapper(f"{orbital.idx + 1:2d} {orbital} (remaining: {n_remaining_nucleons}): ")
                if (ans == "q") or (ans == "quit") or (ans == "exit"): return None
                if ans == "f": ans = orbital.j + 1  # Fill the orbital.
                try:
                    ans = int(ans)
                except ValueError:
                    continue

                if (ans > (orbital.j + 1)) or (ans < 0):
                    vum.addstr(1, 0, f"Allowed occupation for this orbital is [0, 1, ..., {orbital.j + 1}]")
                    continue
                
                n_remaining_nucleons -= ans
                if ans > 0:
                    """
                    No point in re-drawing the line if the occupation is
                    unchanged.
                    """
                    draw_shell_map(
                        vum = vum,
                        model_space = model_space,
                        is_proton = is_proton,
                        is_neutron = is_neutron,
                        occupation = (orbital.idx, ans)
                    )
                break

            occupation.append((orbital.idx, ans))

            cum_occupation = sum(tup[1] for tup in occupation)
            if cum_occupation > n_valence_nucleons:
                vum.addstr(1, 0, f"INVALID: Total occupation ({cum_occupation}) exceeds the number of valence {nucleon}s ({n_valence_nucleons})")
                draw_shell_map(
                    vum = vum,
                    model_space = model_space,
                    is_proton = is_proton,
                    is_neutron = is_neutron,
                )
                return []
        
        cum_occupation = sum(tup[1] for tup in occupation)
        if cum_occupation < n_valence_nucleons:
            vum.addstr(1, 0, f"INVALID: Total occupation ({cum_occupation}) does not use the total number of valence {nucleon}s ({n_valence_nucleons})")
            draw_shell_map(
                vum = vum,
                model_space = model_space,
                is_proton = is_proton,
                is_neutron = is_neutron,
            )
            return []

        occupation.sort(key=lambda tup: tup[0]) # Sort list of tuples based on the orbital.idx.
        return [tup[1] for tup in occupation]   # Return only the occupation numbers now sorted based on the orbital order of the .snt file.