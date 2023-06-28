import os
from vum import Vum

def _prompt_user_for_interaction_and_partition(
    vum: Vum,
    is_compare_mode: bool,
):  
    """
    Parameterss
    -----------
    is_compare_mode : bool
        User will be prompted for two partition files and they will be
        returned as a list of the two names.
    """
    filenames_interaction = sorted([i for i in os.listdir() if i.endswith(".snt")])
    filenames_partition = sorted([i for i in os.listdir() if i.endswith(".ptn")])
    
    if not filenames_interaction:
        return f"No interaction file present in {os.getcwd()}. Exiting..."
    if not filenames_partition:
        return f"No partition file present in {os.getcwd()}. Exiting..."

    if len(filenames_interaction) == 1:
        filename_interaction = filenames_interaction[0]
        vum.screen.addstr(0, 0, f"{filename_interaction} chosen")
        vum.screen.refresh()

    elif len(filenames_interaction) > 1:
        interaction_choices: str = ""
        for i in range(len(filenames_interaction)):
            interaction_choices += f"{filenames_interaction[i]} ({i}), "
        
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, "Several interaction files detected.")
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, interaction_choices)
        vum.screen.refresh()
        
        while True:
            ans = vum.input("Several interaction files detected. Please make a choice")
            if ans == "q": return False
            try:
                ans = int(ans)
            except ValueError:
                continue
            
            try:
                filename_interaction = filenames_interaction[ans]
            except IndexError:
                continue

            break

    vum.screen.addstr(0, 0, vum.blank_line)
    vum.screen.addstr(1, 0, vum.blank_line)
    vum.screen.refresh()
    
    if len(filenames_partition) == 1:
        if is_compare_mode: return "Cannot compare, only one partition file found!"
        filename_partition = filenames_partition[0]
        vum.screen.addstr(0, 0, f"{filename_partition} chosen")
        vum.screen.refresh()

    elif len(filenames_partition) > 1:
        if is_compare_mode: filename_partition: list[str] = []
        partition_choices: str = ""
        for i in range(len(filenames_partition)):
            partition_choices += f"{filenames_partition[i]} ({i}), "
        
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, "Several partition files detected.")
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, partition_choices)
        vum.screen.refresh()
        
        while True:
            ans = vum.input("Several partition files detected. Please make a choice")
            if ans == "q": return False
            try:
                ans = int(ans)
            except ValueError:
                continue
            
            try:
                if is_compare_mode: filename_partition.append(filenames_partition[ans])
                else: filename_partition = filenames_partition[ans]
            except IndexError:
                continue
            
            if is_compare_mode and (len(filename_interaction) < 2): continue
            break
    
    vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, vum.blank_line)
    vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, vum.blank_line)
    vum.screen.refresh()

    return filename_interaction, filename_partition