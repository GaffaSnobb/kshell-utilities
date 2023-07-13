import os
from vum import Vum
from .data_structures import (
    OrbitalParameters, Partition, Interaction, Configuration
)

def _sanity_checks(
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
    interaction: Interaction,
):
    """
    A few different sanity checks to make sure that the new
    configurations are physical.
    """
    for i, configuration in enumerate(partition_proton.configurations):
        assert len(configuration.configuration) == interaction.model_space_proton.n_orbitals
        assert sum(configuration.configuration) == interaction.model_space_proton.n_valence_nucleons
        assert sum([n*orb.ho_quanta for n, orb in zip(configuration.configuration, interaction.model_space_proton.orbitals)]) == configuration.ho_quanta

        for j in range(i+1, partition_proton.n_configurations):
            assert partition_proton.configurations[i].configuration != partition_proton.configurations[j].configuration, f"Duplicate proton configs {i} and {j}!"

        for orbital, occupation in zip(interaction.model_space_proton.orbitals, configuration.configuration):
            """
            Check that max degeneracy is respected.
            """
            assert occupation <= (orbital.j + 1), "Occupation should never be lager than the max degeneracy!"

    assert (i + 1) == partition_proton.n_configurations

    for i, configuration in enumerate(partition_neutron.configurations):
        assert len(configuration.configuration) == interaction.model_space_neutron.n_orbitals
        assert sum(configuration.configuration) == interaction.model_space_neutron.n_valence_nucleons
        assert sum([n*orb.ho_quanta for n, orb in zip(configuration.configuration, interaction.model_space_neutron.orbitals)]) == configuration.ho_quanta

        for j in range(i+1, partition_neutron.n_configurations):
            assert partition_neutron.configurations[i].configuration != partition_neutron.configurations[j].configuration, f"Duplicate neutron configs {i} and {j}!"

        for orbital, occupation in zip(interaction.model_space_neutron.orbitals, configuration.configuration):
            """
            Check that max degeneracy is respected.
            """
            assert occupation <= (orbital.j + 1), "Occupation should never be lager than the max degeneracy!"

    assert (i + 1) == partition_neutron.n_configurations

    for configuration in partition_combined.configurations:
        p_idx, n_idx = configuration.configuration
        
        assert partition_combined.parity == partition_proton.configurations[p_idx].parity*partition_neutron.configurations[n_idx].parity
        assert configuration.ho_quanta == partition_proton.configurations[p_idx].ho_quanta + partition_neutron.configurations[n_idx].ho_quanta

def _calculate_configuration_parity(
    configuration: list[int],
    model_space: list[OrbitalParameters]
) -> int:
    """
    Calculate the parity of a configuration.

    Parameters
    ----------
    configuration : list[int]
        The configuration to calculate the parity of.
    
    model_space : list[OrbitalParameters]
        The model space orbitals to use for the calculation.
    """
    if not configuration:
        msg = "Configuration is empty! Undefined behaviour."
        raise ValueError(msg)

    parity: int = 1
    for i in range(len(configuration)):
        if not configuration[i]: continue   # Empty orbitals do not count towards the total parity.
        parity *= model_space[i].parity**configuration[i]

    return parity

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
        
        while True:
            vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, "Several partition files detected.")
            vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, partition_choices)
            vum.screen.refresh()
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
            
            if is_compare_mode and (len(filename_partition) < 2): continue
            break
    
    vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, vum.blank_line)
    vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, vum.blank_line)
    vum.screen.refresh()

    return filename_interaction, filename_partition

# def is_configuration_below_energy_threshold(
#     interaction: Interaction,
#     threshold_energy: float,
#     proton_configuration: Configuration,
#     neutron_configuration: Configuration,
# ) -> bool:
#     """
#     Calculate the energy of a pn configuration. Return True if the
#     energy is lower than the threshold energy and False otherwise. Used
#     for monopole truncation.
#     """
#     energy = configuration_energy(
#         interaction = interaction,
#         proton_configuration = proton_configuration,
#         neutron_configuration = neutron_configuration,
#     )

#     return energy < threshold_energy

def configuration_energy(
    interaction: Interaction,
    proton_configuration: Configuration,
    neutron_configuration: Configuration,
) -> float:
    """
    Code from https://github.com/GaffaSnobb/kshell/blob/6e6edd6ac7ae70513b4bdaa94099cd3a3e32351d/bin/espe.py#L164
    """
    combined_configuration = proton_configuration.configuration + neutron_configuration.configuration
    if interaction.fmd_mass == 0:
        fmd = 1
    else:
        mass = interaction.n_core_neutrons + interaction.n_core_protons
        mass += interaction.model_space.n_valence_nucleons
        fmd = (mass/interaction.fmd_mass)**interaction.fmd_power

    assert len(interaction.spe) == len(combined_configuration)

    energy: float = 0.0
    
    for spe, occupation in zip(interaction.spe, combined_configuration, strict=True):
        energy += occupation*spe

    for i0 in range(len(combined_configuration)):
        for i1 in range(len(combined_configuration)):
            if i0 < i1: continue    # Why?
            
            elif i0 == i1:
                energy += combined_configuration[i0]*(combined_configuration[i0] - 1)*0.5*interaction.vm[i0, i0]*fmd
            else:
                energy += combined_configuration[i0]*combined_configuration[i1]*interaction.vm[i0, i1]*fmd

    return energy