from dataclasses import dataclass

@dataclass
class OrbitalOrder:
    """
    For storing information about the general classification of the
    shell model orbitals.
    """
    idx: int    # NOTE: Indices are in the "standard shell model order", not the order of the interaction file.
    # idx_snt: None | int # Index of the orbital as listed in the interaction file.
    major_shell_idx: int
    major_shell_name: str

@dataclass
class OrbitalParameters:
    """
    For storing parameters of the model space orbitals.
    """
    idx: int        # Index of the orbital from the interaction file.
    n: int          # The "principal quantum number".
    l: int          # Orbital angular momentum.
    j: int          # Total angular momentum.
    tz: int         # Isospin.
    nucleon: str    # 'p' or 'n'.
    name: str       # Ex: 'p 0d5/2'.
    parity: int
    order: OrbitalOrder
    ho_quanta: int  # Harmonic oscillator quanta (2*n + l) of the orbital (little bit unsure if this is a good name). Used to limit the possible combinations of pn configuration combinations.

    def __str__(self):
        return self.name

@dataclass
class ModelSpace:
    orbitals: list[OrbitalParameters]
    n_major_shells: int
    major_shell_names: set[str]
    n_orbitals: int
    n_valence_nucleons: int
    # n_proton_orbitals: int
    # n_neutron_orbitals: int

@dataclass
class Interaction:
    model_space: ModelSpace
    model_space_proton: ModelSpace
    model_space_neutron: ModelSpace
    name: str
    n_core_protons: int
    n_core_neutrons: int

@dataclass
class Configuration:
    """
    Terminology:
        - "Occupation" refers to the number of particles occupying one
        orbital. Ex: 1 is an occupation.
        
        - "Configuration" refers to a set of orbitals with a given
        occupation. Ex: [0, 1, 1, 0, 0, 0] is a configuration.

        - "Partition" refers to a set of configurations with a given
        number of particles. Ex: [[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]]
        is a partition.

    Parameters
    ----------    
    parity : int
        The parity of the configuration.

    configuration : list[int]
        The configuration as a list of occupations. Ex: [0, 1, 1, 0, 0, 0].

    ho_quanta : int
        The sum of the harmonic oscillator quanta of all the particles
        in the configuration. Used to limit the possible combinations of
        pn configuration combinations when using hw truncation.
    """
    parity: int
    configuration: list[int]
    ho_quanta: int

@dataclass
class Partition:
    """
    Parameters
    ----------
    parity : int
        The parity of the partition file.

    n_configurations : int
        The number of configurations.
    """
    parity: int
    # n_existing_configurations: int
    # existing_configurations: list[Configuration]
    # new_configurations: list[Configuration]
    configurations: list[Configuration] 
    n_existing_positive_configurations: int
    n_existing_negative_configurations: int
    n_new_positive_configurations: int
    n_new_negative_configurations: int
    ho_quanta_min_opposite_parity: int
    ho_quanta_max_opposite_parity: int
    ho_quanta_min_this_parity: int
    ho_quanta_max_this_parity: int
    ho_quanta_min: int
    ho_quanta_max: int

    @property
    def n_configurations(self) -> int:
        expected = (
            self.n_existing_negative_configurations + self.n_existing_positive_configurations +
            self.n_new_negative_configurations + self.n_new_positive_configurations
        )
        calculated = len(self.configurations)
        assert expected == calculated
        return calculated
    
    @property
    def n_new_configurations(self) -> int:
        return self.n_new_negative_configurations + self.n_new_positive_configurations
    
    @property
    def n_existing_configurations(self) -> int:
        return self.n_existing_negative_configurations + self.n_existing_positive_configurations
    
    def clear(self):
        self.configurations.clear()
        self.n_existing_positive_configurations = 0
        self.n_existing_negative_configurations = 0
        self.n_new_positive_configurations = 0
        self.n_new_negative_configurations = 0
        self.ho_quanta_min_this_parity = +1000
        self.ho_quanta_max_this_parity = -1000