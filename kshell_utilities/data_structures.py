from dataclasses import dataclass, field

@dataclass
class OrbitalOrder:
    """
    For storing information about the general classification of the
    shell model orbitals.
    """
    idx: int    # NOTE: Indices are in the "standard shell model order", not the order of the interaction file.
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
    orbitals: list[OrbitalParameters] = field(default_factory=list)
    major_shell_names: set[str] = field(default_factory=set)
    n_major_shells: int = 0
    n_orbitals: int = 0
    n_valence_nucleons: int = 0

@dataclass
class Interaction:
    model_space: ModelSpace = field(default_factory=ModelSpace)
    model_space_proton: ModelSpace = field(default_factory=ModelSpace)
    model_space_neutron: ModelSpace = field(default_factory=ModelSpace)
    name: str = ""
    n_core_protons: int = 0
    n_core_neutrons: int = 0
    n_spe: int = 0
    spe: list[float] = field(default_factory=list)  # Single-particle energies.
    n_tbme: int = 0
    # tbme: list[list[int | float]] = field(default_factory=list) # Two-body matrix elements.
    tbme: dict[tuple[int, int, int, int, int], float] = field(default_factory=dict) # Two-body matrix elements.
    fmd_mass: int = 0   # Dont know what this is yet.
    fmd_power: float = 0    # The exponent of fmd_mass.

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

    Properties
    ----------
    n_configurations : int
        The number of configurations.
    """
    configurations: list[Configuration] = field(default_factory=list)
    parity: int = 0
    n_existing_positive_configurations: int = 0
    n_existing_negative_configurations: int = 0
    n_new_positive_configurations: int = 0
    n_new_negative_configurations: int = 0
    ho_quanta_min_opposite_parity: int = +1000
    ho_quanta_max_opposite_parity: int = -1000
    ho_quanta_min_this_parity: int = +1000
    ho_quanta_max_this_parity: int = -1000
    ho_quanta_min: int = +1000
    ho_quanta_max: int = -1000

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