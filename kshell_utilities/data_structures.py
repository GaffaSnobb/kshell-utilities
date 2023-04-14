from dataclasses import dataclass

@dataclass
class OrbitalParameters:
    """
    For storing parameters of the model space orbitals.
    """
    idx: int        # Index of the orbital.
    n: int
    l: int          # Orbital angular momentum.
    j: int          # Total angular momentum.
    tz: int         # Isospin.
    nucleon: str    # Proton or neutron
    name: str       # Ex: p 0d5/2.
    parity: int

    def __str__(self):
        return self.name
    
@dataclass
class OrbitalOrder:
    idx: int
    major_shell_idx: int
    major_shell_name: str
    
@dataclass
class ConfigurationParameters:
    idx: int
    parity: int
    configuration: list[int]