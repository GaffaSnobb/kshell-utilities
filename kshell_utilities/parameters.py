from __future__ import annotations
from .data_structures import OrbitalOrder

GS_FREE_PROTON = 5.585
GS_FREE_NEUTRON = -3.826
flags = {"debug": False, "parallel": True}

def debug_mode(switch):
    if isinstance(switch, bool):
        flags["debug"] = switch
    else:
        print(f"Invalid debug switch '{switch}'")

def latex_plot():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["roman"],
        "legend.fontsize": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "figure.titlesize": 15
    })

spectroscopic_conversion: dict[int, str] = {
    0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"
}

atomic_numbers = {
    "oxygen": 8, "fluorine": 9, "neon": 10, "sodium": 11, "magnesium": 12,
    "aluminium": 13, "silicon": 14, "phosphorus": 15, "sulfur": 16,
    "chlorine": 17, "argon": 18
}

atomic_numbers_reversed = {
    8: 'oxygen', 9: 'fluorine', 10: 'neon', 11: 'sodium', 12: 'magnesium',
    13: 'aluminium', 14: 'silicon', 15: 'phosphorus', 16: 'sulfur',
    17: 'chlorine', 18: 'argon'
}

elements = {
    'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9,
    'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16,
    'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 'v': 23,
    'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30,
    'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37,
    'sr': 38, 'y': 39, 'zr': 40, 'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44,
    'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48, 'in': 49, 'sn': 50, 'sb': 51,
    'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56, 'la': 57, 'ce': 58,
    'pr': 59, 'nd': 60, 'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64, 'tb': 65,
    'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70, 'lu': 71, 'hf': 72,
    'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79,
    'hg': 80, 'tl': 81, 'pb': 82, 'bi': 83, 'po': 84, 'at': 85, 'rn': 86,
    'fr': 87, 'ra': 88, 'ac': 89, 'th': 90, 'pa': 91, 'u': 92, 'np': 93,
    'pu': 94, 'am': 95, 'cm': 96, 'bk': 97, 'cf': 98, 'es': 99, 'fm': 100,
    'md': 101, 'no': 102, 'lr': 103, 'rf': 104, 'db': 105, 'sg': 106,
    'bh': 107, 'hs': 108, 'mt': 109, 'ds': 110, 'rg': 111, 'cn': 112,
    'nh': 113, 'fl': 114, 'mc': 115, 'lv': 116, 'ts': 117, 'og': 118
}

recommended_quenching_factors = {
    "GCLSTsdpfsdgix5pn.snt": f"0.75*GS_FREE = {round(0.75*GS_FREE_PROTON, 3), round(0.75*GS_FREE_NEUTRON, 3)}",
    "gs8.snt": f"0.75*GS_FREE = {round(0.75*GS_FREE_PROTON, 3), round(0.75*GS_FREE_NEUTRON, 3)}",
    "jun45.snt": f"0.7*GS_FREE = {round(0.7*GS_FREE_PROTON, 3), round(0.7*GS_FREE_NEUTRON, 3)}",
    "gxpf1a.snt": f"0.9*GS_FREE = {round(0.9*GS_FREE_PROTON, 3), round(0.9*GS_FREE_NEUTRON, 3)}",
    "gxpf1.snt": f"0.9*GS_FREE = {round(0.9*GS_FREE_PROTON, 3), round(0.9*GS_FREE_NEUTRON, 3)}",
    "sdpf-mu.snt": f"0.9*GS_FREE = {round(0.9*GS_FREE_PROTON, 3), round(0.9*GS_FREE_NEUTRON, 3)}",
    "sn100pn.snt": f"0.7*GS_FREE = {round(0.7*GS_FREE_PROTON, 3), round(0.7*GS_FREE_NEUTRON, 3)}"
}

# shell_model_order: dict[str, int] = {   # Standard shell order for spherical nuclei.
#     "0s1": 0,
#     "0p3": 1,
#     "0p1": 2,
#     "0d5": 3,
#     "1s1": 4,
#     "0d3": 5,
#     "0f7": 6,
#     "1p3": 7,
#     "0f5": 8,
#     "1p1": 9,
#     "0g9": 10,
#     "1d5": 11,
#     "0g7": 12,
#     "1d3": 13,
#     "2s1": 14,
# }

shell_model_order: dict[str, OrbitalOrder] = {   # Standard shell order for spherical nuclei.
    "0s1": OrbitalOrder(idx=0,  major_shell_idx=0, major_shell_name="s"),
    "0p3": OrbitalOrder(idx=1,  major_shell_idx=1, major_shell_name="p"),
    "0p1": OrbitalOrder(idx=2,  major_shell_idx=1, major_shell_name="p"),
    "0d5": OrbitalOrder(idx=3,  major_shell_idx=2, major_shell_name="sd"),
    "1s1": OrbitalOrder(idx=4,  major_shell_idx=2, major_shell_name="sd"),
    "0d3": OrbitalOrder(idx=5,  major_shell_idx=2, major_shell_name="sd"),
    "0f7": OrbitalOrder(idx=6,  major_shell_idx=3, major_shell_name="pf"),
    "1p3": OrbitalOrder(idx=7,  major_shell_idx=3, major_shell_name="pf"),
    "0f5": OrbitalOrder(idx=8,  major_shell_idx=3, major_shell_name="pf"),
    "1p1": OrbitalOrder(idx=9,  major_shell_idx=3, major_shell_name="pf"),
    "0g9": OrbitalOrder(idx=10, major_shell_idx=4, major_shell_name="sdg"),
    "1d5": OrbitalOrder(idx=11, major_shell_idx=4, major_shell_name="sdg"),
    "0g7": OrbitalOrder(idx=12, major_shell_idx=4, major_shell_name="sdg"),
    "1d3": OrbitalOrder(idx=13, major_shell_idx=4, major_shell_name="sdg"),
    "2s1": OrbitalOrder(idx=14, major_shell_idx=4, major_shell_name="sdg"),
}
major_shell_order: dict[str, int] = {
    "s"  : 0,
    "p"  : 1,
    "sd" : 2,
    "pf" : 3,
    "sdg": 4,
}
