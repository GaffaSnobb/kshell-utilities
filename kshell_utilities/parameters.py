GS_FREE_PROTON = 5.585
GS_FREE_NEUTRON = -3.826
flags = {"debug": False}

def debug_mode(switch):
    if isinstance(switch, bool):
        flags["debug"] = switch
    else:
        print(f"Invalid debug switch '{switch}'")

def latex_plot():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "backend": "pgf",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["roman"],
        "legend.fontsize": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
    })

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