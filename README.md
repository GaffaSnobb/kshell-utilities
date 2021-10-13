# kshell-utilities
Handy utilities for processing nuclear shell model data from `KSHELL`. Se the [KSHELL repository](https://github.com/GaffaSnobb/kshell) for installation and usage instructions for `KSHELL`.

## Installation
Install from the PyPi repository
``` bash
pip install kshell-utilities
```
Or, for the very latest version, clone this repository to your downloads directory. cd to the root directory of this repository and run
```
pip install .
```

## Usage

<details>
<summary>Load and view data from KSHELL</summary>
<p>

`KSHELL` summary files are easily read with:
``` python
import kshell_utilities as ksutil

ne20 = ksutil.loadtxt("summary_Ne20_usda.txt")[0]
```
`ne20` is an instance containing several useful attributes. To see the available attributes:
``` python
> print(ne20.help)
['BE2',
'BM1',
'Ex',
'help',
'level_plot',
'level_density_plot',
'levels',
'model_space',
'neutron_partition',
'nucleus',
'proton_partition',
'transitions',
'transitions_BE2',
'transitions_BM1',
'truncation']
```
To see the energy, 2\*spin and parity of each level:
``` python
> print(ne20.levels)
[[-40.467   0.      1.   ]
[-38.771   4.      1.   ]
[-36.376   8.      1.   ]
[-33.919   0.      1.   ]
[-32.882   4.      1.   ]
[-32.107  12.      1.   ]
...
[-25.978  12.      1.   ]
[-25.904  10.      1.   ]
[-25.834   8.      1.   ]
[-25.829   2.      1.   ]]
```
Slice the array to get only selected values, if needed (`ne20.levels[:, 0]` for only the energies). To see 2\*spin_initial, parity_initial, Ex_initial, 2\*spin_final, parity_final, Ex_final, E_gamma, B(.., i->f), B(.., f<-i)] for the M1 transitions:
``` python
> print(ne20.transitions_BM1)
[[4.0000e+00 1.0000e+00 1.6960e+00 ... 7.5850e+00 5.8890e+00 0.0000e+00]
[4.0000e+00 1.0000e+00 1.6960e+00 ... 9.9770e+00 8.2810e+00 4.8200e-01]
[4.0000e+00 1.0000e+00 7.5850e+00 ... 9.9770e+00 2.3920e+00 1.1040e+00]
...
[4.0000e+00 1.0000e+00 1.3971e+01 ... 1.4638e+01 6.6700e-01 6.0000e-03]
[0.0000e+00 1.0000e+00 1.4126e+01 ... 1.4638e+01 5.1200e-01 2.0000e-02]
[2.0000e+00 1.0000e+00 1.4336e+01 ... 1.4638e+01 3.0200e-01 0.0000e+00]]
```

</p>
</details>

<details>
<summary>Visualise data from KSHELL </summary>
<p>


You can easily create a level density plot by
``` python
ne20.level_density_plot(bin_size=1)
```
or by
``` python
ksutil.level_density(
    energy_levels = ne20.levels[:, 0],
    bin_size = 1,
    plot = True
)
```
or by
``` python
import matplotlib.pyplot as plt

bins, density = ksutil.level_density(
    energy_levels = ne20.levels[:, 0],
    bin_size = 1
)
plt.step(bins, density)
plt.show()
```
Choose an appropriate bin size. The two latter ways of generating the plot does not require that the data comes from `KSHELL`. Use any energy level data. The plot will look like this:

<details>
<summary>Click to see level density plot</summary>
<p>

![level_density_plot](https://github.com/GaffaSnobb/kshell-utilities/blob/main/doc/level_density_plot_ne20.png)

</p>
</details>

To generate a level plot:
``` python
ne20.level_plot()
```
or
``` python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ksutil.level_plot(
    levels = ne20.levels,
    ax = ax
)
plt.show()
```

<details>
<summary>Click to see level plot</summary>
<p>

![level_plot](https://github.com/GaffaSnobb/kshell-utilities/blob/main/doc/level_plot_ne20.png)

</p>
</details>

Both ways of generating the level plot supports selecting what spins to include in the plot, and how many levels per spin:
``` python
ne20.level_plot(
    max_spin_states = 3,
    filter_spins = [0, 3, 5]
)
```

<details>
<summary>Click to see filtered level plot</summary>
<p>

![filtered_level_plot](https://github.com/GaffaSnobb/kshell-utilities/blob/main/doc/level_plot_filtered_ne20.png)

</p>
</details>

The gamma strengh function (averaged over spins and parities) can easily be calculated by:
``` python
import matplotlib.pyplot as plt

bins, gsf = ksutil.gamma_strength_function_average(
    levels = ne20.levels,
    transitions = ne20.transitions_BM1,
    bin_width = 1,
    Ex_min = 0,
    Ex_max = 14,
    multipole_type = "M1"
)
plt.plot(bins, gsf)
plt.show()
```
where `bin_width`, `Ex_max` and `Ex_min` are in the same unit as the input energy levels, which from `KSHELL` is in MeV. `bin_width` is the width of the bins when the level density is calculated. `Ex_min` and `Ex_max` are the lower and upper limits for the excitation energy of the initial state of the transitions.

<details>
<summary>Click to see gamma strength function plot</summary>
<p>

![gsf_plot](https://github.com/GaffaSnobb/kshell-utilities/blob/main/doc/gsf_ne20.png)

</p>
</details>

</p>
</details>


## Credits
KSHELL is created by Noritaka Shimizu https://arxiv.org/abs/1310.5431. Code in this repository is built upon tools created by Jørgen Eriksson Midtbø: https://github.com/jorgenem/kshell_public.
