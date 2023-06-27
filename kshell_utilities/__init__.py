__version__ = "1.5.0.0"
__author__ = "Jon Kristian Dahl, Johannes Heines"
__credits__ = "Noritaka Shimizu, Jørgen Eriksson Midtbø"

"""
Version legend:
a.b.c.d

a: major release
b: new functionality added
c: new feature to existing functionality
d: bug fixes
"""

from .kshell_utilities import *
from .general_utilities import *
from .kshell_exceptions import *
from .count_dim import *
from .parameters import *
from .compare import Compare
from .collect_logs import collect_logs, check_multipolarities
from .script_editing import edit_and_queue_executables
from .partition_editor import partition_editor, test_partition_editor, test_partition_editor_2, load_interaction, load_partition
from . import data_structures
