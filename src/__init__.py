from .main import main

# Álgebra lineal y optimización
from .introduction.intro_scipy import solve_linear
from .introduction.intro_scipy import get_matrix_properties
from .introduction.intro_scipy import find_min

# Estadística y análisis de datos
from .introduction.intro_numpy import get_statistics as get_statistics_numpy
from .introduction.intro_scipy import get_statistics as get_statistics_scipy

# Señales y espectros
from .introduction.intro_scipy import get_spectrum
from .introduction.intro_scipy import low_pass_filter

# Lectura y manipulación de archivos
from .introduction.intro_pandas import csv_registers
from .introduction.intro_pandas import json_registers
from .introduction.intro_pandas import yaml_registers
from .introduction.intro_pandas import get_head
from .introduction.intro_pandas import get_above
from .introduction.intro_pandas import group_and_average
from .introduction.intro_pandas import count_in_col
from .introduction.intro_pandas import export_data
from .introduction.intro_pandas import compare_dfs

# Operaciones con arrays y matrices
from .introduction.intro_numpy import ten_zeros_array
from .introduction.intro_numpy import floats_array
from .introduction.intro_numpy import invert_array
from .introduction.intro_numpy import square_matrix
from .introduction.intro_numpy import find_upper_five
from .introduction.intro_numpy import identity_matrix
from .introduction.intro_numpy import multiply_matrices
from .introduction.intro_numpy import normalize
from .introduction.intro_numpy import count_in_range