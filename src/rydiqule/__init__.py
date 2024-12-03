'''
Parent computational module.
'''

from .experiments import get_transmission_coef, get_susceptibility, get_phase_shift, get_solution_element, get_snr, get_OD

from .cell import Cell
from .sensor import Sensor
from .sensor_utils import generate_eom, get_basis_transform, get_rho_ij, get_rho_populations, scale_dipole, draw_diagram
from .atom_utils import D1_states, D2_states, calc_kappa, calc_eta
from .solvers import solve_steady_state
from .timesolvers import solve_time, solve_eom_stack, generate_eom_time
from .doppler_utils import get_doppler_equations, generate_doppler_shift_eom, doppler_classes, doppler_mesh, apply_doppler_weights
from .rydiqule_utils import about

from .slicing.slicing import compute_grid, matrix_slice, memory_size, get_slice_num, get_slice_num_t

__version__ = '1.2.3'
