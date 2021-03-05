import time
from typing import Callable

import numpy as np
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import estimation_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import conversion
from matplotlib import pyplot as plt
from collections import namedtuple

spice_interface.load_standard_kernels()
DynamicsSim = namedtuple("DynamicsSim", ["raw_time", "state_history", "dependent_variable_history", "time_days"])

# util methods
def unpack2d(the_dict):
    return np.fromiter(the_dict.keys(), dtype=float), np.vstack(list(the_dict.values()))


# Propagation methods
def create_bodies():
    bodies_to_create = ["Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Sun"]

    global_frame_origin = "Sun"
    global_frame_orientation = "ECLIPJ2000"

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle object
    bodies.create_empty_body("Spacecraft")

    return bodies


def get_propagation_settings(initial_state, initial_time, bodies):
    bodies_to_propagate = ["Spacecraft"]
    central_bodies = ["Sun"]

    acceleration_settings_on_spacecraft = dict(
        **{
            body: [propagation_setup.acceleration.point_mass_gravity()]
            for body in ["Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Sun"]
        }
    )

    # Create global accelerations dictionary.
    acceleration_settings = {"Spacecraft": acceleration_settings_on_spacecraft}

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_distance("Spacecraft", "Earth"),
        propagation_setup.dependent_variable.relative_distance("Spacecraft", "Mars"),
        propagation_setup.dependent_variable.relative_position("Earth", "Sun"),
        propagation_setup.dependent_variable.relative_position("Mars", "Sun")
    ]

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    # Termination condition
    time_termination_settings = propagation_setup.propagator.time_termination(
        initial_time + constants.JULIAN_YEAR,
        terminate_exactly_on_final_condition=False
    )

    distance_termination_setting = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance("Spacecraft", "Sun"),
        limit_value=constants.ASTRONOMICAL_UNIT * 5,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )

    # prevent earth clinging
    earth_distance_termination_setting = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance("Spacecraft", "Earth"),
        limit_value=15e6,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )

    earth_time_termination_settings = propagation_setup.propagator.time_termination(
        initial_time + constants.JULIAN_DAY * 5,
        terminate_exactly_on_final_condition=False
    )

    earth_hybrid = propagation_setup.propagator.hybrid_termination([earth_distance_termination_setting, earth_time_termination_settings], fulfill_single_condition=False)

    # fuse them together
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination([time_termination_settings, distance_termination_setting, earth_hybrid],
                                                                                  fulfill_single_condition=True)

    # Create propagation settings with termination time.
    propagator_settings = propagation_setup.propagator.translational(
        ["Sun"],
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        hybrid_termination_settings,
        output_variables=dependent_variables_to_save
    )

    return propagator_settings


def get_initial_state_function(initial_time: float, delta_v_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray], parking_orbit_altitude: float = 1_000):
    # generate bodies
    bodies = create_bodies()
    mu_earth = bodies.get_body("Earth").gravitational_parameter

    # determine initial state in parking orbit around earth
    earth_initial_state = np.array(
        spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name="Earth",
            observer_body_name="Sun",
            reference_frame_name="ECLIPJ2000",
            aberration_corrections="NONE",
            ephemeris_time=initial_time)
    )

    parking_semi_major_axis = constants.EARTH_EQUATORIAL_RADIUS + parking_orbit_altitude * 1e3  # m

    distance_sun_earth = np.linalg.norm(earth_initial_state[:3])
    direction_sun_earth = earth_initial_state[:3] / distance_sun_earth

    earth_inertial_velocity = earth_initial_state[3:]

    initial_position = direction_sun_earth * (distance_sun_earth + parking_semi_major_axis)
    parking_orbit_velocity = earth_inertial_velocity / np.linalg.norm(earth_inertial_velocity) * (mu_earth / parking_semi_major_axis) ** 0.5

    total_inertial_velocity = earth_inertial_velocity + parking_orbit_velocity
    escape_velocity = (mu_earth * 2 / parking_semi_major_axis) ** 0.5

    initial_state = np.hstack((initial_position, total_inertial_velocity))

    def get_initial_state():
        initial_time_ = initial_state * 1
        delta_v = delta_v_func(parking_orbit_velocity, direction_sun_earth, escape_velocity)
        initial_time_[3:] += delta_v

        return initial_time_, delta_v

    return get_initial_state


def delta_v_func(parking_velocity: np.ndarray, sun_earth_dir: np.ndarray, escape_velocity: float):
    parking_velocity_dir = parking_velocity / np.linalg.norm(parking_velocity)

    min_v = escape_velocity - np.linalg.norm(parking_velocity)
    max_v = 7e3

    v_magn = (np.random.random() * (max_v - min_v) + min_v)

    vector_in_between = sun_earth_dir - parking_velocity_dir

    delta_v_dir = parking_velocity_dir + vector_in_between * np.random.random()
    delta_v_dir = np.random.random(3)
    delta_v_dir[2] = 0
    delta_v_dir /= np.linalg.norm(delta_v_dir)

    delta_v = delta_v_dir * v_magn

    return delta_v


def propagate_trajectory(propagator_settings, initial_time):

    integrator_settings = propagation_setup.integrator.bulirsch_stoer(
        initial_time,
        50,  # initial_time_step
        propagation_setup.integrator.bulirsch_stoer_sequence,
        4,  # max steps
        10,  # min step_size
        10_000,  # max step_size
        1e-9,  # rel. tolerance
        1e-9   # abs. tolerance
    )

    # Propagate forward/backward perturbed/unperturbed arc and save results to files
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(bodies, integrator_settings, propagator_settings, True, print_dependent_variable_data = False)

    time, state_history = unpack2d(dynamics_simulator.state_history)
    time, dependent_variable_history = unpack2d(dynamics_simulator.dependent_variable_history)

    return DynamicsSim(time, state_history, dependent_variable_history, (time - initial_time) / constants.JULIAN_DAY)


lines = []

if __name__ == "__main__":

    do_plotting: bool = False

    initial_time = 8304.5 * constants.JULIAN_DAY

    bodies = create_bodies()
    initial_state_function = get_initial_state_function(initial_time, delta_v_func)

    # Create numerical integrator settings
    while True:
        initial_state, delta_v = initial_state_function()

        # Get propagator settings for perturbed/unperturbed forwards/backwards arcs
        propagator_settings = get_propagation_settings(initial_state, initial_time, bodies)
        propagation_result = propagate_trajectory(propagator_settings, initial_time)

        mars_distance = propagation_result.dependent_variable_history[:, 1]
        final_index = mars_distance.argmin()
        final_distance = mars_distance[final_index]

        if do_plotting:
            print(f'{final_distance / constants.ASTRONOMICAL_UNIT:.2f}AU')

        if final_distance > 0.3 * constants.ASTRONOMICAL_UNIT:
            continue

        mars_position = propagation_result.dependent_variable_history[:, 5:7]
        mars_direction = mars_position[final_index] - mars_position[final_index - 1]
        mars_direction /= np.linalg.norm(mars_direction)

        final_distance *= (1 - 2 * ((mars_direction @ (propagation_result.state_history[final_index, :2] - mars_position[final_index])) < 0))

        if do_plotting:

            print()
            print(f'initial pos: {propagation_result.state_history[0, :3]}')
            plt.plot(*propagation_result.state_history[0, :2].T, 'x', c='b')
            plt.plot(*propagation_result.state_history[final_index, :2].T, 'x', c='r' if final_distance < 0 else 'g')

            spacecraft_position = propagation_result.state_history[:final_index + 3, :2]
            earth_position = propagation_result.dependent_variable_history[:final_index + 3, 2:4]
            mars_position = propagation_result.dependent_variable_history[:final_index + 3, 5:7]

            if not lines:
                lines.append(plt.plot(*spacecraft_position.T, rasterized=True, label='Spacecraft')[0])
                lines.append(plt.plot(*earth_position.T, rasterized=True, label='Earth')[0])
                lines.append(plt.plot(*mars_position.T, rasterized=True, label='Mars')[0])

                lines.append(plt.plot(*spacecraft_position[final_index].T, 'x', c='r' if final_distance < 0 else 'g')[0])
                lines.append(plt.plot(*mars_position[final_index].T, 'x', c='r' if final_distance < 0 else 'g')[0])

                plt.pause(0.0001)
            else:
                lines[0].set_data(*spacecraft_position.T)
                lines[1].set_data(*earth_position.T)
                lines[2].set_data(*mars_position.T)
                lines[3].set_data(*spacecraft_position[final_index].T)
                lines[4].set_data(*mars_position[final_index].T)

                plt.pause(0.0001)

        print(*delta_v, end=' ')
        print(final_distance / 1e3, end=' ')
        print(propagation_result.raw_time[final_index] - initial_time, end='\n')

