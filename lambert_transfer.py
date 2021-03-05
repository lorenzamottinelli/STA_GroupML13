from tudatpy.kernel import constants
from tudatpy.kernel.astro import two_body_dynamics, conversion
from tudatpy.kernel.astro.ephemerides import KeplerEphemeris
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup

import matplotlib.pyplot as plt
import numpy as np

spice_interface.load_standard_kernels()

global_frame_origin = "Sun"
global_frame_orientation = "ECLIPJ2000"
ephemeris_kwargs = dict(observer_body_name=global_frame_origin,
                        reference_frame_name=global_frame_orientation,
                        aberration_corrections="NONE",)

# create bodies
bodies_to_create = ["Earth", "Mars", "Sun"]

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

bodies = environment_setup.create_system_of_bodies(body_settings)

# launch_window
launch_window22 = 8304.5 * constants.JULIAN_DAY


def get_lambert_arc_history(lambert_arc_state_model, simulation_result):

    lambert_arc_states = dict()
    for state in simulation_result:
        lambert_arc_states[state] = lambert_arc_state_model.get_cartesian_state(state)

    return lambert_arc_states


def get_positions(kepler_ephemeris: KeplerEphemeris, departure_epoch: float, arrival_epoch: float, n_timesteps: int = 100):
    times = np.linspace(departure_epoch, arrival_epoch, n_timesteps)
    positions = np.zeros((3, 100, 3))

    for i_time, time in enumerate(times):
        positions[0, i_time] = kepler_ephemeris.get_cartesian_position(time)
        positions[1, i_time] = spice_interface.get_body_cartesian_state_at_epoch(
            **ephemeris_kwargs,
            target_body_name="Earth",
            ephemeris_time=time)[:3]
        positions[2, i_time] = spice_interface.get_body_cartesian_state_at_epoch(
            **ephemeris_kwargs,
            target_body_name="Mars",
            ephemeris_time=time)[:3]

    return positions


def do_lambert(departure_epoch: float, arrival_epoch: float, earth_parking_orbit: float = 250_000, mars_parking_orbit: float = 500_000):
    def get_circular_velocity(body, altitude):
        mu = bodies.get_body(body).gravitational_parameter
        radius = bodies.get_body(body).shape_model.average_radius

        return (mu / (altitude + radius)) ** 0.5

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body("Sun").gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        **ephemeris_kwargs,
        target_body_name="Earth",
        ephemeris_time=departure_epoch)

    final_state = spice_interface.get_body_cartesian_state_at_epoch(
        **ephemeris_kwargs,
        target_body_name="Mars",
        ephemeris_time=arrival_epoch)

    # Create Lambert targeter
    lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter)

    v_lambert_earth, v_lambert_mars = lambert_targeter.get_velocity_vectors()

    v_earth_orbit = get_circular_velocity("Earth", earth_parking_orbit) + np.linalg.norm(initial_state[3:])
    v_mars_orbit = get_circular_velocity("Mars", mars_parking_orbit) + np.linalg.norm(final_state[3:])

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambert_targeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = conversion.cartesian_to_keplerian(lambert_arc_initial_state,
                                                                       central_body_gravitational_parameter)

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, departure_epoch, central_body_gravitational_parameter), "")

    return kepler_ephemeris, abs(np.linalg.norm(v_lambert_earth) - v_earth_orbit), abs(np.linalg.norm(v_lambert_mars) - v_mars_orbit)


if __name__ == "__main__":
    plotting: bool = False

    lines = []
    while True:
        start_time = launch_window22 + np.random.randint(-20, 20) * constants.JULIAN_DAY
        arrival_time = start_time + np.random.randint(60, 450) * constants.JULIAN_DAY

        kepler_ephemeris, delta_v1, delta_v2 = do_lambert(start_time, arrival_time)

        if plotting:
            positions = get_positions(kepler_ephemeris, start_time, arrival_time)

            spacecraft_position = positions[0, :, :2]
            earth_position = positions[1, :, :2]
            mars_position = positions[2, :, :2]

            if not lines:
                plt.grid()
                lines.append(plt.plot(*spacecraft_position.T, rasterized=True, label='Spacecraft')[0])
                lines.append(plt.plot(*earth_position.T, rasterized=True, label='Earth')[0])
                lines.append(plt.plot(*mars_position.T, rasterized=True, label='Mars')[0])

                plt.ylim((-1.8 * constants.ASTRONOMICAL_UNIT, 1.8 * constants.ASTRONOMICAL_UNIT))
                plt.xlim((-1.8 * constants.ASTRONOMICAL_UNIT, 1.8 * constants.ASTRONOMICAL_UNIT))
                plt.legend()

                plt.pause(0.5)
            else:
                lines[0].set_data(*spacecraft_position.T)
                lines[1].set_data(*earth_position.T)
                lines[2].set_data(*mars_position.T)

                plt.pause(0.5)

        print(start_time, arrival_time, delta_v1, delta_v2)
