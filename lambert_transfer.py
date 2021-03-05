from tudatpy.kernel import constants
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
import numpy as np

spice_interface.load_standard_kernels()

global_frame_origin = "Sun"
global_frame_orientation = "ECLIPJ2000"

# create bodies
bodies_to_create = ["Earth", "Mars", "Sun"]

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

bodies = environment_setup.create_system_of_bodies(body_settings)

# launch_window
launch_window22 = 8304.5 * constants.JULIAN_DAY


def do_lambert(departure_epoch: float, arrival_epoch: float, earth_parking_orbit: float = 250_000, mars_parking_orbit: float = 500_000):
    def get_circular_velocity(body, altitude):
        mu = bodies.get_body(body).gravitational_parameter
        radius = bodies.get_body(body).shape_model.average_radius

        return (mu / (altitude + radius)) ** 0.5

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body("Sun").gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name=global_frame_origin,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch)

    final_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Mars",
        observer_body_name=global_frame_origin,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch)

    # Create Lambert targeter
    lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter)

    v1_lambert, v2_lambert = lambert_targeter.get_velocity_vectors()
    v_earth_orbit, v_mars_orbit = get_circular_velocity("Earth", earth_parking_orbit) + np.linalg.norm(initial_state[3:]), get_circular_velocity("Mars", mars_parking_orbit) + np.linalg.norm(initial_state[3:])
    return abs(v1_lambert - v_earth_orbit), abs(v2_lambert - v_mars_orbit)

