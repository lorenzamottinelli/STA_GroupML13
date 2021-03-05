from tudatpy.kernel import constants
from tudatpy.kernel.astro import two_body_dynamics, conversion
from tudatpy.kernel.astro.ephemerides import KeplerEphemeris
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup

import matplotlib.pyplot as plt
import numpy as np

spice_interface.load_standard_kernels()

# Set global frame origin, orientation
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"
ephemeris_kwargs = dict(observer_body_name=global_frame_origin,
                        reference_frame_name=global_frame_orientation,
                        aberration_corrections="NONE",)

# Create bodies
bodies_to_create = ["Earth", "Mars", "Sun"]

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

bodies = environment_setup.create_system_of_bodies(body_settings)

# Launch window at September 26 2022
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
            target_body_name="Earth",
            **ephemeris_kwargs,
            ephemeris_time=time)[:3]
        positions[2, i_time] = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name="Mars",
            **ephemeris_kwargs,
            ephemeris_time=time)[:3]

    return positions


# Get the lambert arc solution, given a departure and arrival time and considering fixed circular parking orbits at
# Earth and Mars at 250 km and 500 km.
def do_lambert(departure_epoch: float, arrival_epoch: float, earth_parking_orbit: float = 250_000,
               mars_parking_orbit: float = 500_000):
    def get_circular_velocity(body, altitude):
        mu = bodies.get_body(body).gravitational_parameter
        radius = bodies.get_body(body).shape_model.average_radius

        return (mu / (altitude + radius)) ** 0.5

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body("Sun").gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        **ephemeris_kwargs,
        ephemeris_time=departure_epoch)

    final_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Mars",
        **ephemeris_kwargs,
        ephemeris_time=arrival_epoch)

    # Create Lambert targeter
    lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter)

    # Get the delta V necessary to leave Earth parking orbit and to inject in Mars parking orbit
    v_lambert_earth, v_lambert_mars = lambert_targeter.get_velocity_vectors()

    # Get the delta V necessary to orbit in Earth and Mars parking orbit
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
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, departure_epoch,
                                              central_body_gravitational_parameter), "")

    return kepler_ephemeris, abs(np.linalg.norm(v_lambert_earth) - v_earth_orbit), \
           abs(np.linalg.norm(v_lambert_mars) - v_mars_orbit)


# Actual main code
data = []
if __name__ == "__main__":
    # For plotting set equal to: True
    plotting: bool = False

    lines = []

    # Set the number of simulations to create the dataset
    simulations_max = 100000
    simulations = 0
    while simulations < simulations_max:
        # Randomize departure and arrival times. A randomly picked deviation of 25 days from the september 2022 launch
        # window is taken as the departure time and the arrival time is randomly taken, as 60 days as a minimum and
        # 450 days as a maximum after departure time
        start_time = launch_window22 + np.random.randint(-25, 25) * constants.JULIAN_DAY
        arrival_time = start_time + np.random.randint(60, 450) * constants.JULIAN_DAY

        # Get the Kepler ephemeris, delta V1 and delta V2
        kepler_ephemeris, delta_v1, delta_v2 = do_lambert(start_time, arrival_time)

        # Plot the 2D transfer orbit
        if plotting:
            positions = get_positions(kepler_ephemeris, start_time, arrival_time)

            spacecraft_position = positions[0, :, :2]
            earth_position = positions[1, :, :2]
            mars_position = positions[2, :, :2]

            #spacecraft_position_3d = positions[0, :, :3]
            #earth_position_3d = positions[1, :, :3]
            #mars_position_3d = positions[2, :, :3]

            if not lines:
                plt.grid()
                lines.append(plt.plot(*spacecraft_position.T, rasterized=True, label='Spacecraft')[0])
                lines.append(plt.plot(*earth_position.T, rasterized=True, label='Earth')[0])
                lines.append(plt.plot(*mars_position.T, rasterized=True, label='Mars')[0])

                plt.ylim((-1.8 * constants.ASTRONOMICAL_UNIT, 1.8 * constants.ASTRONOMICAL_UNIT))
                plt.xlim((-1.8 * constants.ASTRONOMICAL_UNIT, 1.8 * constants.ASTRONOMICAL_UNIT))
                plt.xlabel("x [m]", fontsize = 20)
                plt.ylabel("y [m]", fontsize = 20)
                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.legend(prop={"size": 16})

                """# To plot in 3D (only works for 1 orbit so far :( )
                from mpl_toolkits import mplot3d
                fig = plt.figure(figsize=(9, 9))
                ax = fig.gca(projection='3d')

                plt.grid()
                lines_3d.append(plt.plot(*spacecraft_position_3d.T, rasterized=True, label='Spacecraft')[0])
                lines_3d.append(plt.plot(*earth_position_3d.T, rasterized=True, label='Earth')[0])
                lines_3d.append(plt.plot(*mars_position_3d.T, rasterized=True, label='Mars')[0])
                ax.set_zlim(-2.5*10**11, 2.5*10**11)
                plt.legend(prop={"size": 16})
                """

                plt.pause(5)

            else:
                lines[0].set_data(*spacecraft_position.T)
                lines[1].set_data(*earth_position.T)
                lines[2].set_data(*mars_position.T)

                plt.pause(5)

        #print(start_time, arrival_time, delta_v1, delta_v2)

        outputs = [start_time, arrival_time, delta_v1, delta_v2]
        data.append(outputs)

        simulations = simulations + 1


# Write the results to a text file
with open('Output.txt', 'w') as f:
    f.write("%s\n" % str(["Start time [s]", "Arrival time [s]", "delta V_1 [m/s]", "delta V_2 [m/s]"]))
    for item in data:
        f.write("%s\n" % item)

