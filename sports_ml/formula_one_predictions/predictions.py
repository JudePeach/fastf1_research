import fastf1
import fastf1.plotting
import time
import random
import numpy as np  
from matplotlib import pyplot as plt 

# Fast f1 is largely implemented ontop of pandas dataframes

# Get the data for a session (Monza 2019 Quali here):
session = fastf1.get_session(2019, 'Monza', 'Q')
session.load(telemetry=False, laps=False, weather=False)

# Print vettels first name (to familiarise with the data formatting)
vettel = session.get_driver('VET')
print(f"Forza {vettel['FirstName']}!")

# Now lets plot any drivers speed during their fastest lap - im choosing albono
fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')

# Load the session data
session.load()

# Get his fastest lap
albon_fastest_lap = session.laps.pick_drivers('ALB').pick_fastest()

# Get his car data from the fastest lap
albon_car_data = albon_fastest_lap.get_car_data()
time = albon_car_data["Time"]
vCar = albon_car_data["Speed"]

fig, ax = plt.subplots()
ax.plot(time, vCar, label="Alex Albons speed during his fastest lap in Monza Quali (2019)")
ax.set_xlabel("Time")
ax.set_ylabel("Speed")
ax.set_title("Alex Albons speed during his fastest lap in Monza Quali (2019)")
ax.legend()
plt.show()

