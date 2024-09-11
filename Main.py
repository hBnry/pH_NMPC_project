import numpy as np
import pandas as pd

from MPC_CONTROLLER.minimizer import optimizer

from PARAMETERS import simulationvariables
from PLANT.pH_model import PhModel

from CONTROL_VARIABLES.setpoint_func import SetPointSetter
from CONTROL_VARIABLES.disturbance_func import disturb_setter
from CONTROL_VARIABLES.start_checker import srt_chk

from FUZZY_BRAIN.fuzzyeventhandler import event_handler, extra_acid

import matplotlib.pyplot as plt

import time as ticker

# Plant Parameters
x1i = simulationvariables.x1i
x2i = simulationvariables.x2i
x3i = simulationvariables.x3i
Kx = simulationvariables.Kx
Kw = simulationvariables.Kw
qa = simulationvariables.qa
v = simulationvariables.v
theta = simulationvariables.theta

# Initial Condition
x0 = simulationvariables.x0

# MPC Controller Parameters
P = simulationvariables.P
M = simulationvariables.M

# Simulation timing parameters
T_SAMPLE = simulationvariables.T_SAMPLE  # in seconds
T_SIM = simulationvariables.TOTAL_TIME

# DD Model Parameters
WINDOW_SIZE = simulationvariables.WINDOW_SIZE  # Time Steps
FEATURES = simulationvariables.FEATURES  # nod of Features (u, du, dph, ph )

# Initial arrays
INIT_U = simulationvariables.INIT_U
u0 = np.ones(M) * INIT_U

# Kindly change Flag values to have correct configuration
"""
NMPC:
fz_event = False
norm_flag = True
dist_flag = False
dist_f_flag = False

NMPC+Disturbance:
fz_event = False
norm_flag = False
dist_flag = True
dist_f_flag = False

NMPC + Disturbance + Fuzzy event handler:
fz_event = True
norm_flag = False
dist_flag = False
dist_f_flag = True
"""
# Flags
fz_event = False
norm_flag = True
dist_flag = False
dist_f_flag = False

# ********initialization of Needed Models******
plant = PhModel(theta=theta,
                qa=qa,
                x1i=x1i,
                x2i=x2i,
                x3i=x3i,
                Kw=Kw,
                Kx=Kx)

# Set pointer class initialization
setpoint = SetPointSetter(tval_pairs=[(0, 6.5), (500, 7)])
# setpoint = SetPointSetter(tval_pairs=[(0, 5), (250, 4.5), (500, 3.5), (750, 4)]) # acidic region setpoints
# setpoint = SetPointSetter(tval_pairs=[(0, 9), (250, 10), (500, 9.5), (750, 8)])  # basic region setpoints
setpoint.pairing()

# Arrays
pH_t = np.zeros(int(T_SIM / T_SAMPLE))
u_t = np.zeros(int(T_SIM / T_SAMPLE))
t_t = np.zeros(int(T_SIM / T_SAMPLE))
sp_t = np.zeros(int(T_SIM / T_SAMPLE))

if norm_flag:
    print('1. NMPC - ON\n2. Fuzzy Event Handler- OFF\n3.Disturbance - Not Added')
elif dist_flag:
    print('1. NMPC -ON\n2. Fuzzy event handler - OFF\n3.Disturbance - Added ')
elif fz_event and dist_f_flag:
    print('1. NMPC -ON\n2. Fuzzy event handler - ON\n3.Disturbance - Added ')

read_checker = input("Did you read the above info?: \npress 'y' to continue")

start = ticker.time()
if __name__ == '__main__':
    # Simulation Loop
    for time in range(int(T_SIM / T_SAMPLE)):

        # Taking values from the sensors and recording values
        pH_t[time] = plant.ph_calculation_eqn(x0)  # acts as a pH sensor reading
        u_t[time] = INIT_U
        t_t[time] = time

        # Taking variables for facilitating MPC Closed loop
        srt_flg = srt_chk(time)  # start flag checking

        # Setpoint info
        sp = setpoint.change(time)  # setting the set point
        sp_hat = setpoint.future_setpoint(time)  # getting the future set points for Prediction horizon
        sp_t[time] = sp

        # Disturbance and Event handler Info

        dist = disturb_setter(time)  # extracting the disturbance data
        qa_valve, event_flg = event_handler(pH_act=pH_t[time], x3i=dist)  # Fuzzy event handler deciding

        # Below if-else code is for window update(previous values from time=t)
        if srt_flg == 1:
            u = np.ones(WINDOW_SIZE) * u_t[time]
            ph = np.ones(WINDOW_SIZE) * pH_t[time]

        else:
            u = np.concatenate((u[1:len(u)],
                                [u_t[time]]))
            ph = np.concatenate((ph[1:len(ph)],
                                 [pH_t[time]]))

        u_hat = np.ones(M) * u[-1]
        y_hat = np.ones(P) * ph[-1]

        res = optimizer(x0=x0,
                        u_hat=u_hat,
                        y_hat=y_hat,
                        pH_act=pH_t[time],
                        ph_window=ph,
                        u_window=u,
                        sp=sp,
                        sp_hat=sp_hat,
                        model='state-space',
                        )  # cost function should be called here

        # Storing first input from the optimizer as the optimal 'u'
        u_opt = res[0]

        if norm_flag:
            # Simulating the Real plant(State space model) with the optimal 'u'
            state = plant.simulate(x0=x0, t=[0, T_SAMPLE], u=u_opt)

        elif dist_flag:
            state = plant.simulate(x0=x0, t=[0, T_SAMPLE], u=u_opt, x3i=dist)

        elif fz_event and dist_f_flag:
            tempqa, xacid = extra_acid(qa_valve)
            # Simulating the Real plant(State space model) in times of disturbance with the optimal 'u'
            state = plant.simulate(x0=x0, t=[0, T_SAMPLE], u=u_opt, x3i=dist, x1i=xacid, qa=tempqa)

        x0 = state[-1]

        # # Shift the horizon for better initial condition for next step
        u0[:-1] = res[1:]
        u0[-1] = res[-1]

        INIT_U = u_opt
        print(f'time:{time}, u:{u_opt}, sp:{sp}, pH_act:{pH_t[time]}', )

# your code...
end = ticker.time()

time_elapsed = end - start  # time in seconds
print(time_elapsed)
data = pd.DataFrame()
data['t(s)'] = t_t
data['sp'] = sp_t
data['u(mL/s)'] = u_t
data['ph'] = pH_t
data['elapsed(s)'] = time_elapsed

