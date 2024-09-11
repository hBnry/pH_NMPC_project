from PARAMETERS import simulationvariables
from numpy import  concatenate, repeat, zeros,ones

import numpy as np
from PARAMETERS import simulationvariables
from PLANT.pH_model import PhModel


ph_window = ones(simulationvariables.WINDOW_SIZE)
u_window = ones(simulationvariables.WINDOW_SIZE)
t_window = zeros(simulationvariables.WINDOW_SIZE)

state_space_model = PhModel(theta=simulationvariables.theta,
                            qa=simulationvariables.qa,
                            x1i=simulationvariables.x1i,
                            x2i=simulationvariables.x2i,
                            x3i=simulationvariables.x3i,
                            Kw=simulationvariables.Kw,
                            Kx=simulationvariables.Kx)


def initmatrix(pH):

    if pH == 6:
        u = [3.31E-01, 3.31E-01, 3.31E-01, 3.31E-01, 3.31E-01, 3.31E-01, 3.31E-01, 3.31E-01, 3.31E-01, 3.31E-01]
        ph = np.ones(simulationvariables.WINDOW_SIZE)*6
        t = 0
        for i in range(simulationvariables.WINDOW_SIZE):

            u_window[i] = u[i]
            ph_window[i] = ph[i]
            t_window[i] = t
            t=t+1
    # a = np.ones(simulationvariables.WINDOW_SIZE) * u_init
    # for hh in range(simulationvariables.WINDOW_SIZE):
    #     pH_act_t = state_space_model.ph_calculation_eqn(x0)
    #     state = state_space_model.simulate(x0=x0, t=[0, simulationvariables.T_SAMPLE_PRED], u=a[hh])
    #     x0 = state[-1]
    #
    #     t_window[hh] = t
    #     u_window[hh] = a[hh]
    #     ph_window[hh] = pH_act_t
    #     t = t + 1
    return t_window,u_window,ph_window


def slide_array_ph(x, start_flg):
    if start_flg == 0:
        for i in range(simulationvariables.WINDOW_SIZE):
            if i < (simulationvariables.WINDOW_SIZE - 1):
                ph_window[i] = ph_window[i + 1]
            else:
                ph_window[i] = x
    else:
        for i in range(simulationvariables.WINDOW_SIZE):
            if i < (simulationvariables.WINDOW_SIZE - 1):
                ph_window[i] = x
            else:
                ph_window[i] = x

    return ph_window


def slide_array_u(x, start_flg):
    if start_flg == 0:
        for i in range(simulationvariables.WINDOW_SIZE):
            if i < (simulationvariables.WINDOW_SIZE - 1):
                u_window[i] = u_window[i + 1]
            else:
                u_window[i] = x
    else:
        for i in range(simulationvariables.WINDOW_SIZE):
            if i < (simulationvariables.WINDOW_SIZE - 1):
                u_window[i] = 0
            else:
                u_window[i] = x
    return u_window


def slide_array_t(x):
    for i in range(simulationvariables.WINDOW_SIZE):
        if i < (simulationvariables.WINDOW_SIZE - 1):
            t_window[i] = t_window[i + 1]
        else:
            t_window[i] = x
    return t_window


def extend(u):
    """We optimise the first M values of 'u' but we need P values for prediction"""
    return concatenate([u, repeat(u[-1], simulationvariables.P - simulationvariables.M)])
