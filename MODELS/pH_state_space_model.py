import numpy as np

from PARAMETERS import simulationvariables
from numpy import array
from DATA_PROCESSING.slidearray import extend
from PLANT.pH_model import PhModel

state_space_model = PhModel(theta=simulationvariables.theta,
                            qa=simulationvariables.qa,
                            x1i=simulationvariables.x1i,
                            x2i=simulationvariables.x2i,
                            x3i=simulationvariables.x3i,
                            Kw=simulationvariables.Kw,
                            Kx=simulationvariables.Kx)

write_pH = np.empty(simulationvariables.P)


def state_space_model_pH(x0, u):
    a = extend(u)

    for hh in range(simulationvariables.P):
        state = state_space_model.simulate(x0=x0, t=[0, simulationvariables.T_SAMPLE], u=a[hh])
        x0 = state[-1]
        pH_act_t = state_space_model.ph_calculation_eqn(x0)
        write_pH[hh] = pH_act_t

    # write_pH = array(write_pH)
    return write_pH
