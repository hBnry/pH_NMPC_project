import numpy as np
from PARAMETERS import simulationvariables


# test func
def arrangement(x1=0, x2=0, x3=0, x4=0):
    if simulationvariables.FEATURES == 4:
        data = np.column_stack((x1.reshape(simulationvariables.WINDOW_SIZE, 1),
                                x2.reshape(simulationvariables.WINDOW_SIZE, 1),
                                x3.reshape(simulationvariables.WINDOW_SIZE, 1),
                                x4.reshape(simulationvariables.WINDOW_SIZE, 1))).reshape(1, simulationvariables.WINDOW_SIZE,
                                                                                         simulationvariables.FEATURES)

    elif simulationvariables.FEATURES == 3:
        data = np.column_stack((x1.reshape(simulationvariables.WINDOW_SIZE, 1),
                                x2.reshape(simulationvariables.WINDOW_SIZE, 1),
                                x3.reshape(simulationvariables.WINDOW_SIZE, 1))).reshape(1, simulationvariables.WINDOW_SIZE,
                                                                                         simulationvariables.FEATURES)

    elif simulationvariables.FEATURES == 2:
        data = np.column_stack((x1.reshape(simulationvariables.WINDOW_SIZE, 1),
                                x2.reshape(simulationvariables.WINDOW_SIZE, 1),
                                )).reshape(1, simulationvariables.WINDOW_SIZE, simulationvariables.FEATURES)

    elif simulationvariables.FEATURES == 1:
        data = np.column_stack(x1.reshape(simulationvariables.WINDOW_SIZE, 1)).reshape(1, simulationvariables.WINDOW_SIZE,
                                                                                       simulationvariables.FEATURES)
    return data

