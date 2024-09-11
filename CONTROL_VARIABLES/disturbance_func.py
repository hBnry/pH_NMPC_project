from PARAMETERS.simulationvariables import x3i as buffer


def disturb_setter(time):
    if 300 <= time <= 310:
        x3i = 0.005
    elif 700 <= time <= 710:
        x3i = 0.005
    else:
        x3i = buffer

    return x3i
