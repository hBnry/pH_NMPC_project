import json
import time

import BondGraphTools
from scipy.integrate import odeint
from scipy.optimize import minimize

from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
import numpy as np

from PARAMETERS import simulationvariables
from numpy import array
from DATA_PROCESSING.slidearray import extend
from PLANT.pH_model import PhModel
from BondGraphTools.actions import new, add, connect
from sympy import simplify
import re
from fractions import Fraction

state_space_model = PhModel(theta=simulationvariables.theta,
                            qa=simulationvariables.qa,
                            x1i=simulationvariables.x1i,
                            x2i=simulationvariables.x2i,
                            x3i=simulationvariables.x3i,
                            Kw=simulationvariables.Kw,
                            Kx=simulationvariables.Kx)

"Bond graph"

"----------BOND_GRAPH_MODEL_CREATION----------"

model_acid = new(name='Acid in a reactor')

# Define the components
# Acid mass
S_ac_flow = new("Sf", name='Acid')
mass_flow_jun = new("1")
mass_bal_acid = new("0", name='hi')
Comp_ac_A = new("C", name='acid_accum', value=1)
Re_a = new("R", name='acid_decay')
T_a = new(name="T_acid", component="TF", library='base')

# adding the components
add(model_acid,
    S_ac_flow,
    mass_bal_acid,
    mass_flow_jun,
    Comp_ac_A,
    Re_a,
    T_a)

# providing the connection between the elements

# providing the connection between the elements
connect(S_ac_flow, mass_flow_jun)  # SF to 1 junction
connect(mass_flow_jun, (T_a, 0))
connect((T_a, 1), mass_bal_acid)
connect(mass_bal_acid, Comp_ac_A)
connect(mass_bal_acid, Re_a)
"---------assign_values---------"

print(model_acid.bonds)


def extract_terms():
    equation_str = str(model_acid.constitutive_relations[0]).replace('dx_0', '').replace('f_9', '')

    if 'f_9' in equation_str:
        result = str(simplify(equation_str)).split('+')[0]
        # print(result)
        # Split the equation into individual terms
        parts = re.split(r'(?<!e)-', result)

    else:
        # Split the equation into individual terms
        result = str(simplify(equation_str)).split('+')[0]
        parts = re.split(r'(?<!e)-', result)
        # print(result)

    # Convert the string to a Fraction object
    fraction_1 = Fraction(parts[0].replace('*', '').replace('x_0', ''))
    result_0 = float(fraction_1)

    fraction_2 = Fraction(parts[1])
    result_1 = float(fraction_2)

    # del equation_str, result, parts, fraction_1, fraction_2
    return result_0, result_1


def Prediction(x2, t, u):
    dxdt = [
        1 / simulationvariables.theta * (simulationvariables.x1i - x2[0]) - (1 / simulationvariables.theta) * x2[0] * (
                    u / simulationvariables.qa),
        -1 / simulationvariables.theta * x2[1] + (1 / simulationvariables.theta) * (simulationvariables.x2i - x2[1]) * (
                    u / simulationvariables.qa),
        -1 / simulationvariables.theta * x2[2] + (1 / simulationvariables.theta) * (simulationvariables.x3i - x2[2]) * (
                    u / simulationvariables.qa)]

    return dxdt


def prediction_bg(x, t, result_0, result_1):
    dxdt = (-result_0 * x) + result_1
    return dxdt


ph_accum_bg = np.empty(simulationvariables.P)


def bond_graph_model(x0, u):
    u_rt_1 = extend(u)

    # Define the plant model
    for i in range(simulationvariables.P):
        qb = simulationvariables.qa * u_rt_1[i]  # ml^3 / sec

        # Transfer ratio and Resistor
        Tf_a = (simulationvariables.qa * simulationvariables.x1i) / simulationvariables.v  # Transformer Ratio
        R_a = ((simulationvariables.qa / simulationvariables.v) + (qb / simulationvariables.v))  # Decay term

        Re_a.set_param('r', 1 / R_a)
        S_ac_flow.set_param('f', 1)
        Comp_ac_A.set_param('C', 1)
        T_a.set_param('r', 1 / Tf_a)
        # a, b = extract_terms()

        # x_next_a = odeint(prediction_bg, x0[0], [0, simulationvariables.T_SAMPLE], args=(a, b,))
        x_next = odeint(Prediction, x0, [0, simulationvariables.T_SAMPLE], args=(u_rt_1[i],))  # [-1]

        # current_x_a = x_next_a[0]
        # x_next[0][0] = current_x_a

        pH_next = state_space_model.ph_calculation_eqn(x_next[0])

        x0 = x_next[-1]
        # x0[0] = x_next_a[-1]
        ph_accum_bg[i] = pH_next

    return ph_accum_bg
