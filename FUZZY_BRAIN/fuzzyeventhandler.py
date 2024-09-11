from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import numpy as np
from PARAMETERS.simulationvariables import qa, x1i
from DATA_PROCESSING.slidearray import extend

# Define the input variables
pH_mon = ctrl.Antecedent(np.arange(1, 14, 0.001), 'pH_monitor')
x3i_mon = ctrl.Antecedent(np.arange(1e-4, 0.008, 0.000001), 'buff_monitor')
cost_mon = ctrl.Antecedent(np.arange(0, 100, 0.001), 'costfn_monitor')

# event_action = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'event_action')
acid_flow = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'acid_flow')
indicator = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'change_indicator')
# model_select = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'model_select')

# Defining the pH_mon sets
pH_mon['VH_acidic'] = mf.trapmf(pH_mon.universe, [1, 1.1, 6, 6.5])
# pH_mon['H_acidic'] = mf.gaussmf(pH_mon.universe, 6, 0.5)
# pH_mon['acidic'] = mf.gaussmf(pH_mon.universe, 6.5, 0.5)
pH_mon['neutral'] = mf.gaussmf(pH_mon.universe, 7, 0.5)
# pH_mon['basic'] = mf.gaussmf(pH_mon.universe, 7.5, 0.5)
# pH_mon['H_basic'] = mf.gaussmf(pH_mon.universe, 8, 0.5)
pH_mon['VH_basic'] = mf.trapmf(pH_mon.universe, [7.5, 8, 14, 15])

# cost_mon['VL'] = mf.trapmf(cost_mon.universe, [-1, -0.9, 3, 3.1])
# # cost_mon['L'] = mf.gaussmf(cost_mon.universe, 25, 8)
# cost_mon['M'] = mf.gaussmf(cost_mon.universe, 5, 1)
# # cost_mon['H'] = mf.gaussmf(cost_mon.universe, 45, 8)
# cost_mon['VH'] = mf.trapmf(cost_mon.universe, [6, 6.1, 100, 101])

x3i_mon['L'] = mf.trapmf(x3i_mon.universe, [0.000, 0.0, 0.0004, 0.0005])
x3i_mon['OPT'] = mf.gaussmf(x3i_mon.universe, 0.001, 0.00025)
x3i_mon['H'] = mf.trapmf(x3i_mon.universe, [0.0015, 0.0016, 0.1, 0.13])

# # Define the fuzzy sets for the output variables
# event_action['no'] = mf.gaussmf(event_action.universe, 0, 0.000005)
# event_action['yes'] = mf.gaussmf(event_action.universe, 1, 0.000005)

# Define the fuzzy sets for the output variables
acid_flow['less'] = mf.gaussmf(acid_flow.universe, 0, 0.0005)
acid_flow['normal'] = mf.gaussmf(acid_flow.universe, 0.5, 0.0005)
acid_flow['more'] = mf.gaussmf(acid_flow.universe, 1, 0.0005)

# Define the fuzzy sets for the output variables
indicator['False'] = mf.gaussmf(indicator.universe, 0, 0.0005)
indicator['True'] = mf.gaussmf(indicator.universe, 1, 0.0005)

# Define the fuzzy rules for the system identification

# Buffer disturbance at acidic region
# Rule 1
rule1h = ctrl.Rule(pH_mon['VH_acidic'] &
                   x3i_mon['OPT']
                   ,

                   [acid_flow['normal'],
                    indicator['False']], label='rule1h')

# Rule 2
rule2h = ctrl.Rule(pH_mon['VH_acidic'] &
                   x3i_mon['H'],

                   [acid_flow['more'],
                    indicator['True']
                    ], label='rule2h')

# Rule 3
rule3h = ctrl.Rule(pH_mon['VH_acidic'] &
                   x3i_mon['L'],

                   [acid_flow['less'],
                    indicator['True']
                    ], label='rule3h')

# Buffer disturbance at neutral region
# Rule 4
rule4h = ctrl.Rule(pH_mon['neutral'] &
                   x3i_mon['OPT'],

                   [
                       acid_flow['normal'],
                       indicator['False']], label='rule4h')

# Rule 5
rule5h = ctrl.Rule(pH_mon['neutral'] &
                   x3i_mon['H'],

                   [
                       acid_flow['more'],
                       indicator['True']
                   ], label='rule5h')

# Rule 6
rule6h = ctrl.Rule(pH_mon['neutral'] &
                   x3i_mon['L'],

                   [
                       acid_flow['less'],
                       indicator['True']
                   ], label='rule6h')

system_h = ctrl.ControlSystem([rule1h,
                               rule2h,
                               rule3h,
                               rule4h,
                               rule5h,
                               rule6h,
                               ])

hybrid_sys_sim_h = ctrl.ControlSystemSimulation(system_h)


def is_near(number, target, threshold):
    return abs(number - target) < threshold


def event_handler(pH_act, x3i):
    hybrid_sys_sim_h.input['pH_monitor'] = pH_act
    hybrid_sys_sim_h.input['buff_monitor'] = x3i

    hybrid_sys_sim_h.compute()

    # e = hybrid_sys_sim.output['event_action']
    qa_rise = hybrid_sys_sim_h.output['acid_flow']
    flag = hybrid_sys_sim_h.output['change_indicator']

    if is_near(number=qa_rise, target=0, threshold=0.1):
        qa_valve = 0
    elif is_near(number=qa_rise, target=0.5, threshold=0.1):
        qa_valve = 0.5
    elif is_near(number=qa_rise, target=1, threshold=0.1):
        qa_valve = 1

    return qa_valve, flag


def extra_acid(valve_info):
    if valve_info == 0.5:
        tempqa = (qa)
        xacid = x1i
        extra_u = 0

    elif valve_info == 1:
        tempqa = (qa) + (0.02 * (qa))
        xacid = 0.001 + 0.0012

    if valve_info == 0:
        tempqa = qa
        xacid = x1i
        extra_u = 0

    return tempqa, xacid
