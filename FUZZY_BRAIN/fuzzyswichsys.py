from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import numpy as np
from PARAMETERS.simulationvariables import P,WINDOW_SIZE
from DATA_PROCESSING.slidearray import extend

"Fuzzy Generalization"

grad_fp = ctrl.Antecedent(np.arange(-0.0001, 0.012, 0.000001), 'Gradient_fp')
# grad_dd = ctrl.Antecedent(np.arange(-0.0001, 0.0001, 0.000001), 'Gradient_dd')
fp_weight_g = ctrl.Consequent(np.arange(0, 1.1, 0.001), 'fp_weight_g')
dd_weight_g = ctrl.Consequent(np.arange(0, 1.1, 0.001), 'dd_weight_g')

# Defining the fuzzy sets
# grad_fp['XXS'] = mf.gaussmf(grad_fp.universe, 0.000, 0.001)
# grad_fp['XS'] = mf.gaussmf(grad_fp.universe, 0.0015, 0.001)
# grad_fp['S'] = mf.gaussmf(grad_fp.universe, 0.003, 0.001)
# grad_fp['M'] = mf.gaussmf(grad_fp.universe, 0.0045, 0.001)
# grad_fp['L'] = mf.gaussmf(grad_fp.universe, 0.005, 0.001)
# grad_fp['XL'] = mf.gaussmf(grad_fp.universe, 0.006, 0.001)
# grad_fp['XXL'] = mf.gaussmf(grad_fp.universe, 0.008, 0.008)

grad_fp['XXS'] = mf.gaussmf(grad_fp.universe, 0.000, 0.0008)
grad_fp['XS'] = mf.gaussmf(grad_fp.universe, 0.0015, 0.0006)
grad_fp['S'] = mf.gaussmf(grad_fp.universe, 0.003, 0.0006)
grad_fp['M'] = mf.gaussmf(grad_fp.universe, 0.0045, 0.0006)
# grad_fp['L']  = mf.gaussmf(grad_fp.universe, 0.006, 0.0006)
# grad_fp['XL'] = mf.gaussmf(grad_fp.universe, 0.0075, 0.0006)
grad_fp['XXL'] = mf.trapmf(grad_fp.universe, [0.004, 0.0045, 0.02, 100])

# Define the fuzzy sets for the output variables
fp_weight_g['VL'] = mf.gaussmf(fp_weight_g.universe, 0, 0.05)
fp_weight_g['L'] = mf.gaussmf(fp_weight_g.universe, 0.25, 0.1)
fp_weight_g['M'] = mf.gaussmf(fp_weight_g.universe, 0.5, 0.1)
fp_weight_g['H'] = mf.gaussmf(fp_weight_g.universe, 0.75, 0.2)
fp_weight_g['VH'] = mf.gaussmf(fp_weight_g.universe, 1, 0.008)

dd_weight_g['d_v_low'] = mf.gaussmf(dd_weight_g.universe, 0, 0.05)
dd_weight_g['d_low'] = mf.gaussmf(dd_weight_g.universe, 0.25, 0.1)
dd_weight_g['d_medium'] = mf.gaussmf(dd_weight_g.universe, 0.5, 0.1)
dd_weight_g['d_high'] = mf.gaussmf(dd_weight_g.universe, 0.75, 0.2)
dd_weight_g['d_v_high'] = mf.gaussmf(dd_weight_g.universe, 1, 0.008)

# Define the fuzzy rules for the system identification

# Define the fuzzy rules for the system identification
rule1 = ctrl.Rule(grad_fp['XXS'], [fp_weight_g['VH'], dd_weight_g['d_v_low']], label='rule1')
rule2 = ctrl.Rule(grad_fp['XS'], [fp_weight_g['M'], dd_weight_g['d_v_high']], label='rule2')
rule3 = ctrl.Rule(grad_fp['S'], [fp_weight_g['VL'], dd_weight_g['d_v_high']], label='rule3')
rule4 = ctrl.Rule(grad_fp['XXL'], [fp_weight_g['VL'], dd_weight_g['d_v_high']], label='rule4')

system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
hybrid_sys_sim = ctrl.ControlSystemSimulation(system)


def fuzzy_switcher(y_bg, y_tcn,prev_pH):
    # prev_pH = np.concatenate((prev_pH,y_tcn),axis=0)

    prev_pH = extend(prev_pH)
    "---------------Fuzzy switching system--------------------"
    grad_TM = abs(np.gradient(prev_pH))
    # print(grad_TM)
    # Define the control system
    # w_fp, w_dd = [], []
    hybrid_ph = np.zeros(P)
    # time_rex = t
    for i in range(P):
        error_value_fp_g = grad_TM[i]
        hybrid_sys_sim.input['Gradient_fp'] = error_value_fp_g
        hybrid_sys_sim.compute()
        w_fp_temp, w_dd_temp = hybrid_sys_sim.output['fp_weight_g'], hybrid_sys_sim.output['dd_weight_g']
        combined_output = ((1 - w_dd_temp) * y_bg[i]) + ((w_dd_temp) * y_tcn[i])
        hybrid_ph[i] = combined_output

    return hybrid_ph
