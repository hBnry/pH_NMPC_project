from scipy.optimize import minimize
from PARAMETERS.simulationvariables import BOUNDS
from MPC_CONTROLLER.costfunction import costfunc_2


def optimizer(x0, pH_act, ph_window, u_window, u_hat, y_hat, sp, model, sp_hat):
    """Model -> state-space, bg,
     lstm, cnn-lstm, tcn,
         bg-tcn """
    res = minimize(fun=costfunc_2,
                   x0=u_hat,
                   method='SLSQP',
                   args=(y_hat, sp_hat, u_window, ph_window, pH_act, sp, x0, model),
                   bounds=BOUNDS, options={'eps': 1e-06, 'ftol': 1e-01})

    return res.x
