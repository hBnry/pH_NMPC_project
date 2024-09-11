import numpy as np

from MODELS.pH_state_space_model import state_space_model_pH
from MODELS.pH_cnnlstm import cnnlstm
from MODELS.pH_lstm import lstm
from MODELS.pH_bond_graph import bond_graph_model
from MODELS.pH_tcn import tcn
from FUZZY_BRAIN.fuzzyswichsys import fuzzy_switcher


def costfunc_2(u, y_hat, sp_hat, u_window, ph_window, pH_act, sp, x0, model):
    u_hat0 = np.append(u_window[-1], u)

    if model == 'cnn-lstm':
        predictions = cnnlstm(u_suggest=u,
                              u=u_window,
                              ph=ph_window,
                              ph_hat=y_hat)
        # print(predictions)
        predictions = np.add(np.ravel(predictions), 0.04)  #1.25

        # Control regulation
        cost_t = 5 * (pH_act - sp) ** 2  # Penalty term for present pH error
        cost = 10 * (np.sum((predictions - sp_hat) ** 2)) + \
               (1 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term
        # for future pH error
        cost = cost + cost_t

        # cost_tl = 5 * (pH_act - sp) ** 2  # Penalty term for present pH error
        # costl = 10 * (np.sum((predictions - sp_hat) ** 2)) + \
        #        (2 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term
        # for future pH error

        # cost = costl + cost_tl

    elif model == 'state-space':
        predictions = state_space_model_pH(x0=x0, u=u)

        # Control regulation
        cost_t = 5 * (pH_act - sp) ** 2  # Penalty term for present pH error
        cost = 200 * (np.sum((predictions - sp_hat) ** 2)) + \
               (5 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term
        # for future pH error
        cost = cost + cost_t

    elif model == 'lstm':
        predictions = lstm(u_suggest=u,
                           u=u_window,
                           ph=ph_window,
                           ph_hat=y_hat)
        predictions = np.add(np.ravel(predictions), 0.04)  #0.4

        # Control regulation
        # cost_t = 1 * (pH_act - sp) ** 2  # Penalty term for present pH error
        # cost = 60 * (np.sum((predictions - sp_hat) ** 2)) + \
        #        (0.9 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term

        cost_tl = 0.5 * (pH_act - sp) ** 2  # Penalty term for present pH error
        costl = 60 * (np.sum((predictions - sp_hat) ** 2)) + \
               (10 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term
        # for future pH error
        # cost = cost + cost_t
        cost = costl + cost_tl

    elif model == 'tcn':
        predictions = tcn(u_suggest=u,
                          u=u_window,
                          ph=ph_window,
                          ph_hat=y_hat)
        # predictions = np.subtract(np.ravel(predictions),1)
        predictions = np.subtract(np.ravel(predictions), 0.006)

        # print(predictions)

        # Control regulation
        cost_t = 1.25 * (pH_act - sp) ** 2  # Penalty term for present pH error
        cost = 2.5 * (np.sum((predictions - sp_hat) ** 2)) + \
               (0.9 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term
        # for future pH error
        cost = cost + cost_t

    elif model == 'bg':
        predictions = bond_graph_model(x0=x0, u=u)

        # Control regulation
        cost_t = 5 * (pH_act - sp) ** 2  # Penalty term for present pH error
        cost = 200 * (np.sum((predictions - sp_hat) ** 2)) + \
               (1 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term
        # for future pH error
        cost = cost + cost_t
        # print(predictions)

    elif model == 'bg-tcn':
        predictions_bg = bond_graph_model(x0=x0, u=u)
        predictions_tcn_temp = tcn(u_suggest=u,
                                   u=u_window,
                                   ph=ph_window,
                                   ph_hat=y_hat)
        predictions_tcn = np.subtract(np.ravel(predictions_tcn_temp), 0.005)
        predictions = fuzzy_switcher(predictions_bg, predictions_tcn, ph_window)
        predictions = np.subtract(np.ravel(predictions), 0.005)

        # Control regulation
        cost_t = 2 * (pH_act - sp) ** 2  # Penalty term for present pH error
        cost = 150 * (np.sum((predictions - sp_hat) ** 2)) + \
               (5 * np.sum((u_hat0[1:] - u_hat0[0:-1]) ** 2))  # penalty term
        # for future pH error
        cost = cost + cost_t

    else:
        raise Exception('Enter correct model')

    # print(cost)
    return cost
