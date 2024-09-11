import pickle
from PARAMETERS.simulationvariables import P, M
import numpy as np
from MODEL_FROZER.freezer import model_defrost
import tensorflow as tf

s_x, s_y, window = pickle.load(
    open('C:\\Users\\Henry\\PycharmProjects\\MPC project\\pickles\\cnnlstm_needed_Param.pkl', 'rb'))

cnnlstm_mdl = model_defrost(file_path='C:\\Users\\Henry\\PycharmProjects\\MPC '
                                      'project\\MODEL_FROZER\\frozen_models\\cnnlstm_frozen_graph.pb',
                            show_graph=True)
print(f'window: {window}')

# Preallocate the full size of u_all
u_all = np.empty(window + M + (P - M))
# Preallocate y_all
y_all = np.empty(window + P)


def cnnlstm(u_suggest, u, ph, ph_hat):
    # u_hat_P = np.ones(P - M) * u_suggest[-1]
    # u_all = np.concatenate((u, u_suggest, u_hat_P), axis=None)
    # y_all = np.append(ph, ph_hat)

    # Fill the preallocated array
    u_all[:len(u)] = u
    u_all[len(u):len(u) + len(u_suggest)] = u_suggest
    u_all[len(u) + len(u_suggest):] = u_suggest[-1]

    # Fill the preallocated array
    y_all[:len(ph)] = ph
    y_all[len(ph):] = ph_hat
    # print(u_all)

    X = np.transpose([u_all, y_all])
    Y = np.transpose([y_all])

    # print(Y)
    Xs = s_x.transform(X)
    Ys = s_y.transform(Y)

    # Append if the window (past) and Prediction (future) array
    # Xsq = Xs.copy()
    # Ysq = Ys.copy()
    # print(Xsq.shape)
    # print(Xsq)

    for i in range(window, len(Xs)):
        Xin = Xs[i - window: i].reshape((1, window, np.shape(Xs)[1]))
        # print(Xin)
        Xin = tf.constant(Xin, dtype=tf.float32)

        # Prediction using frozen lstm model
        Xs[i][(Xs.shape[1] - Ys.shape[1]):] = cnnlstm_mdl(x=tf.constant(Xin))[0][0].numpy()

        Ys[i] = Xs[i][(Xs.shape[1] - Ys.shape[1]):]
    # print(Ysq.shape)
    Ytu = np.ravel(s_y.inverse_transform(Ys))

    return Ytu[window:]
