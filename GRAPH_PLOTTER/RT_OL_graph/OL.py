import numpy as np
import pandas as pd
# import scienceplots
import matplotlib.pyplot as plt
# plt.style.use(['ieee'])


data = pd.read_csv('C:\\Users\\Henry\\PycharmProjects\\MPC project\\GRAPH_PLOTTER\\RT_OL_graph\\sys_modelling_data.csv')
print(data.head())
data = data.values[::10]
print(data)

colors = np.arange( 0, len(data[:, 1]), 1)
area = np.random.rand(len(data[:, 1])) ** 2

fig = plt.figure(1, figsize=(5, 5))

plt.scatter(data[:, 6], data[:, 1], label='$y_{RT}$', c=colors, s=area)
# plt.plot(data[:, 6], data[:, 2], label='$y_{TCN}$')
plt.plot(data[:, 6], data[:, 3], label='$y_{BG-TCN}$', color='#fdc086',linestyle='--')
# plt.plot(data[:, 6], data[:, 4], label='$y_{BG}$')
plt.plot(data[:, 6], data[:, 5], label='$y_{CNN-LSTM}$',linestyle='--')
plt.xlabel('$u$')
plt.ylabel('$pH$')
plt.legend()
plt.tight_layout()
plt.show()


def relu(x):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    Parameters:
    x (float, int, or np.array): Input value or array.

    Returns:
    float, int, or np.array: Output after applying ReLU.
    """
    return np.maximum(0, x)

#
# test_n = len(data[:, 1])
# data_bg_tcn = (data[:, 3])
# data_cnn_lstm = (data[:, 5])
# data_rt = (data[:, 1])
#
# bg_tcn_abs_dev = np.abs(data_bg_tcn - data_rt)
# cnn_lstm_abs_dev = np.abs(data_cnn_lstm - data_rt)
#
# diff_pos = relu(cnn_lstm_abs_dev - bg_tcn_abs_dev).reshape(test_n)
# diff_min = -relu(bg_tcn_abs_dev - cnn_lstm_abs_dev).reshape(test_n)
#
# plt.title('HWES Predictor VS FCNN Predictor')
# plt.hlines(0, xmin=0, xmax=test_n, linestyles='dashed')
# plt.bar(list(range(test_n)), diff_pos, color='g', label='BG-TCN Wins')
# plt.bar(list(range(test_n)), diff_min, color='r', label='CNN-LSTM Wins')
# plt.legend()
# plt.show()
