import pandas as pd
import csv
from numpy import min, max


class ProcessData:

    df = pd.read_csv('C:\\Users\\Henry\\PycharmProjects\\CNN-LSTM_pH_MPC\\Datasets\\pH_RT_ramp_input.csv', header=None)
    # normalization_params = {'X_min': X_min, 'X_max': X_max, 'y_min': y_min, 'y_max': y_max}
    df = df[::10]
    df = pd.DataFrame(df)
    df = df.values[:, 1:]
    df = pd.DataFrame(df)


    def write_to_csv(self, norm_data):
        field_names = ['X_min', 'X_max', 'y_min', 'y_max']
        data = [norm_data]
        with open('../norm_data.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data)
        print('Normalization value written')

    def findmaxmin(self):
        """"array should be 2*2"""
        # Normalize X_train (Min-Max)
        X_min = min(self.df.values, axis=0)
        X_max = max(self.df.values, axis=0)
        # Save normalization parameters
        normalization_params = {'X_min': X_min, 'X_max': X_max}
        self.write_to_csv(normalization_params)

        return normalization_params

    def normalization_elem(self, array, norm_data, loc):
        # Normalize new data using the same parameters
        return (array - norm_data['X_min'][loc]) / (norm_data['X_max'][loc] - norm_data['X_min'][loc])

    def normalization(self, array, norm_data):
        # Normalize new data using the same parameters
        return (array - norm_data['X_min']) / (norm_data['X_max'] - norm_data['X_min'])

    def denormalization_elem(self, array, norm_data, loc):
        # Denormalize the predicted values
        return array * (norm_data['X_max'][loc] - norm_data['X_min'][loc]) + norm_data['X_min'][loc]
