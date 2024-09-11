import numpy as np

from PARAMETERS import simulationvariables


class SetPointSetter:
    """setpoint = SetpointSetter(tval_pairs = [(0,5), (50,8), (80,9)])
    First - Create instance like above
    Second - call pairing function
    Third - use change function at a time instant for particular setpoint to be set"""
    TOT_TIME = simulationvariables.TOTAL_TIME
    sp_arr = np.zeros(simulationvariables.P)
    P = simulationvariables.P
    pairs = []

    def __init__(self, tval_pairs):
        self.tval_pairs = tval_pairs

    def pairing(self):

        for i in range(len(self.tval_pairs)):
            if i < (len(self.tval_pairs) - 1):
                self.pairs.append((self.tval_pairs[i][0],
                                   self.tval_pairs[i + 1][0],
                                   self.tval_pairs[i][1]))
            else:
                self.pairs.append((self.tval_pairs[i][0],
                                   self.TOT_TIME,
                                   self.tval_pairs[i][1]))
        return self.pairs

    def change(self, t):
        # print(self.pairs)
        for i in range(len(self.pairs)):
            if self.pairs[i][0] <= t <= self.pairs[i][1]:
                sp = self.pairs[i][2]
        return sp

    def future_setpoint(self, t):
        # for i in range(len(self.pairs)):
        #     print('t in boundary')
        #     if self.pairs[i][0] < (t + self.P) < self.pairs[i][1]:
        #         for j in range(self.P):
        #             self.sp_arr[j] = self.pairs[i][2]
        #
        #     elif t + self.P >= self.pairs[i][1]:
        #         print('t on and out boundary')
        #         for k in range(self.P):
        #
        #             for m in range(len(self.pairs)):
        #                 if self.pairs[m][0] <= t <= self.pairs[m][1]:
        #                     sp_1 = self.pairs[m][2]
        #                 self.sp_arr[k] = sp_1
        #             t+=1
        t=t+2
        for i in range(self.P):
            for j in range(len(self.pairs)):
                if self.pairs[j][0] < t <= self.pairs[j][1]:
                    spdat = self.pairs[j][2]
                    self.sp_arr[i] = spdat

            t = t + 1
        self.sp_arr[-1] = self.sp_arr[-2]
        return self.sp_arr
