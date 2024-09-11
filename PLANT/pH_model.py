from scipy.optimize import fsolve
from scipy.integrate import odeint
from math import log10
from PARAMETERS import simulationvariables

# from numpy import ravel

"""Define the Actual plant model in the loop"""


class PhModel:
    """Create an instance of ODEModel eg;-
model = pH_model(theta, qa, x1i, x2i, x3i, Kw, Kx)"""

    def __init__(self, theta, qa, x1i, x2i, x3i, Kw, Kx):
        self.theta = theta
        self.qa = qa
        self.x1i = x1i
        self.x2i = x2i
        self.x3i = x3i
        self.Kw = Kw
        self.Kx = Kx

    def plant(self, x, t, u, x3i, x1i, qa):
        dxdt = [1 / self.theta * (x1i - x[0]) - (1 / self.theta) * x[0] * (u / qa),
                -1 / self.theta * x[1] + (1 / self.theta) * (self.x2i - x[1]) * (u / qa),
                -1 / self.theta * x[2] + (1 / self.theta) * (x3i - x[2]) * (u / qa)]

        return dxdt

    # Define the pH calculation function
    def ph_calculation_eqn(self, x):
        # print(x)
        def conPH(zeta, X0):
            return zeta + X0[1] + X0[2] - X0[0] - (self.Kw / zeta) - (X0[2] / (1 + ((self.Kx * zeta) / self.Kw)))

        ph_zeta_a = fsolve(conPH, [1e-20], x)
        ph_a = -(log10(ph_zeta_a[0]))
        return ph_a  # values

    def simulate(self,
                 x0,
                 t,
                 u,
                 x3i=simulationvariables.x3i,
                 x1i=simulationvariables.x1i,
                 qa=simulationvariables.qa):

        x_next = odeint(self.plant, x0, t, args=(u, x3i, x1i, qa))
        return x_next
