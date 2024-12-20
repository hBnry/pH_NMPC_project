# Plant Parameters
x1i = 0.0012#0.00012
x2i = 0.0029#0.00029
x3i = 0.001#0.0001
Kx = 1e-7
Kw = 1e-14
qa = 16.67#3.212 * 0.33
v = 2500#2120.57
theta = v / qa

# Initial Condition
# x0 = [0.00011529292693590292, 1.1375426571558682e-05, 3.922560886741408e-06]  # 4
# x0= [9.406706401025964e-05,6.267126197522844e-05, 2.161077999145915e-05] # 5
x0 = [9.148935409630777e-05, 6.89007276006391e-05, 2.3758871586426582e-05]  # 6
init_pH = 6

# x0= [8.869555344456115e-05, 7.565241250901284e-05,2.6087038796211367e-05]# 7
# x0= [8.53563604613996e-05,8.372212888494015e-05, 2.8869699615496362e-05] # 8
# x0= [8.2043097709575e-05,9.172918053521971e-05,3.163075190869646e-05]  #9

# x0 =[0.0012,0,0]#[8.79E-05, 7.90E-05, 2.59E-05]
INIT_U = 0.01  # initial flow rate to be set

# Simulation timing parameters
T_SAMPLE = 1  # in seconds (deltaT)
T_SAMPLE_PRED = 1
TOTAL_TIME = 1000  # in seconds

# Declaring MPC Parameters
P = 20  # Prediction Horizon
M = 2  # Control Horizon
BOUNDS = [(0.0541296, qa)] * M  # Define the constraints

# DD MODEL PARAMETERS
WINDOW_SIZE = 10  # Time Steps
FEATURES = 2  # nod of Features (u, du, dph, ph )

CNTRL_WGT, SETPT_WGT, T_WGT = 0.5, 10, 50

