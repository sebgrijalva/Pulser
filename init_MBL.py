import numpy as np
from skopt.utils import cook_initial_point_generator

def init():
    
    global U_0
    U_0 = 2 * np.pi * 2.0

    global Omega_max
    Omega_max = 1.0 * U_0 
    
    global delta_0
    delta_0 = -2.0 * U_0
    
    global delta_f
    delta_f = 1.8 * U_0 
    
    global m
    m = 5 
    
    global bounds 
    bounds = [(0, Omega_max)] * m + [(delta_0, delta_f)] * m
    
    global T
    T= 5000

    global N
    N = 7
    
    global time_domain
    time_domain = np.linspace(0,T,T)
    
    global sampling_rate
    sampling_rate = 0.02
    
    global n_jobs
    n_jobs = 1
    
    global W
    W = 0.06
    
    global lhs2
    lhs2 = cook_initial_point_generator("lhs", criterion="maximin")
    
    global op
    op = {'bounds':bounds,'n_r':30,'n_c':60,'initial_point_generator':lhs2,'score_limit':0.1}
    
    global result_string
    result_string = "Result_MBL.txt"
    
    global alpha
    alpha = 0.07 * np.pi
