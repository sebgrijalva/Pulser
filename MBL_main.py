import numpy as np
import matplotlib.pyplot as plt
import qutip
import time

from pulser import Pulse, Sequence, Register, Simulation
from pulser.waveforms import ConstantWaveform, RampWaveform, CustomWaveform
from pulser.devices import Chadoq2
from pulser.simresults import SimulationResults

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from skopt import gp_minimize
from skopt import callbacks
from skopt.utils import cook_initial_point_generator

from routines_MBL import *
import init_MBL as iMBL
import csv
import json
from mpi4py import MPI


def main():
    iMBL.init()
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()
    indices = splitting(p)
    main_MBL(indices[my_rank])
    
def main_MBL(indices_array):
    register_dict = {i:{} for i in indices_array}
    for r in register_dict.values():
        r['W_list'] = np.random.uniform(-iMBL.W,iMBL.W,iMBL.N)
        r['reg'] = reg_hc_from_W_list(r['W_list'])
    register_dict, R_opt = pulses_optimisation(register_dict)
    R_opt = {'{}'.format(k):{'W_list':v['W_list'].tolist(),'para':v['para']} for k,v in R_opt.items()}
    my_data = json.dumps(R_opt)
    String = iMBL.result_string  
    f = open(String, "w")
    f.close()
    f = open(String, "a")
    f.write(my_data + "\n")
    f.close()

def av_occup_opti(register_dict, op):
    register_dict_opt = {}
    register_dict_unopt = {}
    t_run = 0
    for i in register_dict.keys():
        def score(para):
            expect_val = create_obs(register_dict[i]['reg'],para)
            F = sum([(-1)**j*expect_val[j][-1] for j in range(iMBL.N)])/iMBL.N
            return 1 - F
        t1=time.process_time()
        if 'x0' in register_dict[i]:
            RESULT = gp_minimize(score, op['bounds'], n_random_starts = 0, n_calls = 40, verbose=False,initial_point_generator=op['initial_point_generator'],x0=register_dict[i]['x0'],y0=register_dict[i]['y0'], kappa = 0.5)
        else : 
            RESULT = gp_minimize(score, op['bounds'], n_random_starts=op['n_r'], n_calls=op['n_c'], verbose=False,initial_point_generator=op['initial_point_generator'])
        t2=time.process_time()
        t_run+=t2-t1
        print('Score reached for config.{}: '.format(i), np.round(RESULT.fun,3))
        if RESULT.fun < op['score_limit']:
            register_dict_opt[i]=merge_two_dicts(register_dict[i],{'para':RESULT.x,'score':RESULT.fun})
        else:
            x0 = [x for _,x in sorted(zip(RESULT.func_vals, RESULT.x_iters))]
            y0 = sorted(RESULT.func_vals)[:op['n_r']]
            x0 = x0[:op['n_r']]
            register_dict_unopt[i]=merge_two_dicts(register_dict[i],{'x0':x0,'y0':y0})
    print('Average run time: ', t_run/len(register_dict))
    print('{}/{}'.format(len(register_dict_opt),len(register_dict)))
    return register_dict_unopt, register_dict_opt

def pulses_optimisation(register_dict):
    A = 0
    R = len(register_dict)
    REGISTER_dict_opt = {}
    while len(REGISTER_dict_opt) < R and A<4:
        register_dict, register_dict_opt = av_occup_opti(register_dict, iMBL.op)
        REGISTER_dict_opt = merge_two_dicts(REGISTER_dict_opt,register_dict_opt)
        A += 1
    print('{}/{}'.format(len(REGISTER_dict_opt),R))
    return register_dict, REGISTER_dict_opt

def splitting(N_cores):
    return np.array_split(range(iMBL.n_jobs),N_cores)

        
def get_dict_from_string(String):
    register_dict={}
    with open(String) as f:
        for line in f:
            print(line)
            dict = json.loads(line)
            for i,r in dict.items():
                register_dict[int(i)]=r
    return register_dict

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z      
