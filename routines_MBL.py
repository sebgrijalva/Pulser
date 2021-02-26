
import numpy as np
import qutip
from scipy.interpolate import PchipInterpolator
import Pulser
from pulser import Pulse, Sequence, Register, Simulation
from pulser.waveforms import ConstantWaveform, RampWaveform, CustomWaveform
from pulser.devices import Chadoq2
from pulser.simresults import SimulationResults


import init_MBL as iMBL
iMBL.init()

def reg(N, W):
    U =  [iMBL.U_0 * (1 + np.random.uniform(-W, W)) for _ in range(N)]
    W = [u/iMBL.U_0-1 for u in U]
    def R(j):
        return Chadoq2.rydberg_blockade_radius(U[j]) 
    coords = np.array([(sum(R(k) for k in range(i)), 0.) for i in range(N)])
    return Register.from_coordinates(coords, prefix='atom'),W

def reg_hc(N,W,alpha):
    U =  [iMBL.U_0 * (1 + np.random.uniform(-W, W)) for _ in range(N)]
    W = [u/iMBL.U_0-1 for u in U]
    def R(j):
        return Chadoq2.rydberg_blockade_radius(U[j]) 
    coords = np.array([(sum(R(k)*np.cos(k*alpha) for k in range(i)),sum(R(k)*np.sin(k*alpha) for k in range(i))) for i in range(N)])
    reg_c = Register.from_coordinates(coords, prefix='atom')
    reg_c.rotate(-(N-2)/2*alpha/np.pi*180)
    return reg_c,W

def reg_from_W_list(W_list):
    U =  [iMBL.U_0 * (1 + w) for w in W_list]
    def R(j):
        return Chadoq2.rydberg_blockade_radius(U[j]) 
    coords = np.array([(sum(R(k) for k in range(i)), 0.) for i in range(len(W_list))])
    return Register.from_coordinates(coords, prefix='atom')

def reg_hc_from_W_list(W_list):
    U =  [iMBL.U_0 * (1 + w) for w in W_list]
    def R(j):
        return Chadoq2.rydberg_blockade_radius(U[j]) 
    coords = np.array([(sum(R(k)*np.cos(k*iMBL.alpha) for k in range(i)),sum(R(k)*np.sin(k*iMBL.alpha) for k in range(i))) for i in range(len(W_list))])
    reg_c = Register.from_coordinates(coords, prefix='atom')
    reg_c.rotate(-(len(W_list)-2)/2*iMBL.alpha/np.pi*180)
    return reg_c

def occupation(j, N):
    up = qutip.basis(2,0)
    prod = [qutip.qeye(2) for _ in range(N)]
    prod[j] = qutip.sigmaz()
    return qutip.tensor(prod)

def stg_mag(state):
    N = int(np.log2(state.shape[0]))
    return sum([(-1)**j*(state.dag()*occupation(j,N)*state).tr() for j in range(N)])

def interp_pulse_functions(Omega_pts,delta_pts,T):
    m=len(Omega_pts)
    ti=np.linspace(0,T,m)

    cso = PchipInterpolator(ti,np.array(Omega_pts))
    csd = PchipInterpolator(ti,np.array(delta_pts))
    def Omega(t,*args):
        return cso(t)
    def delta(t,*args):
        return csd(t)
    return Omega,delta
def create_interp_pulse(para):
    Omega_pts = np.r_[1e-9, para[:iMBL.m], 1e-9]
    delta_pts = np.r_[iMBL.delta_0, para[iMBL.m:], iMBL.delta_f]
    Omega_func, delta_func = interp_pulse_functions(Omega_pts, delta_pts,iMBL.T)
    Omega,delta=np.array(Omega_func(iMBL.time_domain)),np.array(delta_func(iMBL.time_domain))
    return Pulse(CustomWaveform(Omega),CustomWaveform(delta),0)

def interp_seq(reg,para):
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel('ising', 'rydberg_global')
    seq.add(create_interp_pulse(para),'ising')
    return seq

def create_obs(reg,para):
    seq = interp_seq(reg,para)
    simul = Simulation(seq,sampling_rate=iMBL.sampling_rate)
    results = simul.run()
    return results.expect([occupation(j,iMBL.N) for j in range(iMBL.N)])

