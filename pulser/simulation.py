# Copyright 2020 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import qutip
import numpy as np
from copy import deepcopy

from pulser import Pulse, Sequence, Register
from pulser.clean_results import CleanResults
from pulser.noisy_results import NoisyResults
from collections import Counter


class Simulation:
    """Simulation of a pulse sequence using QuTiP.

    Creates a Hamiltonian object with the proper dimension according to the
    pulse sequence given, then provides a method to time-evolve an initial
    state using the QuTiP solvers.

    Args:
        sequence (Sequence): An instance of a Pulser Sequence that we
            want to simulate.

    Keyword Args:
        sampling_rate (float): The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0
    """

    def __init__(self, sequence, sampling_rate=1.0, noise={"Doppler": False},
                 damping={"Emission": False}):
        """Initialize the Simulation with a specific pulser.Sequence."""
        if not isinstance(sequence, Sequence):
            raise TypeError("The provided sequence has to be a valid "
                            "pulser.Sequence instance.")
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(sequence._schedule[x][-1].tf == 0 for x in sequence._channels):
            raise ValueError("No instructions given for the channels in the "
                             "sequence.")
        self._seq = sequence
        self._noise = noise
        self._damping = damping
        self._qdict = self._seq.qubit_info
        self._size = len(self._qdict)
        self._tot_duration = max(
            [self._seq._last(ch).tf for ch in self._seq._schedule]
        )

        if not (0 < sampling_rate <= 1.0):
            raise ValueError("`sampling_rate` must be positive and "
                             "not larger than 1.0")
        if int(self._tot_duration*sampling_rate) < 4:
            raise ValueError("`sampling_rate` is too small, less than 4 data "
                             "points.")
        self.sampling_rate = sampling_rate

        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}
        self.samples = {addr: {basis: {}
                               for basis in ['ground-rydberg', 'digital']}
                        for addr in ['Global', 'Local']}
        self.operators = deepcopy(self.samples)

        self._extract_samples()
        self._build_basis_and_op_matrices()
        self._construct_hamiltonian()

    def _extract_samples(self):
        """Populate samples dictionary with every pulse in the sequence."""

        def prepare_dict():
            # Duration includes retargeting, delays, etc.
            return {'amp': np.zeros(self._tot_duration),
                    'det': np.zeros(self._tot_duration),
                    'phase': np.zeros(self._tot_duration)}

        def write_samples(slot, samples_dict):
            samples_dict['amp'][slot.ti:slot.tf] += slot.type.amplitude.samples
            noise = 1
            if(self._noise["Doppler"]):
                # sigma = k_eff \Delta v : See Sylvain's paper
                noise = 1 + np.random.normal(
                            0, 2*np.pi*0.12, slot.tf - slot.ti)
            samples_dict['det'][slot.ti:slot.tf] += \
                slot.type.detuning.samples * noise
            samples_dict['phase'][slot.ti:slot.tf] = slot.type.phase

        for channel in self._seq.declared_channels:
            addr = self._seq.declared_channels[channel].addressing
            basis = self._seq.declared_channels[channel].basis

            samples_dict = self.samples[addr][basis]

            if addr == 'Global':
                if not samples_dict:
                    samples_dict = prepare_dict()
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        write_samples(slot, samples_dict)

            elif addr == 'Local':
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        for qubit in slot.targets:  # Allow multiaddressing
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            write_samples(slot, samples_dict[qubit])

            self.samples[addr][basis] = samples_dict

    def _build_basis_and_op_matrices(self):
        """Determine dimension, basis and projector operators."""
        # This case when computing spontaneous emission errors
        if self._damping["Emission"]:
            self.basis_name = 'all'  # All three states
            self.dim = 3
            basis = ['r', 'g', 'h']
            projectors = ['gr', 'hg', 'rr', 'gg', 'hh', 'hr']
        else:
            # No samples => Empty dict entry => False
            if (not self.samples['Global']['digital']
                    and not self.samples['Local']['digital']):
                self.basis_name = 'ground-rydberg'
                self.dim = 2
                basis = ['r', 'g']
                projectors = ['gr', 'rr', 'gg']
            elif (not self.samples['Global']['ground-rydberg']
                    and not self.samples['Local']['ground-rydberg']):
                self.basis_name = 'digital'
                self.dim = 2
                basis = ['g', 'h']
                projectors = ['hg', 'hh', 'gg']
            else:
                self.basis_name = 'all'  # All three states
                self.dim = 3
                basis = ['r', 'g', 'h']
                projectors = ['gr', 'hg', 'rr', 'gg', 'hh', 'hr']

        self.basis = {b: qutip.basis(self.dim, i) for i, b in enumerate(basis)}
        self.op_matrix = {'I': qutip.qeye(self.dim)}

        for proj in projectors:
            self.op_matrix['sigma_' + proj] = (
                self.basis[proj[0]] * self.basis[proj[1]].dag()
            )

    def _build_operator(self, op_id, *qubit_ids, global_op=False):
        """Create qutip.Qobj with nontrivial action at *qubit_ids."""
        if global_op:
            return sum(self._build_operator(op_id, q_id)
                       for q_id in self._qdict)
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list.")
        # List of identity operators, except for op_id where requested:
        op_list = [self.op_matrix[op_id]
                   if j in map(self._qid_index.get, qubit_ids)
                   else self.op_matrix['I'] for j in range(self._size)]
        return qutip.tensor(op_list)

    def _construct_hamiltonian(self):
        def adapt(full_array):
            """Adapt list to correspond to sampling rate"""
            indexes = np.linspace(0, self._tot_duration-1,
                                  int(self.sampling_rate*self._tot_duration),
                                  dtype=int)
            return full_array[indexes]

        def make_vdw_term():
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_rr" at each pair. The units are
            given so that the coefficient includes a 1/hbar factor.
            """
            vdw = 0
            # Get every pair without duplicates
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                dist = np.linalg.norm(
                    self._qdict[q1] - self._qdict[q2])
                U = 0.5 * self._seq._device.interaction_coeff / dist**6
                vdw += U * self._build_operator('sigma_rr', q1, q2)
            return vdw

        def make_spontaneous_emission_term(self):
            H_emi = 0
            omega_r = 2 * np.pi * 0.1
            omega_b = 2 * np.pi * 0.1
            big_delta = 2 * np.pi * 740
            # detuning
            delta = 2 * np.pi * 100
            for q in self._qdict.keys():
                H_emi += (omega_r / 2) * self._build_operator('sigma_hg',
                                                              q).dag()
                H_emi += (omega_b / 2) * self._build_operator('sigma_hr', q)
                H_emi -= (big_delta / 2) * self._build_operator('sigma_hh', q)
                H_emi -= (delta / 2) * self._build_operator('sigma_rr', q)
            return H_emi

        def build_coeffs_ops(basis, addr):
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == 'ground-rydberg':
                op_ids = ['sigma_gr', 'sigma_rr']
            elif basis == 'digital':
                op_ids = ['sigma_hg', 'sigma_gg']

            terms = []
            if addr == 'Global':
                coeffs = [0.5*samples['amp'] * np.exp(-1j * samples['phase']),
                          -0.5 * samples['det']]
                for op_id, coeff in zip(op_ids, coeffs):
                    if np.any(coeff != 0):
                        # Build once global operators as they are needed
                        if op_id not in operators:
                            operators[op_id] =\
                                self._build_operator(op_id, global_op=True)
                        terms.append([operators[op_id], adapt(coeff)])
            elif addr == 'Local':
                for q_id, samples_q in samples.items():
                    if q_id not in operators:
                        operators[q_id] = {}
                    coeffs = [0.5*samples_q['amp'] *
                              np.exp(-1j * samples_q['phase']),
                              -0.5 * samples_q['det']]
                    for coeff, op_id in zip(coeffs, op_ids):
                        if np.any(coeff != 0):
                            if op_id not in operators[q_id]:
                                operators[q_id][op_id] = \
                                    self._build_operator(op_id, q_id)
                            terms.append([operators[q_id][op_id],
                                          adapt(coeff)])

            self.operators[addr][basis] = operators
            return terms

        # Time independent term:
        if self.basis_name == 'digital':
            qobj_list = []
        else:
            # Van der Waals Interaction Terms
            qobj_list = [make_vdw_term()] if self._size > 1 else []

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += build_coeffs_ops(basis, addr)

        # Spontaneous emission hamiltonian
        if self._damping["Emission"]:
            qobj_list += [make_spontaneous_emission_term(self)]

        self._times = adapt(np.arange(self._tot_duration,
                                      dtype=np.double)/1000)

        ham = qutip.QobjEvo(qobj_list, tlist=self._times)
        ham = ham + ham.dag()
        ham.compress()

        self._hamiltonian = ham

    def get_hamiltonian(self, time):
        """Get the Hamiltonian created from the sequence at a fixed time.

        Args:
            time (float): The specific time in which we want to extract the
                    Hamiltonian (in ns).

        Returns:
            Qutip.Qobj: A new Qobj for the Hamiltonian with coefficients
                    extracted from the effective sequence (determined by
                    `self.sampling_rate`) at the specified time.
        """
        if time > 1000 * self._times[-1]:
            raise ValueError("Provided time is larger than sequence duration.")
        if time < 0:
            raise ValueError("Provided time is negative.")
        return self._hamiltonian(time/1000)  # Creates new Qutip.Qobj

    # Run Simulation Evolution using Qutip
    def run(self, initial_state=None, progress_bar=None, spam=False,
            **options):
        """Simulate the sequence using QuTiP's solvers.

        Keyword Args:
            initial_state (array): The initial quantum state of the
               evolution. Will be transformed into a
               qutip.Qobj instance.
            progress_bar (bool): If True, the progress bar of QuTiP's sesolve()
                will be shown.
            spam (bool): If True, returns a NoisyResults object instead of a
                CleanResults one, taking into account SPAM errors.

        Returns:
            SimulationResults: Object containing the time evolution results.
        """

        def _build_lindblad_term(self):
            L = []
            Gamma_r = 4 * np.pi
            Gamma_g = 2 * np.pi
            for q in self._qdict.keys():
                C_1 = np.sqrt(Gamma_r) * self._build_operator('sigma_hr', q)
                C_2 = np.sqrt(Gamma_g) * self._build_operator('sigma_hg', q)
                L += [C_1, C_2]
            return L

        if initial_state is not None:
            if isinstance(initial_state, qutip.Qobj):
                if initial_state.shape != (self.dim**self._size, 1):
                    raise ValueError("Incompatible shape of initial_state")
                self._initial_state = initial_state
            else:
                if initial_state.shape != (self.dim**self._size,):
                    raise ValueError("Incompatible shape of initial_state")
                self._initial_state = qutip.Qobj(initial_state)
        else:
            # by default, initial state is "ground" state of g-r basis.
            all_ground = [self.basis['g'] for _ in range(self._size)]
            self._initial_state = qutip.tensor(all_ground)

        L = []

        if self._damping["Emission"]:
            L = _build_lindblad_term(self)

        # qutip uses sesolve if c_ops = []
        result = qutip.mesolve(self._hamiltonian,
                               self._initial_state,
                               self._times,
                               c_ops=L,
                               progress_bar=progress_bar,
                               options=qutip.Options(max_step=5,
                                                     **options)
                               )

        if hasattr(self._seq, '_measurement'):
            meas_basis = self._seq._measurement
        else:
            meas_basis = None
        if not spam:
            return CleanResults(result.states, self.dim, self._size,
                                self.basis_name, meas_basis=meas_basis)
        else:
            return NoisyResults(
                result.states, self.dim, self._size, self.basis_name,
                meas_basis=meas_basis)

    def detection_SPAM_test(self, t=-1, spam={"eta": 0.005, "epsilon": 0.01,
                                              "epsilon_prime": 0.05},
                            N_samples=1000, meas_basis='ground-rydberg'):
        """
            Returns the probability dictionary accounting for SPAM errors.
        """
        N = self._size
        eta = spam["eta"]
        eps = spam["epsilon"]
        seq = self._seq

        def _seq_without_k(self, k):
            """
                Returns the original sequence with a modified register :
                no more atom k
            """
            seq_k = deepcopy(seq)
            dict_k = seq_k.qubit_info
            dict_k.pop('q' + str(k))
            seq_k._register = Register(dict_k)
            return seq_k

        def _evolve_without_k(self, k):
            """
                Returns a sample, in the form of a Counter, of the
                state of the system that evolved without atom k
                at time t (= -1 by default)
            """
            sim_k = Simulation(_seq_without_k(self, k), self.sampling_rate)
            results_k = sim_k.run()
            return Counter(results_k.sample_state(t, meas_basis, N_samples))

        def _add_atom_k(self, counter_k_missing, k):
            """
                Args :
                counter_k_missing is a dictionary of bitstrings of length N-1,
                corresponding to simulations run without atom k
                Returns the dictionary corresponding to the detection of atom k
                in states g or r (ground_rydberg for now), with probability
                epsilon to be measured as r, 1-epsilon to be measured as g
            """
            counter_k_added = Counter()
            for b_k, v in counter_k_missing.items():
                bitstring_0 = b_k[:k] + str(0) + b_k[k:]
                bitstring_1 = b_k[:k] + str(1) + b_k[k:]
                counter_k_added[bitstring_0] += (1-eps) * v
                counter_k_added[bitstring_1] += eps * v
            return counter_k_added

        def _build_p_faulty(self):
            """
                Builds the Counter for all faulty atoms, not yet considering
                ideal runs.
            """
            prob_faulty = Counter()
            for k in range(N):
                counter_k_missing = _evolve_without_k(self, k)
                counter_k_added = _add_atom_k(self, counter_k_missing, k)
                prob_faulty += counter_k_added
            # Going from number to probability
            for k, v in prob_faulty.items():
                prob_faulty[k] /= (N * N_samples)
            return prob_faulty

        def _build_total_p(self):
            """
                Returns the total probability dictionary, counting errors and
                no prep errors situations
                First order corrections : if one atom is faulty, we don't count
                detection errors for other atoms, as that would O(eta*epsilon)
                (1/10000)
            """
            no_prep_errors_results = self.run()
            prob_no_prep_errors = \
                no_prep_errors_results.sampling_with_detection_errors(t=t,
                                                        meas_basis=meas_basis,
                                                                      spam=spam
                                                                      )
            prob_total = Counter()
            # Can't simulate an empty register... The method is for 1 qubit
            if N == 1:
                prob_total["0"] = eta * (1 - eps) + (1 - eta) * \
                    (prob_no_prep_errors["0"])
                prob_total["1"] = eta * eps + (1 - eta) * \
                    (prob_no_prep_errors["1"])
                return prob_total
            # From now on : several qubits
            prob_faulty = _build_p_faulty(self)
            for k in prob_faulty.keys():
                prob_faulty[k] *= eta
            for k in prob_no_prep_errors.keys():
                prob_no_prep_errors[k] *= (1-eta)
            prob_total = prob_faulty + prob_no_prep_errors
            return prob_total

        return _build_total_p(self)
