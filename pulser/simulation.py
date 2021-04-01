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

from pulser import Pulse, Sequence
from pulser.simresults import SimulationResults


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

    def __init__(self, sequence, sampling_rate=1.0, interaction='ising'):
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

        self._interaction = interaction

        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}

        if self._interaction == 'ising':
            self.samples = {addr: {basis: {}
                                   for basis in ['ground-rydberg', 'digital']}
                            for addr in ['Global', 'Local']}
        elif self._interaction == 'XY':
            self.samples = {'Global': {}, 'Local': {}}
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
            samples_dict['det'][slot.ti:slot.tf] += slot.type.detuning.samples
            samples_dict['phase'][slot.ti:slot.tf] = slot.type.phase

        if self._interaction == 'ising':
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
        elif self._interaction == 'XY':
            for channel in self._seq.declared_channels:
                addr = self._seq.declared_channels[channel].addressing
                basis = self._seq.declared_channels[channel].basis

                if basis == 'digital':
                    raise ValueError("Only Rydberg basis allowed for XY mode")

                samples_dict = self.samples[addr]

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

                self.samples[addr] = samples_dict

    def _build_basis_and_op_matrices(self):
        """Determine dimension, basis and projector operators."""
        if self._interaction == 'ising':
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
                projectors = ['gr', 'hg', 'rr', 'gg', 'hh']

            self.basis = {b: qutip.basis(self.dim, i)
                          for i, b in enumerate(basis)}
            self.op_matrix = {'I': qutip.qeye(self.dim)}

        elif self._interaction == 'XY':
            # No samples => Empty dict entry => False
            self.basis_name = 'ryd-ryd'
            self.dim = 2
            basis = ['u', 'd']
            projectors = ['du', 'ud', 'uu', 'dd']
            self.basis = {'u': qutip.basis(self.dim, 0),
                          'd': qutip.basis(self.dim, 1)}

            self.op_matrix = {'I': qutip.qeye(self.dim), 'Z': qutip.sigmaz()}
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

    def _build_multi_operator(self, op_ids, qubit_ids):
        """Create qutip.Qobj acting nontrivially at each of `qubit_ids` with
        each of `op_ids` (one to one correspondence, so `op_ids` would include
        repetitions)"""
        if len(op_ids) != len(qubit_ids):
            raise ValueError("Operator list and qubit list must be 1-to-1.")
        # List of identity operators, except for op_id where requested:
        qubit_index_list = list(map(self._qid_index.get, qubit_ids))
        op_list = [self.op_matrix[op_ids[qubit_index_list.index(j)]]
                   if j in qubit_index_list
                   else self.op_matrix['I'] for j in range(self._size)]
        return qutip.tensor(op_list)

    def _construct_hamiltonian(self):
        def adapt(full_array):
            """Adapt list to correspond to sampling rate"""
            indexes = np.linspace(0, self._tot_duration-1,
                                  int(self.sampling_rate*self._tot_duration),
                                  dtype=int)
            return full_array[indexes]

        def make_interaction_term():
            """Construct the interaction Term.

            If the interaction is 'ising':
            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_rr" at each pair. The units are
            given so that the coefficient includes a 1/hbar factor.

            If the interaction is 'XY':
            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_ud + sigma_du" at each pair.
            Units are such that the coefficient includes a 1/hbar factor.
            """
            term = 0
            # Get every pair without duplicates
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                dist = np.linalg.norm(
                    self._qdict[q1] - self._qdict[q2])
                if self._interaction == 'ising':
                    U = 0.5 * self._seq._device.interaction_coeff / dist**6
                    term += U * self._build_operator('sigma_rr', q1, q2)
                elif self._interaction == 'XY':
                    U = (self._seq._device.interaction_coeff)**(1/3) / dist**3
                    term += U * self._build_multi_operator(
                                        ['sigma_ud', 'sigma_du'], [q1, q2])
            return term

        def build_coeffs_ops(basis, addr):
            """Build coefficients and operators for the hamiltonian QobjEvo."""

            if self._interaction == 'ising':
                samples = self.samples[addr][basis]
                operators = self.operators[addr][basis]
            elif self._interaction == 'XY':
                samples = self.samples[addr]
                operators = self.operators[addr]
            # Choose operator names according to addressing:
            if basis == 'ground-rydberg':
                op_ids = ['sigma_gr', 'sigma_rr']
            elif basis == 'digital':
                op_ids = ['sigma_hg', 'sigma_gg']
            elif basis == 'ryd-ryd':
                op_ids = ['sigma_du', 'Z']

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
            if self._interaction == 'ising':
                self.operators[addr][basis] = operators
            elif self._interaction == 'XY':
                self.operators[addr] = operators
            return terms

        # Time independent term:
        if self.basis_name == 'digital':
            qobj_list = []
        else:
            # Add Interaction Terms
            qobj_list = [make_interaction_term()] if self._size > 1 else []

        # Time dependent terms:
        for addr in self.samples:
            if self._interaction == 'ising':
                for basis in self.samples[addr]:
                    if self.samples[addr][basis]:
                        qobj_list += build_coeffs_ops(basis, addr)
            elif self._interaction == 'XY':
                if self.samples[addr]:
                    qobj_list += build_coeffs_ops('ryd-ryd', addr)

        self._times = adapt(np.arange(self._tot_duration,
                                      dtype=np.double)/1000)

        ham = qutip.QobjEvo(qobj_list, tlist=self._times)
        ham = ham + ham.dag()
        ham.compress()

        self._hamiltonian = ham

    # Run Simulation Evolution using Qutip
    def run(self, initial_state=None, progress_bar=None, **options):
        """Simulate the sequence using QuTiP's solvers.

        Keyword Args:
            initial_state (array): The initial quantum state of the
                           evolution. Will be transformed into a
                           qutip.Qobj instance.
            progress_bar (bool): If True, the progress bar of QuTiP's sesolve()
                        will be shown.

        Returns:
            SimulationResults: Object containing the time evolution results.
        """
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
            tag = 'g' if self._interaction == 'ising' else 'd'
            self._initial_state = qutip.tensor([self.basis[tag]
                                                for _ in range(self._size)])

        result = qutip.sesolve(self._hamiltonian,
                               self._initial_state,
                               self._times,
                               progress_bar=progress_bar,
                               options=qutip.Options(max_step=5,
                                                     **options)
                               )

        if hasattr(self._seq, '_measurement'):
            meas_basis = self._seq._measurement
        else:
            meas_basis = None

        return SimulationResults(
            result.states, self.dim, self._size, self.basis_name,
            meas_basis=meas_basis
        )
