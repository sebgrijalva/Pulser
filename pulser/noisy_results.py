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

import qutip
import numpy as np
from pulser.simresults import SimulationResults

from collections import Counter


class NoisyResults(SimulationResults):
    """
    Results of a noisy simulation run of a pulse sequence. Contrary to a
    CleanResults object, this object contains a unique Counter describing the
    state distribution at the time it was created by using
    sim.run(spam=True, t).

    Contains methods for studying the populations and extracting useful
    information from them.
    """

    def __init__(self, run_output, dim, size, basis_name,
                 meas_basis="ground-rydberg"):
        """
        Initializes a new NoisyResults instance.

        Args:
            run_output (Counter) : Counter returning the population of each
                multi-qubits state, represented as a bitstring.
            dim (int): The dimension of the local space of each atom (2 or 3).
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').

        Keyword Args:
            meas_basis (None or str): The basis in which a sampling measurement
                is desired.
        """
        super().__init__(run_output, dim, size, basis_name, meas_basis)
        # Can't have more than 2 states : projections have already been applied
        # This is not the case for a CleanResults object, containing states in
        # Hilbert space, but NoisyResults contains a probability distribution
        # of bitstrings, not atomic states
        self._dim = 2

    def get_final_state(self):
        """Get the final state (density matrix here !) of the simulation.

        Returns:
            qutip.Qobj: The resulting final state as a density matrix.
        """
        def _proj_from_bitstring(bitstring):
            proj = qutip.tensor([self.basis[i] * self.basis[i].dag() for i
                                 in bitstring])
            return proj

        density_matrix = 0
        for b, v in self._states.items():
            density_matrix += v * _proj_from_bitstring(b)

        return density_matrix

    def expect(self, obs_list):
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (array-like of qutip.Qobj or array-like of numpy.ndarray):
                A list of observables whose expectation value will be
                calculated. If necessary, each member will be transformed into
                a qutip.Qobj instance.
        Returns:
            list: the list of expectation values of each operator.
        """

        density_matrix = self.get_final_state()
        if not isinstance(obs_list, (list, np.ndarray)):
            raise TypeError("`obs_list` must be a list of operators")

        qobj_list = []
        for obs in obs_list:
            if not (isinstance(obs, np.ndarray)
                    or isinstance(obs, qutip.Qobj)):
                raise TypeError("Incompatible type of observable.")
            if obs.shape != (2**self._size, 2**self._size):
                raise ValueError("Incompatible shape of observable.")
            qobj_list.append(qutip.Qobj(obs))

        return [qutip.expect(qobj, density_matrix) for qobj in qobj_list]

    def sample_state(self, meas_basis=None, N_samples=1000):
        r"""Returns the result of multiple measurements in a given basis.
        Keyword Args:
            meas_basis (str, default=None): 'ground-rydberg' or 'digital'. If
                left as None, uses the measurement basis defined in the
                original sequence.
            N_samples (int, default=1000): Number of samples to take.
            t (int, default=-1) : Time at which the system is measured.

        Raises:
            ValueError: If trying to sample without a defined 'meas_basis' in
                the arguments when the original sequence is not measured.
        """
        if meas_basis is None:
            if self._meas_basis is None:
                raise ValueError(
                    "Can't accept an undefined measurement basis because the "
                    "original sequence has no measurement."
                    )
            meas_basis = self._meas_basis

        if meas_basis not in {'ground-rydberg', 'digital'}:
            raise ValueError(
                "`meas_basis` can only be 'ground-rydberg' or 'digital'."
                )

        N = self._size
        self.N_samples = N_samples
        bitstrings = [np.binary_repr(k, N) for k in range(2**N)]
        probs = [self._states[b] for b in bitstrings]

        # State vector ordered with r first for 'ground_rydberg'
        # e.g. N=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
        # Invert the order ->  [00, 01, 10, 11] correspondence
        # VERIFIED : order already reversed in detection_SPAM when
        # producing a NoiseResult !
        weights = probs[::-1] if meas_basis == 'digital' else probs

        dist = np.random.multinomial(N_samples, weights)
        return Counter(
               {np.binary_repr(i, N): dist[i] for i in np.nonzero(dist)[0]})

    def sample_final_state(self, meas_basis='ground-rydberg', N_samples=1000):
        return self.sample_state(meas_basis, N_samples)
