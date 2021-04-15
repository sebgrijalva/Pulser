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
    Results of a noisy simulation run of a pulse sequence.

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

    def expect(self, obs_list):
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (array-like of qutip.Qobj or array-like of numpy.ndarray):
                A list of observables whose expectation value will be
                calculated. If necessary, each member will be transformed into
                a qutip.Qobj instance.
        """
        def _proj_from_bitstring(bitstring):
            proj = qutip.tensor([self.basis[i] * self.basis[i].dag() for i
                                 in bitstring])
            return proj

        # To calculate expectation values with QuTiP
        density_matrix = 0
        for b, v in self._states.items():
            density_matrix += v * _proj_from_bitstring(b)
        if not isinstance(obs_list, (list, np.ndarray)):
            raise TypeError("`obs_list` must be a list of operators")

        qobj_list = []
        for obs in obs_list:
            if not (isinstance(obs, np.ndarray)
                    or isinstance(obs, qutip.Qobj)):
                raise TypeError("Incompatible type of observable.")
            if obs.shape != (self._dim**self._size, self._dim**self._size):
                raise ValueError("Incompatible shape of observable.")
            # Transfrom to qutip.Qobj and take dims from state
            # dim_list = [self._dim, self._dim]
            qobj_list.append(qutip.Qobj(obs))

        return [qutip.expect(qobj, density_matrix) for qobj in qobj_list]

    def sample_state(self, t=-1, meas_basis='ground-rydberg', N_samples=1000):
        r"""Returns the result of multiple measurements in a given basis.

        The encoding of the results depends on the meaurement basis. Namely:

        - *ground-rydberg* : :math:`1 = |r\rangle;~ 0 = |g\rangle, |h\rangle`
        - *digital* : :math:`1 = |h\rangle;~ 0 = |g\rangle, |r\rangle`

        Note:
            The results are presented using a big-endian representation,
            according to the pre-established qubit ordering in the register.
            This means that, when sampling a register with qubits ('q0','q1',
            ...), in this order, the corresponding value, in binary, will be
            0Bb0b1..., where b0 is the outcome of measuring 'q0', 'b1' of
            measuring 'q1' and so on.

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
        N = self._size
        self.N_samples = N_samples
        bitstrings = [np.binary_repr(k, N) for k in range(2**N)]
        probs = [self._states[b] for b in bitstrings]
        if meas_basis is None:
            if self._meas_basis is None:
                raise ValueError(
                    "Can't accept an undefined measurement basis because the "
                    "original sequence has no measurement."
                    )
            meas_basis = self._meas_basis

        if meas_basis not in {'ground-rydberg', 'digital'}:
            raise ValueError(
                "'meas_basis' can only be 'ground-rydberg' or 'digital'."
                )
        if self._dim == 2:
            if meas_basis == self._basis_name:
                # State vector ordered with r first for 'ground_rydberg'
                # e.g. N=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
                # Invert the order ->  [00, 01, 10, 11] correspondence
                # Verified : order already reversed in clean_results when
                # producing a NoiseResults !
                weights = probs[::-1] if meas_basis == 'digital' else probs
            else:
                return {'0' * N: int(N_samples)}

        elif self._dim == 3:
            if meas_basis == 'ground-rydberg':
                one_state = 0       # 1 = |r>
                ex_one = slice(1, 3)
            elif meas_basis == 'digital':
                one_state = 2       # 1 = |h>
                ex_one = slice(0, 2)

            probs = probs.reshape(tuple([3 for _ in range(N)]))
            weights = []

            for dec_val in range(2**N):
                ind = []
                for v in np.binary_repr(dec_val, width=N):
                    if v == '0':
                        ind.append(ex_one)
                    else:
                        ind.append(one_state)
                # Eg: 'digital' basis => |1> = index 2, |0> = index 0, 1 = 0:2
                # p_11010 = sum(probs[2, 2, 0:2, 2, 0:2])
                # We sum all probabilites that correspond to measuring 11010,
                # namely hhghg, hhrhg, hhghr, hhrhr
                weights.append(np.sum(probs[tuple(ind)]))
        else:
            raise NotImplementedError(
                "Cannot sample system with single-atom state vectors of "
                "dimension > 3."
                )
        dist = np.random.multinomial(N_samples, weights)
        return Counter(
               {np.binary_repr(i, N): dist[i] for i in np.nonzero(dist)[0]})

    def sample_final_state(self, meas_basis='ground-rydberg', N_samples=1000):
        return self.sample_state(-1, meas_basis, N_samples)
