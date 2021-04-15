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


class SimulationResults:
    """Results of a simulation run of a pulse sequence. Parent class for
    NoisyResults and CleanResults.

    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, run_output, dim, size, basis_name,
                 meas_basis="ground-rydberg"):
        """Initializes a new SimulationResults instance.

        Args:
            run_output: Depends on whether the results are clean or noisy.
            dim (int): The dimension of the local space of each atom (2 or 3).
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').

        Keyword Args:
            meas_basis (None or str): The basis in which a sampling measurement
                is desired.
        """
        self._states = run_output
        self._dim = dim
        self._size = size
        if basis_name not in {'ground-rydberg', 'digital', 'all'}:
            raise ValueError(
                "`basis_name` must be 'ground-rydberg', 'digital' or 'all'."
                )
        self._basis_name = basis_name
        if meas_basis:
            if meas_basis not in {'ground-rydberg', 'digital'}:
                raise ValueError(
                    "'meas_basis' must be 'ground-rydberg' or 'digital'."
                    )
        self._meas_basis = meas_basis
        self._build_basis()

    def expect(self, obs_list):
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (array-like of qutip.Qobj or array-like of numpy.ndarray):
                A list of observables whose expectation value will be
                calculated. If necessary, each member will be transformed into
                a qutip.Qobj instance.
        """
        pass

    def sample_state(self, t=-1, meas_basis='ground-rydberg', N_samples=1000):
        r"""Returns the result of multiple measurements in a given basis.

        The enconding of the results depends on the meaurement basis. Namely:

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

        Raises:
            ValueError: If trying to sample without a defined 'meas_basis' in
                the arguments when the original sequence is not measured.
        """
        pass

    def _build_basis(self):
        """Determine dimension and basis in 0 and 1
        notation depending on the measurement basis"""
        basis = []
        if (self._meas_basis == "ground-rydberg"):
            basis = ['r', 'g']
        elif (self._meas_basis == "digital"):
            basis = ['g', 'h']
        # verified
        self.basis = {
            str(i): qutip.basis(self._dim, 1-i) for i, b in enumerate(basis)}
