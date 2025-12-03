import unittest

import numpy as np

from impala.physics import physical_models_vec as pmh


# Constant yield stress model
class TestConstantYieldStress(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.consts = {
            "yield_stress": 0.1,  # MBar
            "rho0": 8.96,  # g/cc
            "Cv0": 0.383e-5,  # MBar*cc/g*K
            "G0": 0.4578,  # MBar
            "Tmelt0": 1356.0,  # K
            "chi": 1.0,
        }
        self.model_const_y = pmh.MaterialModel()

    def test_isothermal_lowrate(self):
        """
        Const. yield, g0, Tm. Rates less than 1e-6/us (~1/s), should be isothermal T=const
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-7]), nhist=1000
        )
        self.model_const_y.initialize(self.params, self.consts)
        self.model_const_y.initialize_state(T=np.array([298.0]))
        results_const_y = self.model_const_y.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertEqual(results_const_y[0][3], results_const_y[-1][3])

    def test_adiabatic_highrate(self):
        """
        Const. yield, g0, Tm. Rates greater than 1e-6/us (~1/s), temperature changes adiabatically
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e0]), nhist=1000
        )
        self.model_const_y.initialize(self.params, self.consts)
        self.model_const_y.initialize_state(T=np.array([298.0]))
        results_const_y = self.model_const_y.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertNotEqual(results_const_y[0][3], results_const_y[-1][3])

    def test_constant_stress(self):
        """
        Const. yield, g0, Tm. Yield stress is constant for all strain
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e0]), nhist=1000
        )
        self.model_const_y.initialize(self.params, self.consts)
        self.model_const_y.initialize_state(T=np.array([298.0]))
        results_const_y = self.model_const_y.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        for time_i in results_const_y:
            self.assertEqual(self.consts["yield_stress"], time_i[2])
