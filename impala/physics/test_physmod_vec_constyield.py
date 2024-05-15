#!/usr/bin/env python3

import physical_models_vec as pmv
import unittest
import numpy as np


# ------------------------------------------------------------------------------
# TestConstantYieldStress
# Constant yield stress model.
# ------------------------------------------------------------------------------

class TestConstantYieldStress(unittest.TestCase):

    # Functions:
    # setUp
    # test_isothermal_lowrate
    # test_adiabatic_highrate
    # test_constant_stress

    # ------------------------
    # setUp
    # ------------------------

    def setUp(self):
        self.params = {}
        self.consts = {
            'yield_stress' : 0.1,      # MBar
            'rho0'         : 8.96,     # g/cc
            'Cv0'          : 0.383e-5, # MBar*cc/g*K
            'G0'           : 0.4578,   # MBar
            'Tmelt0'       : 1356.0,   # K
            'chi'          : 1.0
        }
        '''
        self.model_const_y = pmh.MaterialModel()
        '''
        self.model_vec_const_y = pmv.MaterialModel()

    # ------------------------
    # test_isothermal_lowrate
    # ------------------------

    def test_isothermal_lowrate(self):
        """
        Const. yield, g0, Tm. Rates less than 1e-6/us (~1/s),
        should be isothermal T=const
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e-7
        nhist_pm = 1000

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_const_y.initialize(self.params, self.consts)
        self.model_const_y.initialize_state(T=298.)
        results_const_y = self.model_const_y.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec  = np.array([emax_pm])
        edot_vec  = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_const_y.initialize(self.params, self.consts)
        self.model_vec_const_y.initialize_state(T=np.array([298.]))
        results_vec_const_y = \
            self.model_vec_const_y.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_const_y[ 0][3][0],
            results_vec_const_y[-1][3][0]
        ))
        '''
        self.assertEqual(
            results_const_y    [ 0][3], results_vec_const_y[ 0][3])
        self.assertEqual(
            results_const_y    [-1][3], results_vec_const_y[-1][3])
        '''
        self.assertEqual(
            results_vec_const_y[ 0][3], results_vec_const_y[-1][3])

    # ------------------------
    # test_adiabatic_highrate
    # ------------------------

    def test_adiabatic_highrate(self):
        """
        Const. yield, g0, Tm. Rates greater than 1e-6/us (~1/s),
        temperature changes adiabatically
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e0
        nhist_pm = 1000

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_const_y.initialize(self.params, self.consts)
        self.model_const_y.initialize_state(T=298.)
        results_const_y = self.model_const_y.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec  = np.array([emax_pm])
        edot_vec  = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_const_y.initialize(self.params, self.consts)
        self.model_vec_const_y.initialize_state(T=np.array([298.]))
        results_vec_const_y = \
            self.model_vec_const_y.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_const_y[ 0][3][0],
            results_vec_const_y[-1][3][0]
        ))
        '''
        self.assertEqual(
            results_const_y    [ 0][3], results_vec_const_y[ 0][3])
        self.assertEqual(
            results_const_y    [-1][3], results_vec_const_y[-1][3])
        '''
        self.assertNotEqual(
            results_vec_const_y[ 0][3], results_vec_const_y[-1][3])

    # ------------------------
    # test_constant_stress
    # ------------------------

    def test_constant_stress(self):
        """
        Const. yield, g0, Tm. Yield stress is constant for all strain
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e0
        nhist_pm = 1000

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_const_y.initialize(self.params, self.consts)
        self.model_const_y.initialize_state(T=298.)
        results_const_y = self.model_const_y.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec  = np.array([emax_pm])
        edot_vec  = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_const_y.initialize(self.params, self.consts)
        self.model_vec_const_y.initialize_state(T=np.array([298.]))
        results_vec_const_y = \
            self.model_vec_const_y.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
            results_vec_const_y[ 0][1][0], results_vec_const_y[ 0][2][0]*1e5,
            results_vec_const_y[-1][1][0], results_vec_const_y[-1][2][0]*1e5
        ))
        '''
        for time_i in results_const_y:
            self.assertEqual(self.consts['yield_stress'], time_i[2])
        '''
        for time_i in results_vec_const_y:
            self.assertEqual(self.consts['yield_stress'], time_i[2])
        '''
        np.testing.assert_allclose(results_const_y, results_vec_const_y[:,:,0])
        '''


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
