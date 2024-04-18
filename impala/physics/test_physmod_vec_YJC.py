#!/usr/bin/env python3
from py_calibration_hier import physical_models as pmh
from py_calibration_hier import physical_models_vec_devin as pmv
import unittest
import numpy as np
import statistics as stat
import sys

# Constant yield stress model
class TestJCYieldStress(unittest.TestCase):
    def setUp(self):
        """
        Parameters are for OFHC copper.
        """
        self.params = {
            'A' : 0.00090,  # MBar
            'B' : 0.00292,  # MBar
            'C' : 0.0250,   # -
            'n' : 0.31,     # -
            'm' : 1.09,     # -
            }
        self.consts = {
            'rho0'          : 8.9375, # g/cc
            'Cv0'           : 0.383e-5, # MBar*cc/g*K
            'G0'            : 0.4578, # MBar
            'Tmelt0'        : 1356.0, # K
            'Tref'          : 298.0, # K
            'edot0'         : 1.0e-6, # 1/micro-s
            'chi'           : 1.0,
            }
        self.model_yjc = pmh.MaterialModel(
            flow_stress_model = pmh.JC_Yield_Stress,
            )
        self.model_vec_yjc = pmv.MaterialModel(
            flow_stress_model = pmv.JC_Yield_Stress,
            )

    def test_isothermal_lowrate(self):
        """
        JC, const. g0, Tm. Rates less than 1e-6/us (~1/s), should be isothermal T=const
        """
        # Get results for original physical models code
        emax_pm = 1.0
        edot_pm = 1.0e-7
        nhist_pm = 1000
        shist = pmh.generate_strain_history(emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=298.)
        results_yjc = self.model_yjc.compute_state_history(shist)

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_yjc.initialize(self.params, self.consts)
        self.model_vec_yjc.initialize_state(T=298.)
        results_vec_yjc = self.model_vec_yjc.compute_state_history(shist_vec)
        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(results_vec_yjc[0][3][0],results_vec_yjc[-1][3][0]))
        self.assertEqual(results_yjc[0][3],results_vec_yjc[0][3])
        self.assertEqual(results_yjc[-1][3],results_vec_yjc[-1][3])
        self.assertEqual(results_vec_yjc[0][3],results_vec_yjc[-1][3])

    def test_adiabatic_highrate(self):
        """
        JC, const. g0, Tm. Rates greater than 1e-6/us (~1/s), temperature changes adiabatically
        """
        # Get results for original physical models code
        emax_pm = 1.0
        edot_pm = 1.0e-2
        nhist_pm = 1000
        shist = pmh.generate_strain_history(emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=298.)
        results_yjc = self.model_yjc.compute_state_history(shist)

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_yjc.initialize(self.params, self.consts)
        self.model_vec_yjc.initialize_state(T=298.)
        results_vec_yjc = self.model_vec_yjc.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(results_vec_yjc[0][3][0],results_vec_yjc[-1][3][0]))
        self.assertEqual(results_yjc[0][3],results_vec_yjc[0][3])
        self.assertEqual(results_yjc[-1][3],results_vec_yjc[-1][3])
        self.assertNotEqual(results_vec_yjc[0][3],results_vec_yjc[-1][3])

    def test_stress_highrate(self):
        """
        JC, constant g0 and Tm, high rate 1e4/s, compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.28179, flow_stress =  343.37 MPa
        tolerance set on original code is within 2%

        Vectorized physical models code should produce same result as original
        code.
        """
        emax_pm = 0.28179
        edot_pm = 1.e-2
        nhist_pm = 1000
        # Get results for original physical models code
        shist = pmh.generate_strain_history(emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=298.)
        results_yjc = self.model_yjc.compute_state_history(shist)

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_yjc.initialize(self.params, self.consts)
        self.model_vec_yjc.initialize_state(T=298.)
        results_vec_yjc = self.model_vec_yjc.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                results_vec_yjc[0][1][0],results_vec_yjc[0][2][0]*1e5,
                results_vec_yjc[-1][1][0],results_vec_yjc[-1][2][0]*1e5))
        np.testing.assert_allclose(results_yjc,results_vec_yjc[:,:,0])

    def test_stress_midrate(self):
        """
        JC, constant g0 and Tm, mid rate 1e2/s, compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.28895, flow_stress =  317.06 MPa
        tolerance set on original code is within 1%

        Vectorized physical models code should produce same result as original
        code.
        """
        emax_pm = 0.28895
        edot_pm = 1.e-4
        nhist_pm = 1000
        # Get results for original physical models code
        shist = pmh.generate_strain_history(emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=298.)
        results_yjc = self.model_yjc.compute_state_history(shist)

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_yjc.initialize(self.params, self.consts)
        self.model_vec_yjc.initialize_state(T=298.)
        results_vec_yjc = self.model_vec_yjc.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                results_vec_yjc[0][1][0],results_vec_yjc[0][2][0]*1e5,
                results_vec_yjc[-1][1][0],results_vec_yjc[-1][2][0]*1e5))
        np.testing.assert_allclose(results_yjc,results_vec_yjc[:,:,0])

    def test_stress_lowrate(self):
        """
        JC, constant g0 and Tm, low rate 1e-1/s, compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-7(micro-s)^-1 = 1e-1(s)^-1

        FLAG sim results
        eps_p = 0.28928, flow_stress =  284.89 MPa
        tolerance set on original code is within 5%

        Vectorized physical models code should produce same result as original
        code.
        """
        emax_pm = 0.28928
        edot_pm = 1.e-7
        nhist_pm = 1000
        # Get results for original physical models code
        shist = pmh.generate_strain_history(emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=298.)
        results_yjc = self.model_yjc.compute_state_history(shist)

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_yjc.initialize(self.params, self.consts)
        self.model_vec_yjc.initialize_state(T=298.)
        results_vec_yjc = self.model_vec_yjc.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                results_vec_yjc[0][1][0],results_vec_yjc[0][2][0]*1e5,
                results_vec_yjc[-1][1][0],results_vec_yjc[-1][2][0]*1e5))
        np.testing.assert_allclose(results_yjc,results_vec_yjc[:,:,0])
if __name__ == '__main__':
    unittest.main()
