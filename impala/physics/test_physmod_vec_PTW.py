#!/usr/bin/env python3

import physical_models_vec as pmv
import unittest
import numpy as np


# ------------------------------------------------------------------------------
# TestPTWYieldStress_Constg0Tm
# Constant yield stress model.
# ------------------------------------------------------------------------------

class TestPTWYieldStress_Constg0Tm(unittest.TestCase):

    # Functions:
    # setUp
    # test_isothermal_lowrate
    # test_adiabatic_highrate
    # test_stress_highrate
    # test_stress_midhighrate
    # test_stress_midlowrate
    # test_stress_lowrate

    # ------------------------
    # setUp
    # ------------------------

    def setUp(self):
        """
        Parameters are for OFHC copper.
        """

        self.params = {
            # PTW
            'theta' : 0.025,
            'p'     : 2.,
            's0'    : 0.0085,
            'sInf'  : 0.00055,
            'kappa' : 0.11,
            'gamma' : 1.e-5,
            'y0'    : 0.0001,
            'yInf'  : 0.0001
        }

        self.params_vec = {
            # PTW
            'theta'  : 0.025,
            'p'      : 2.,
            's0'     : 0.0085,
            'sInf'   : 0.00055,
            'kappa'  : 0.11,
            'lgamma' : np.log(1e-5),
            'y0'     : 0.0001,
            'yInf'   : 0.0001,
            'y1'     : 0.094,
            'y2'     : 0.575,
        }

        self.consts = {
            # PTW
            'y1'     : 0.094,
            'y2'     : 0.575,
            'beta'   : 0.25,
            'matomic': 63.546,
            'chi'    : 1.0,
            # Constant Spec. Heat
            'Cv0'    : 0.383e-5,
            # Constant Density
            'rho0'   : 8.9375,
            # Constant Melt Temp.
            'Tmelt0' : 1625.,
            # # Constant Shear Mod.
            'G0'     : 0.4578,
            # # Simple Shear Mod.
            # 'G0'     : 0.50889, # Cold shear
            # 'alpha'  : 0.21
            # # SG Shear Mod.
            # 'G0'     : 0.4578, # MBar, 300K Shear mod.
            # 'sgB'    : 3.8e-4, # K^-1
        }

        self.consts_vec = {
            # PTW
            'beta'   : 0.25,
            'matomic': 63.546,
            'chi'    : 1.0,
            # Constant Spec. Heat
            'Cv0'    : 0.383e-5,
            # Constant Density
            'rho0'   : 8.9375,
            # Constant Melt Temp.
            'Tmelt0' : 1625.,
            # # Constant Shear Mod.
            'G0'     : 0.4578,
            # # Simple Shear Mod.
            # 'G0'     : 0.50889, # Cold shear
            # 'alpha'  : 0.21
            # # SG Shear Mod.
            # 'G0'     : 0.4578, # MBar, 300K Shear mod.
            # 'sgB'    : 3.8e-4, # K^-1
        }

        '''
        self.model_ptw_cg0Tm = pmh.MaterialModel(
            flow_stress_model = pmh.PTW_Yield_Stress,
        )
        '''
        self.model_vec_ptw_cg0Tm = pmv.MaterialModel(
            flow_stress_model = pmv.PTW_Yield_Stress,
        )

    # ------------------------
    # test_isothermal_lowrate
    # ------------------------

    def test_isothermal_lowrate(self):
        """
        PTW, const. g0, Tm. Rates less than 1e-6/us (~1/s),
        should be isothermal T=const
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e-7
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=298.)
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_cg0Tm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_cg0Tm.initialize_state(T=np.array([298.]))
        results_vec_ptw_cg0Tm = \
            self.model_vec_ptw_cg0Tm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_ptw_cg0Tm[ 0][3][0],
            results_vec_ptw_cg0Tm[-1][3][0]
        ))
        '''
        self.assertEqual(results_ptw_cg0Tm[ 0][3], results_vec_ptw_cg0Tm[ 0][3])
        self.assertEqual(results_ptw_cg0Tm[-1][3], results_vec_ptw_cg0Tm[-1][3])
        '''
        self.assertEqual(
            results_vec_ptw_cg0Tm[ 0][3],
            results_vec_ptw_cg0Tm[-1][3]
        )

    # ------------------------
    # test_adiabatic_highrate
    # ------------------------

    def test_adiabatic_highrate(self):
        """
        PTW, const. g0, Tm. Rates greater than 1e-6/us (~1/s),
        temp. changes adiabatically
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e-2
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=298.)
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_cg0Tm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_cg0Tm.initialize_state(T=np.array([298.]))
        results_vec_ptw_cg0Tm = \
            self.model_vec_ptw_cg0Tm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_ptw_cg0Tm[ 0][3][0],
            results_vec_ptw_cg0Tm[-1][3][0]
        ))
        '''
        self.assertEqual(
            results_ptw_cg0Tm    [ 0][3],
            results_vec_ptw_cg0Tm[ 0][3]
        )
        np.testing.assert_allclose(
            results_ptw_cg0Tm    [-1][3],
            results_vec_ptw_cg0Tm[-1][3]
        )
        '''
        self.assertNotEqual(
            results_vec_ptw_cg0Tm[ 0][3],
            results_vec_ptw_cg0Tm[-1][3]
        )

    # ------------------------
    # test_stress_highrate
    # ------------------------

    def test_stress_highrate(self):
        """
        PTW, constant g0 and Tm, high rate 1e6/s, compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e0(micro-s)^-1 = 1e6(s)^-1

        FLAG sim results
        eps_p = 0.244376, flow_stress =  521.4 MPa
        tolerance set on original code is within 1%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.244376
        edot_pm  = 1e0
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=298.)
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_cg0Tm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_cg0Tm.initialize_state(T=np.array([298.]))
        results_vec_ptw_cg0Tm = \
            self.model_vec_ptw_cg0Tm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_cg0Tm[ 0][1][0],
                  results_vec_ptw_cg0Tm[ 0][2][0]*1e5,
                  results_vec_ptw_cg0Tm[-1][1][0],
                  results_vec_ptw_cg0Tm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_cg0Tm,
            results_vec_ptw_cg0Tm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_midhighrate
    # ------------------------

    def test_stress_midhighrate(self):
        """
        PTW, constant g0 and Tm, mid-high rate 1e4/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.382749, flow_stress =  458.198 MPa
        tolerance set on original code is within 1%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.382749
        edot_pm  = 1e-2
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=298.)
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_cg0Tm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_cg0Tm.initialize_state(T=np.array([298.]))
        results_vec_ptw_cg0Tm = \
            self.model_vec_ptw_cg0Tm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_cg0Tm[ 0][1][0],
                  results_vec_ptw_cg0Tm[ 0][2][0]*1e5,
                  results_vec_ptw_cg0Tm[-1][1][0],
                  results_vec_ptw_cg0Tm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_cg0Tm,
            results_vec_ptw_cg0Tm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_midlowrate
    # ------------------------

    def test_stress_midlowrate(self):
        """
        PTW, constant g0 and Tm, low-mid rate 1e2/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.388339, flow_stress =  418.512 MPa
        tolerance set on original code is within 11%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.388339
        edot_pm  = 1e-4
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=298.)
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_cg0Tm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_cg0Tm.initialize_state(T=np.array([298.]))
        results_vec_ptw_cg0Tm = \
            self.model_vec_ptw_cg0Tm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_cg0Tm[ 0][1][0],
                  results_vec_ptw_cg0Tm[ 0][2][0]*1e5,
                  results_vec_ptw_cg0Tm[-1][1][0],
                  results_vec_ptw_cg0Tm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_cg0Tm,
            results_vec_ptw_cg0Tm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_lowrate
    # ------------------------

    def test_stress_lowrate(self):
        """
        PTW, constant g0 and Tm, low rate 1e0/s, compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-6(micro-s)^-1 = 1e0(s)^-1

        FLAG sim results
        eps_p = 0.371143, flow_stress =  385.751 MPa
        tolerance set on original code is within 11%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.371143
        edot_pm  = 1e-6
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=298.)
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_cg0Tm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_cg0Tm.initialize_state(T=np.array([298.]))
        results_vec_ptw_cg0Tm = \
            self.model_vec_ptw_cg0Tm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_cg0Tm[ 0][1][0],
                  results_vec_ptw_cg0Tm[ 0][2][0]*1e5,
                  results_vec_ptw_cg0Tm[-1][1][0],
                  results_vec_ptw_cg0Tm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_cg0Tm,
            results_vec_ptw_cg0Tm[:,:,0]
        )
        '''


# ------------------------------------------------------------------------------
# TestPTWYieldStress_SimpShearConstTm
# ------------------------------------------------------------------------------

class TestPTWYieldStress_SimpShearConstTm(unittest.TestCase):

    # Functions:
    # setUp
    # test_isothermal_lowrate
    # test_adiabatic_highrate
    # test_stress_highrate
    # test_stress_midhighrate
    # test_stress_midlowrate
    # test_stress_lowrate

    # ------------------------
    # setUp
    # ------------------------

    def setUp(self):
        """
        Parameters are for OFHC copper.
        """

        self.params = {
            # PTW
            'theta' : 0.025,
            'p'     : 2.,
            's0'    : 0.0085,
            'sInf'  : 0.00055,
            'kappa' : 0.11,
            'gamma' : 1.e-5,
            'y0'    : 0.0001,
            'yInf'  : 0.0001
        }

        self.params_vec = {
            # PTW
            'theta'  : 0.025,
            'p'      : 2.,
            's0'     : 0.0085,
            'sInf'   : 0.00055,
            'kappa'  : 0.11,
            'lgamma' : np.log(1e-5),
            'y0'     : 0.0001,
            'yInf'   : 0.0001,
            'y1'     : 0.094,
            'y2'     : 0.575,
        }

        self.consts = {
            # PTW
            'y1'     : 0.094,
            'y2'     : 0.575,
            'beta'   : 0.25,
            'matomic': 63.546,
            'chi'    : 1.0,
            # Constant Spec. Heat
            'Cv0'    : 0.383e-5,
            # Constant Density
            'rho0'   : 8.9375,
            # Constant Melt Temp.
            'Tmelt0' : 1625.,
            # # Constant Shear Mod.
            # 'G0'     : 0.4578,
            # Simple Shear Mod.
            'G0'     : 0.50889, # Cold shear
            'alpha'  : 0.21
            # # SG Shear Mod.
            # 'G0'     : 0.4578, # MBar, 300K Shear mod.
            # 'sgB'    : 3.8e-4, # K^-1
        }

        self.consts_vec = {
            # PTW
            'beta'   : 0.25,
            'matomic': 63.546,
            'chi'    : 1.0,
            # Constant Spec. Heat
            'Cv0'    : 0.383e-5,
            # Constant Density
            'rho0'   : 8.9375,
            # Constant Melt Temp.
            'Tmelt0' : 1625.,
            # # Constant Shear Mod.
            # 'G0'     : 0.4578,
            # Simple Shear Mod.
            'G0'     : 0.50889, # Cold shear
            'alpha'  : 0.21
            # # SG Shear Mod.
            # 'G0'     : 0.4578, # MBar, 300K Shear mod.
            # 'sgB'    : 3.8e-4, # K^-1
        }

        '''
        self.model_ptw_ss_cTm = pmh.MaterialModel(
            flow_stress_model = pmh.PTW_Yield_Stress,
            shear_modulus_model = pmh.Simple_Shear_Modulus,
        )
        '''
        self.model_vec_ptw_ss_cTm = pmv.MaterialModel(
            flow_stress_model = pmv.PTW_Yield_Stress,
            shear_modulus_model = pmv.Simple_Shear_Modulus,
        )

    # ------------------------
    # test_isothermal_lowrate
    # ------------------------

    def test_isothermal_lowrate(self):
        """
        PTW, const. Tm, PW shear. Rates less than 1e-6/us (~1/s),
        should be isothermal T=const
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e-7
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=298.)
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_ss_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_ss_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_ss_cTm = \
            self.model_vec_ptw_ss_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_ptw_ss_cTm[ 0][3][0],
            results_vec_ptw_ss_cTm[-1][3][0]
        ))
        '''
        self.assertEqual(
            results_ptw_ss_cTm    [ 0][3],
            results_vec_ptw_ss_cTm[ 0][3]
        )
        self.assertEqual(
            results_ptw_ss_cTm    [-1][3],
            results_vec_ptw_ss_cTm[-1][3]
        )
        '''
        self.assertEqual(
            results_vec_ptw_ss_cTm[ 0][3],
            results_vec_ptw_ss_cTm[-1][3]
        )

    # ------------------------
    # test_adiabatic_highrate
    # ------------------------

    def test_adiabatic_highrate(self):
        """
        PTW, const. Tm, PW shear. Rates greater than 1e-6/us (~1/s),
        temp. changes adiabatically
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e-2
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=298.)
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_ss_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_ss_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_ss_cTm = \
            self.model_vec_ptw_ss_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_ptw_ss_cTm[ 0][3][0],
            results_vec_ptw_ss_cTm[-1][3][0]
        ))
        '''
        self.assertEqual(
            results_ptw_ss_cTm    [ 0][3],
            results_vec_ptw_ss_cTm[ 0][3]
        )
        self.assertEqual(
            results_ptw_ss_cTm    [-1][3],
            results_vec_ptw_ss_cTm[-1][3]
        )
        '''
        self.assertNotEqual(
            results_vec_ptw_ss_cTm[ 0][3],
            results_vec_ptw_ss_cTm[-1][3]
        )

    # ------------------------
    # test_stress_highrate
    # ------------------------

    def test_stress_highrate(self):
        """
        PTW, constant Tm, simple shear mod, high rate 1e6/s, temp.=620K,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e0(micro-s)^-1 = 1e6(s)^-1

        FLAG sim results
        eps_p = 0.244463, flow_stress =  485.301 MPa, temp ~ 620 K
        tolerance set on original code is within 6%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.244463
        edot_pm  = 1e0
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=298.)
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_ss_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_ss_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_ss_cTm = \
            self.model_vec_ptw_ss_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_ss_cTm[ 0][1][0],
                  results_vec_ptw_ss_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_ss_cTm[-1][1][0],
                  results_vec_ptw_ss_cTm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_ss_cTm,
            results_vec_ptw_ss_cTm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_midhighrate
    # ------------------------

    def test_stress_midhighrate(self):
        """
        PTW, constant Tm, simple shear mod, mid-high rate 1e4/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.383542, flow_stress =  439.790 MPa, temp ~ 341.5 K
        tolerance set on original code is within 1%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.383542
        edot_pm  = 1e-2
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=298.)
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_ss_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_ss_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_ss_cTm = \
            self.model_vec_ptw_ss_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_ss_cTm[ 0][1][0],
                  results_vec_ptw_ss_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_ss_cTm[-1][1][0],
                  results_vec_ptw_ss_cTm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_ss_cTm,
            results_vec_ptw_ss_cTm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_midlowrate
    # ------------------------

    def test_stress_midlowrate(self):
        """
        PTW, constant Tm, simple shear mod, low-mid rate 1e2/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.388302, flow_stress =  401.108 MPa, temp ~ 328.1 K
        tolerance set on original code is within 1%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.388302
        edot_pm  = 1e-4
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=298.)
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_ss_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_ss_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_ss_cTm = \
            self.model_vec_ptw_ss_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_ss_cTm[ 0][1][0],
                  results_vec_ptw_ss_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_ss_cTm[-1][1][0],
                  results_vec_ptw_ss_cTm[-1][2][0]*1e5
              ))
        np.testing.assert_allclose(
            results_ptw_ss_cTm,
            results_vec_ptw_ss_cTm[:,:,0]
        )

    # ------------------------
    # test_stress_lowrate
    # ------------------------

    def test_stress_lowrate(self):
        """
        PTW, constant Tm, simple shear mod, low rate 1e0/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-6(micro-s)^-1 = 1e0(s)^-1

        FLAG sim results
        eps_p = 0.385416, flow_stress =  375.227 MPa
        tolerance set on original code is within 2%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.385416
        edot_pm  = 1e-6
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=298.)
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_ss_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_ss_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_ss_cTm = \
            self.model_vec_ptw_ss_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_ss_cTm[ 0][1][0],
                  results_vec_ptw_ss_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_ss_cTm[-1][1][0],
                  results_vec_ptw_ss_cTm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_ss_cTm,
            results_vec_ptw_ss_cTm[:,:,0]
        )
        '''


# ------------------------------------------------------------------------------
# TestPTWYieldStress_SteinShearConstTm
# ------------------------------------------------------------------------------

class TestPTWYieldStress_SteinShearConstTm(unittest.TestCase):

    # Functions:
    # setUp
    # test_isothermal_lowrate
    # test_adiabatic_highrate
    # test_stress_highrate
    # test_stress_midhighrate
    # test_stress_midlowrate
    # test_stress_lowrate

    # ------------------------
    # setUp
    # ------------------------

    def setUp(self):
        """
        Parameters are for OFHC copper.
        """

        self.params = {
            # PTW
            'theta' : 0.025,
            'p'     : 2.,
            's0'    : 0.0085,
            'sInf'  : 0.00055,
            'kappa' : 0.11,
            'gamma' : 1.e-5,
            'y0'    : 0.0001,
            'yInf'  : 0.0001
        }

        self.params_vec = {
            # PTW
            'theta'  : 0.025,
            'p'      : 2.,
            's0'     : 0.0085,
            'sInf'   : 0.00055,
            'kappa'  : 0.11,
            'lgamma' : np.log(1e-5),
            'y0'     : 0.0001,
            'yInf'   : 0.0001,
            'y1'     : 0.094,
            'y2'     : 0.575,
        }

        self.consts = {
            # PTW
            'y1'     : 0.094,
            'y2'     : 0.575,
            'beta'   : 0.25,
            'matomic': 63.546,
            'chi'    : 1.0,
            # Constant Spec. Heat
            'Cv0'    : 0.383e-5,
            # Constant Density
            'rho0'   : 8.9375,
            # Constant Melt Temp.
            'Tmelt0' : 1625.,
            # # Constant Shear Mod.
            # 'G0'     : 0.4578,
            # # Simple Shear Mod.
            # 'G0'     : 0.50889, # Cold shear
            # 'alpha'  : 0.21
            # SG Shear Mod.
            'G0'     : 0.4578, # MBar, 300K Shear mod.
            'sgB'    : 3.8e-4, # K^-1
        }

        self.consts_vec = {
            # PTW
            'beta'   : 0.25,
            'matomic': 63.546,
            'chi'    : 1.0,
            # Constant Spec. Heat
            'Cv0'    : 0.383e-5,
            # Constant Density
            'rho0'   : 8.9375,
            # Constant Melt Temp.
            'Tmelt0' : 1625.,
            # # Constant Shear Mod.
            # 'G0'     : 0.4578,
            # # Simple Shear Mod.
            # 'G0'     : 0.50889, # Cold shear
            # 'alpha'  : 0.21
            # SG Shear Mod.
            'G0'     : 0.4578, # MBar, 300K Shear mod.
            'sgB'    : 3.8e-4, # K^-1
        }

        '''
        self.model_ptw_sg_cTm = pmh.MaterialModel(
            flow_stress_model = pmh.PTW_Yield_Stress,
            shear_modulus_model = pmh.Stein_Shear_Modulus,
        )
        '''
        self.model_vec_ptw_sg_cTm = pmv.MaterialModel(
            flow_stress_model = pmv.PTW_Yield_Stress,
            shear_modulus_model = pmv.Stein_Shear_Modulus,
        )

    # ------------------------
    # test_isothermal_lowrate
    # ------------------------

    def test_isothermal_lowrate(self):
        """
        PTW, const. Tm, SG shear. Rates less than 1e-6/us (~1/s),
        should be isothermal T=const
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e-7
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=298.)
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_sg_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_sg_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_sg_cTm = \
            self.model_vec_ptw_sg_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_ptw_sg_cTm[ 0][3][0],
            results_vec_ptw_sg_cTm[-1][3][0]
        ))
        '''
        self.assertEqual(
            results_ptw_sg_cTm    [ 0][3],
            results_vec_ptw_sg_cTm[ 0][3]
        )
        self.assertEqual(
            results_ptw_sg_cTm    [-1][3],
            results_vec_ptw_sg_cTm[-1][3]
        )
        '''
        self.assertEqual(
            results_vec_ptw_sg_cTm[ 0][3],
            results_vec_ptw_sg_cTm[-1][3]
        )

    # ------------------------
    # test_adiabatic_highrate
    # ------------------------

    def test_adiabatic_highrate(self):
        """
        PTW, const. Tm, SG shear. Rates greater than 1e-6/us (~1/s),
        temp. changes adiabatically
        """

        emax_pm  = 1.0
        edot_pm  = 1.0e-2
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=298.)
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_sg_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_sg_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_sg_cTm = \
            self.model_vec_ptw_sg_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Temp. at t=0: {0:.2f} K and t=end: {1:.2f} K".format(
            results_vec_ptw_sg_cTm [0][3][0],
            results_vec_ptw_sg_cTm[-1][3][0]
        ))
        '''
        self.assertEqual(
            results_ptw_sg_cTm    [ 0][3],
            results_vec_ptw_sg_cTm[ 0][3]
        )
        self.assertEqual(
            results_ptw_sg_cTm    [-1][3],
            results_vec_ptw_sg_cTm[-1][3]
        )
        '''
        self.assertNotEqual(
            results_vec_ptw_sg_cTm[ 0][3],
            results_vec_ptw_sg_cTm[-1][3]
        )

    # ------------------------
    # test_stress_highrate
    # ------------------------

    def test_stress_highrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, high rate 1e6/s,
        temp.=617K, compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e0(micro-s)^-1 = 1e6(s)^-1

        FLAG sim results
        eps_p = 0.244632, flow_stress =  427.011 MPa, temp ~ 617 K
        tolerance set on original code is within 4%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.244632
        edot_pm  = 1e0
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=298.)
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_sg_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_sg_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_sg_cTm = \
            self.model_vec_ptw_sg_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_sg_cTm[ 0][1][0],
                  results_vec_ptw_sg_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_sg_cTm[-1][1][0],
                  results_vec_ptw_sg_cTm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_sg_cTm,
            results_vec_ptw_sg_cTm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_midhighrate
    # ------------------------

    def test_stress_midhighrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, mid-high rate 1e4/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.384834, flow_stress =  410.483 MPa, temp ~ 339 K
        tolerance set on original code is within 2%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.384834
        edot_pm  = 1e-2
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=298.)
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_sg_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_sg_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_sg_cTm = \
            self.model_vec_ptw_sg_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_sg_cTm[ 0][1][0],
                  results_vec_ptw_sg_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_sg_cTm[-1][1][0],
                  results_vec_ptw_sg_cTm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_sg_cTm,
            results_vec_ptw_sg_cTm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_midlowrate
    # ------------------------

    def test_stress_midlowrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, low-mid rate 1e2/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.388306, flow_stress =  373.629 MPa, temp ~ 326 K
        tolerance set on original code is within 1%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.388306
        edot_pm  = 1e-4
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=298.)
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_sg_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_sg_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_sg_cTm = \
            self.model_vec_ptw_sg_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_sg_cTm[ 0][1][0],
                  results_vec_ptw_sg_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_sg_cTm[-1][1][0],
                  results_vec_ptw_sg_cTm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_sg_cTm,
            results_vec_ptw_sg_cTm[:,:,0]
        )
        '''

    # ------------------------
    # test_stress_lowrate
    # ------------------------

    def test_stress_lowrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, low rate 1e0/s,
        compare vectorized w/ original.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s).

        High rate = 1e-6(micro-s)^-1 = 1e0(s)^-1

        FLAG sim results
        eps_p = 0.388966, flow_stress =  351.103 MPa
        tolerance set on original code is within 2%

        Vectorized physical models code should produce same result as original
        code.
        """

        emax_pm  = 0.388966
        edot_pm  = 1e-6
        nhist_pm = 100

        '''
        # Get results for original physical models code
        shist = pmh.generate_strain_history(
            emax = emax_pm, edot = edot_pm, Nhist = nhist_pm)
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=298.)
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        '''

        # Get results for vectorized physical models code
        emax_vec = np.array([emax_pm])
        edot_vec = np.array([edot_pm])
        shist_vec = pmv.generate_strain_history_new(
            emax = emax_vec, edot = edot_vec, nhist = nhist_pm)
        self.model_vec_ptw_sg_cTm.initialize(self.params_vec, self.consts_vec)
        self.model_vec_ptw_sg_cTm.initialize_state(T=np.array([298.]))
        results_vec_ptw_sg_cTm = \
            self.model_vec_ptw_sg_cTm.compute_state_history(shist_vec)

        # result format
        # [time, strain, stress, temp, shear_mod, density]

        # Compare results at beginning and end between original and vectorized
        print("Eps/Sig at t=0: {0:.2f}/{1:.2f} MPa and"
              " t=end: {2:.2f}/{3:.2f} MPa".format(
                  results_vec_ptw_sg_cTm[ 0][1][0],
                  results_vec_ptw_sg_cTm[ 0][2][0]*1e5,
                  results_vec_ptw_sg_cTm[-1][1][0],
                  results_vec_ptw_sg_cTm[-1][2][0]*1e5
              ))
        '''
        np.testing.assert_allclose(
            results_ptw_sg_cTm,
            results_vec_ptw_sg_cTm[:,:,0]
        )
        '''


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
