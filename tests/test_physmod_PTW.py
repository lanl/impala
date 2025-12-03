import unittest

import numpy as np

from impala.physics import physical_models_vec as pmh


# Constant yield stress model
class TestPTWYieldStress_Constg0Tm(unittest.TestCase):
    def setUp(self):
        """
        Parameters are for OFHC copper.
        """
        self.params = {
            # PTW
            "theta": np.array([0.025]),
            "p": np.array([2.0]),
            "s0": np.array([0.0085]),
            "sInf": np.array([0.00055]),
            "kappa": np.array([0.11]),
            "lgamma": np.array([np.log(1e-5)]),
            "y0": np.array([0.0001]),
            "yInf": np.array([0.00009999]),
            "y1": np.array([0.094]),
            "y2": np.array([0.575]),
        }
        self.consts = {
            # PTW
            "beta": 0.25,
            "matomic": 63.546,
            "chi": 1.0,
            # Constant Spec. Heat
            "Cv0": 0.383e-5,
            # Constant Density
            "rho0": 8.9375,
            # Constant Melt Temp.
            "Tmelt0": 1625.0,
            # # Constant Shear Mod.
            "G0": 0.4578,
            # # Simple Shear Mod.
            # 'G0'     : 0.50889, # Cold shear
            # 'alpha'  : 0.21
            # # SG Shear Mod.
            # 'G0'     : 0.4578, # MBar, 300K Shear mod.
            # 'sgB'    : 3.8e-4, # K^-1
        }
        self.model_ptw_cg0Tm = pmh.MaterialModel(
            flow_stress_model=pmh.PTW_Yield_Stress,
        )

    def test_isothermal_lowrate(self):
        """
        PTW, const. g0, Tm. Rates less than 1e-6/us (~1/s), should be isothermal T=const
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-7]), nhist=1000
        )
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=np.array([298.0]))
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertEqual(results_ptw_cg0Tm[0][3], results_ptw_cg0Tm[-1][3])

    def test_adiabatic_highrate(self):
        """
        PTW, const. g0, Tm. Rates greater than 1e-6/us (~1/s), temp. changes adiabatically
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=np.array([298.0]))
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertNotEqual(results_ptw_cg0Tm[0][3], results_ptw_cg0Tm[-1][3])

    def test_stress_highrate(self):
        """
        PTW, constant g0 and Tm, high rate 1e6/s, FLAG/python comparison within 1%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e0(micro-s)^-1 = 1e6(s)^-1

        FLAG sim results
        eps_p = 0.244376, flow_stress =  521.4 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.

        NOTE: Temperature affects from volumetric heating may be offsetting the
        affect of holding the shear modulus constant, thus producing a tight match
        with the Python result, although for likely non-physical reasons.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.244376]), edot=np.array([1e0]), nhist=1000
        )
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=np.array([298.0]))
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_cg0Tm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.01
        stress_FLAG = 521.416e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_midhighrate(self):
        """
        PTW, constant g0 and Tm, mid-high rate 1e4/s, FLAG/python comparison within 9%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.382749, flow_stress =  458.198 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.382749]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=np.array([298.0]))
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_cg0Tm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.09
        stress_FLAG = 458.198e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_midlowrate(self):
        """
        PTW, constant g0 and Tm, low-mid rate 1e2/s, FLAG/python comparison within 11%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.388339, flow_stress =  418.512 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.388339]), edot=np.array([1e-4]), nhist=1000
        )
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=np.array([298.0]))
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_cg0Tm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.11
        stress_FLAG = 418.512e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_lowrate(self):
        """
        PTW, constant g0 and Tm, low rate 1e0/s, FLAG/python comparison within 9%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-6(micro-s)^-1 = 1e0(s)^-1

        FLAG sim results
        eps_p = 0.371143, flow_stress =  385.751 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.371143]), edot=np.array([1e-6]), nhist=1000
        )
        self.model_ptw_cg0Tm.initialize(self.params, self.consts)
        self.model_ptw_cg0Tm.initialize_state(T=np.asarray([298.0]))
        results_ptw_cg0Tm = self.model_ptw_cg0Tm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_cg0Tm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.09
        stress_FLAG = 385.751e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)


class TestPTWYieldStress_SimpShearConstTm(unittest.TestCase):
    def setUp(self):
        """
        Parameters are for OFHC copper.
        """
        self.params = {
            # PTW
            "theta": np.array([0.025]),
            "p": np.array([2.0]),
            "s0": np.array([0.0085]),
            "sInf": np.array([0.00055]),
            "kappa": np.array([0.11]),
            "lgamma": np.array([np.log(1e-5)]),
            "y0": np.array([0.0001]),
            "yInf": np.array([0.00009999]),
            "y1": np.array([0.094]),
            "y2": np.array([0.575]),
        }
        self.consts = {
            # PTW
            "beta": 0.25,
            "matomic": 63.546,
            "chi": 1.0,
            # Constant Spec. Heat
            "Cv0": 0.383e-5,
            # Constant Density
            "rho0": 8.9375,
            # Constant Melt Temp.
            "Tmelt0": 1625.0,
            # # Constant Shear Mod.
            # 'G0'     : 0.4578,
            # Simple Shear Mod.
            "G0": 0.50889,  # Cold shear
            "alpha": 0.21,
            # # SG Shear Mod.
            # 'G0'     : 0.4578, # MBar, 300K Shear mod.
            # 'sgB'    : 3.8e-4, # K^-1
        }
        self.model_ptw_ss_cTm = pmh.MaterialModel(
            flow_stress_model=pmh.PTW_Yield_Stress,
            shear_modulus_model=pmh.Simple_Shear_Modulus,
        )

    def test_isothermal_lowrate(self):
        """
        PTW, const. Tm, PW shear. Rates less than 1e-6/us (~1/s), should be isothermal T=const
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-7]), nhist=1000
        )
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertEqual(results_ptw_ss_cTm[0][3], results_ptw_ss_cTm[-1][3])

    def test_adiabatic_highrate(self):
        """
        PTW, const. Tm, PW shear. Rates greater than 1e-6/us (~1/s), temp. changes adiabatically
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertNotEqual(results_ptw_ss_cTm[0][3], results_ptw_ss_cTm[-1][3])

    def test_stress_highrate(self):
        """
        PTW, constant Tm, simple shear mod, high rate 1e6/s, temp.=620K, FLAG/python comparison within 6%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e0(micro-s)^-1 = 1e6(s)^-1

        FLAG sim results
        eps_p = 0.244463, flow_stress =  485.301 MPa, temp ~ 620 K

        Results are expected to be different at this high rate of strain due
        to the fact that there are temperature effects from volumetric strain
        that significantly affect the shear modulus and flow stress. A trick
        would be to initialize the model closer to the temperatures expressed
        by FLAG at maximum strain. This could get the python result closer to
        the FLAG result
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.244463]), edot=np.array([1e0]), nhist=1000
        )
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=np.array([620.0]))
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_ss_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.06
        stress_FLAG = 485.301e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_midhighrate(self):
        """
        PTW, constant Tm, simple shear mod, mid-high rate 1e4/s, FLAG/python comparison within 2%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.383542, flow_stress =  439.790 MPa, temp ~ 341.5 K

        See note above from the high rate test. Temperature effects may still
        create significant differences in computed stress.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.383542]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_ss_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.02
        stress_FLAG = 439.790e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_midlowrate(self):
        """
        PTW, constant Tm, simple shear mod, low-mid rate 1e2/s, FLAG/python comparison within 1%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.388302, flow_stress =  401.108 MPa, temp ~ 328.1 K

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.388302]), edot=1e-4, nhist=1000
        )
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_ss_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.01
        stress_FLAG = 401.108e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_lowrate(self):
        """
        PTW, constant Tm, simple shear mod, low rate 1e0/s, FLAG/python comparison within 2%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-6(micro-s)^-1 = 1e0(s)^-1

        FLAG sim results
        eps_p = 0.385416, flow_stress =  375.227 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.385416]), edot=np.array([1e-6]), nhist=1000
        )
        self.model_ptw_ss_cTm.initialize(self.params, self.consts)
        self.model_ptw_ss_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_ss_cTm = self.model_ptw_ss_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_ss_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.02
        stress_FLAG = 375.227e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)


class TestPTWYieldStress_SteinShearConstTm(unittest.TestCase):
    def setUp(self):
        """
        Parameters are for OFHC copper.
        """
        self.params = {
            # PTW
            "theta": np.array([0.025]),
            "p": np.array([2.0]),
            "s0": np.array([0.0085]),
            "sInf": np.array([0.00055]),
            "kappa": np.array([0.11]),
            "lgamma": np.array([np.log(1e-5)]),
            "y0": np.array([0.0001]),
            "yInf": np.array([0.00009999]),
            "y1": np.array([0.094]),
            "y2": np.array([0.575]),
        }
        self.consts = {
            # PTW
            "beta": 0.25,
            "matomic": 63.546,
            "chi": 1.0,
            # Constant Spec. Heat
            "Cv0": 0.383e-5,
            # Constant Density
            "rho0": 8.9375,
            # Constant Melt Temp.
            "Tmelt0": 1625.0,
            # # Constant Shear Mod.
            # 'G0'     : 0.4578,
            # # Simple Shear Mod.
            # 'G0'     : 0.50889, # Cold shear
            # 'alpha'  : 0.21
            # SG Shear Mod.
            "G0": 0.4578,  # MBar, 300K Shear mod.
            "sgB": 3.8e-4,  # K^-1
        }
        self.model_ptw_sg_cTm = pmh.MaterialModel(
            flow_stress_model=pmh.PTW_Yield_Stress,
            shear_modulus_model=pmh.Stein_Shear_Modulus,
        )

    def test_isothermal_lowrate(self):
        """
        PTW, const. Tm, SG shear. Rates less than 1e-6/us (~1/s), should be isothermal T=const
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-7]), nhist=1000
        )
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertEqual(results_ptw_sg_cTm[0][3], results_ptw_sg_cTm[-1][3])

    def test_adiabatic_highrate(self):
        """
        PTW, const. Tm, SG shear. Rates greater than 1e-6/us (~1/s), temp. changes adiabatically
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertNotEqual(results_ptw_sg_cTm[0][3], results_ptw_sg_cTm[-1][3])

    def test_stress_highrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, high rate 1e6/s, temp.=617K, FLAG/python comparison within 4%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e0(micro-s)^-1 = 1e6(s)^-1

        FLAG sim results
        eps_p = 0.244632, flow_stress =  427.011 MPa, temp ~ 617 K

        Results are expected to be different at this high rate of strain due
        to the fact that there are temperature effects from volumetric strain
        that significantly affect the shear modulus and flow stress. A trick
        would be to initialize the model closer to the temperatures expressed
        by FLAG at maximum strain. This could get the python result closer to
        the FLAG result
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.244632]), edot=np.array([1e0]), nhist=1000
        )
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=np.array([617.0]))
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_sg_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.04
        stress_FLAG = 427.011e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_midhighrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, mid-high rate 1e4/s, FLAG/python comparison within 2%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.384834, flow_stress =  410.483 MPa, temp ~ 339 K

        See note above from the high rate test. Temperature effects may still
        create significant differences in computed stress.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.384834]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_sg_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.02
        stress_FLAG = 410.483e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_midlowrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, low-mid rate 1e2/s, FLAG/python comparison within 1%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.388306, flow_stress =  373.629 MPa, temp ~ 326 K

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.388306]), edot=np.array([1e-4]), nhist=1000
        )
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_sg_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.01
        stress_FLAG = 373.629e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_lowrate(self):
        """
        PTW, constant Tm, Steinberg-Guinan shear mod, low rate 1e0/s, FLAG/python comparison within 2%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-6(micro-s)^-1 = 1e0(s)^-1

        FLAG sim results
        eps_p = 0.388966, flow_stress =  351.103 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.388966]), edot=np.array([1e-6]), nhist=1000
        )
        self.model_ptw_sg_cTm.initialize(self.params, self.consts)
        self.model_ptw_sg_cTm.initialize_state(T=np.array([298.0]))
        results_ptw_sg_cTm = self.model_ptw_sg_cTm.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_ptw_sg_cTm[-1, 2][0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.02
        stress_FLAG = 351.103e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / np.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)


if __name__ == "__main__":
    unittest.main()
