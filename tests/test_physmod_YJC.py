import statistics as stat
import unittest

import numpy as np

from impala.physics import physical_models_vec as pmh


# Constant yield stress model
class TestJCYieldStress(unittest.TestCase):
    def setUp(self):
        """
        Parameters are for OFHC copper.
        """
        self.params = {
            "A": 0.00090,  # MBar
            "B": 0.00292,  # MBar
            "C": 0.0250,  # -
            "n": np.array([0.31]),  # -
            "m": np.array([1.09]),  # -
        }
        self.consts = {
            "rho0": 8.9375,  # g/cc
            "Cv0": 0.383e-5,  # MBar*cc/g*K
            "G0": 0.4578,  # MBar
            "Tmelt0": 1356.0,  # K
            "Tref": 298.0,  # K
            "edot0": 1.0e-6,  # 1/micro-s
            "chi": 1.0,
        }
        self.model_yjc = pmh.MaterialModel(
            flow_stress_model=pmh.JC_Yield_Stress,
        )

    def test_isothermal_lowrate(self):
        """
        JC, const. g0, Tm. Rates less than 1e-6/us (~1/s), should be isothermal T=const
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-7]), nhist=1000
        )
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=np.array([298.0]))
        results_yjc = self.model_yjc.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertEqual(results_yjc[0][3], results_yjc[-1][3])

    def test_adiabatic_highrate(self):
        """
        JC, const. g0, Tm. Rates greater than 1e-6/us (~1/s), temperature changes adiabatically
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([1.0]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=np.array([298.0]))
        results_yjc = self.model_yjc.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        self.assertNotEqual(results_yjc[0][3], results_yjc[-1][3])

    def test_stress_highrate(self):
        """
        JC, constant g0 and Tm, high rate 1e4/s, FLAG/python comparison within 2%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-2(micro-s)^-1 = 1e4(s)^-1

        FLAG sim results
        eps_p = 0.28179, flow_stress =  343.37 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.28179]), edot=np.array([1e-2]), nhist=1000
        )
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=np.array([298.0]))
        results_yjc = self.model_yjc.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_yjc[-1, 2, 0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.02
        stress_FLAG = 343.37e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / stat.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_midrate(self):
        """
        JC, constant g0 and Tm, mid rate 1e2/s, FLAG/python comparison within 1%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-4(micro-s)^-1 = 1e2(s)^-1

        FLAG sim results
        eps_p = 0.28895, flow_stress =  317.06 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.28895]), edot=np.array([1e-4]), nhist=1000
        )
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=np.array([298.0]))
        results_yjc = self.model_yjc.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_yjc[-1, 2, 0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.01
        stress_FLAG = 317.06e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / stat.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)

    def test_stress_lowrate(self):
        """
        JC, constant g0 and Tm, low rate 1e-1/s, FLAG/python comparison within 5%.

        Compare flow stress with 3D single cell uniaxial compression simulation
        performed with FLAG. 2D and 1D produced multiaxial stress states as
        these dimensional spaces assume plane strain on the non-simulated
        dimension(s). Constant shear stress is the only node available with
        Johnson-Cook flow stress model in FLAG.

        High rate = 1e-7(micro-s)^-1 = 1e-1(s)^-1

        FLAG sim results
        eps_p = 0.28928, flow_stress =  284.89 MPa

        Results will not be exact due to numeric differences. Therefore,
        perform assertion that Python result is within certain percentage of
        FLAG result at an advanced level of plastic strain.

        This makes it difficult to directly compare the initial yield stress at
        zero plastic strain.
        """
        shist = pmh.generate_strain_history_new(
            emax=np.array([0.28928]), edot=np.array([1e-7]), nhist=1000
        )
        self.model_yjc.initialize(self.params, self.consts)
        self.model_yjc.initialize_state(T=np.array([298.0]))
        results_yjc = self.model_yjc.compute_state_history(shist)
        # result format
        # [time, strain, stress, temp, shear_mod, density]
        # Assess stress at the emax
        stress_emax = results_yjc[-1, 2, 0]
        # Set maximum relative tolerance between FLAG and Python
        stress_tol = 0.05
        stress_FLAG = 284.89e-5  # Units MBar
        rel_diff = abs(stress_emax - stress_FLAG) / stat.mean([
            stress_emax,
            stress_FLAG,
        ])

        self.assertTrue(rel_diff < stress_tol)
