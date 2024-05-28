#!/usr/bin/env python3

import unittest

import test_physmod_vec_constyield as pmv_cy
import test_physmod_vec_YJC as pmv_yjc
import test_physmod_vec_PTW as pmv_ptw

"""
This script runs all tests.
Invoke with:
python run_tests_physmod_vec_compare.py

All test methods are found in the py_tests directory.
Add or subtract tests by using suite.addTest(...).
Add as many lines as needed.
"""

suite = unittest.TestSuite()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add as many tests as desired. Point to test module and test case.
assert False
suite.addTest(unittest.makeSuite(pmv_cy.TestConstantYieldStress))
suite.addTest(unittest.makeSuite(pmv_yjc.TestJCYieldStress))
suite.addTest(unittest.makeSuite(pmv_ptw.TestPTWYieldStress_Constg0Tm))
suite.addTest(unittest.makeSuite(pmv_ptw.TestPTWYieldStress_SimpShearConstTm))
suite.addTest(unittest.makeSuite(pmv_ptw.TestPTWYieldStress_SteinShearConstTm))
# End adding tests.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Store the result of all tests run.
# Verbosity is the amount of output printed to the std.out.
# verbosity = 2: print full test name and result summary.
# verbosity = 1: print dots for completed tests and result summary.
# verbosity = 0: print no test name or dots, just summary.
result = unittest.TextTestRunner(verbosity=2).run(suite)

# Produce exit codes for console interaction,
# based on the success or failure of tests.
if result.wasSuccessful():
    exit(0)
else:
    exit(1)
