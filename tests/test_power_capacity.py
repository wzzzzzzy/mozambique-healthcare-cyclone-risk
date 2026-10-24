import unittest

from power_capacity import (
    hospital_capacity_failure_probability,
    legacy_capacity_failure_probability,
)


class HospitalCapacityFailureProbabilityTest(unittest.TestCase):
    def test_no_hospitals_has_no_failure_probability(self):
        self.assertEqual(hospital_capacity_failure_probability(0.0, 0), 0.0)

    def test_probability_decreases_with_supply(self):
        self.assertEqual(hospital_capacity_failure_probability(0.0, 1), 1.0)
        self.assertEqual(hospital_capacity_failure_probability(2.0, 1), 0.5)
        self.assertEqual(hospital_capacity_failure_probability(4.0, 1), 0.0)

    def test_demand_scales_with_hospital_count(self):
        self.assertEqual(hospital_capacity_failure_probability(4.0, 2), 0.5)

    def test_probability_is_clipped(self):
        self.assertEqual(hospital_capacity_failure_probability(-1.0, 1), 1.0)
        self.assertEqual(hospital_capacity_failure_probability(8.0, 1), 0.0)

    def test_legacy_formula_is_preserved_for_other_workflows(self):
        self.assertEqual(legacy_capacity_failure_probability(1.0, 1), 0.25)
        self.assertEqual(legacy_capacity_failure_probability(3.0, 1), 0.75)


if __name__ == "__main__":
    unittest.main()
