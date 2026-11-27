"""Formatting-only source audit regression; arithmetic drift must still fail."""
import unittest
from verify_qualified_prefix import has_outward_l1_fold


class L1FoldFormat(unittest.TestCase):
    def test_equivalent_source_forms(self):
        self.assertTrue(has_outward_l1_fold('.fold(0.0_f64, |sum, coefficient| outward_add(sum, coefficient.abs()))'))
        self.assertTrue(has_outward_l1_fold('.fold(0.0_f64, |sum, coefficient| {\n outward_add(sum, coefficient.abs())\n})'))

    def test_changed_arithmetic_rejected(self):
        for text in (
            '.fold(0.0_f64, |sum, coefficient| sum + coefficient.abs())',
            '.fold(0.0_f64, |sum, coefficient| { outward_add(sum, coefficient) })',
            '.fold(0.0_f64, |sum, coefficient| { outward_add(sum, coefficient.abs()) + 1.0 })',
            '.fold(1.0_f64, |sum, coefficient| outward_add(sum, coefficient.abs()))',
        ):
            self.assertFalse(has_outward_l1_fold(text), text)


if __name__ == '__main__':
    unittest.main()
