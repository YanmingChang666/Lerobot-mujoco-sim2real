"""Geometry checks for English pen paths (no font or Hanzi data dependency)."""
import string
import unittest
import numpy as np
from .latin_strokes import latin_to_strokes


class LatinStrokesTest(unittest.TestCase):
    def test_alphabet_and_word_bounds(self):
        for text in [*string.ascii_uppercase, 'HELLO', 'HELLO WORLD']:
            strokes = latin_to_strokes(text, .08)
            self.assertTrue(all(len(s) >= 2 for s in strokes))
            points = np.concatenate(strokes)
            self.assertTrue(np.isfinite(points).all())
            self.assertLessEqual(np.abs(points).max(), .0400000001)
        self.assertEqual(latin_to_strokes('Hello'), latin_to_strokes('HELLO'))

    def test_spaces_are_gaps(self):
        tight = latin_to_strokes('LL')
        spaced = latin_to_strokes('L L')
        self.assertEqual(len(tight), len(spaced))
        self.assertGreater(spaced[1][0][0]-spaced[0][-1][0], tight[1][0][0]-tight[0][-1][0])

    def test_reject_unsupported_input(self):
        for text in ['', ' ', 'HELLO!', 'A/B', '123', 'é', '永', 'A\nB']:
            with self.assertRaises(ValueError): latin_to_strokes(text)
        for size in [0, -1, float('nan')]:
            with self.assertRaises(ValueError): latin_to_strokes('A', size)


if __name__ == '__main__': unittest.main()
