"""Transcription checks for the standalone paper's released result tables."""

import json
import unittest

from paper_results import PAPER, render, synchronise


class PaperTables(unittest.TestCase):
    def setUp(self):
        self.tables = render(json.loads((PAPER / 'numerical_results.json').read_text()))
        self.source = (PAPER / 'historical_results.tex').read_text()

    def test_paper_matches_released_summaries(self):
        self.assertEqual(synchronise(self.source, self.tables), self.source)
        self.assertEqual(len(self.tables), 9)

    def test_numeric_transcription_change_is_rejected(self):
        changed = self.source.replace('0.652833', '0.652834', 1)
        self.assertNotEqual(changed, self.source)
        with self.assertRaisesRegex(ValueError, 'differs'):
            synchronise(changed, self.tables)

    def test_missing_or_duplicated_marker_is_rejected(self):
        marker = '% Begin result-table:measurement'
        for changed in (self.source.replace(marker, '', 1), self.source + marker):
            with self.subTest(changed=changed[-50:]):
                with self.assertRaisesRegex(ValueError, 'Missing or repeated'):
                    synchronise(changed, self.tables, write=True)

    def test_update_preserves_surrounding_research_text(self):
        changed = self.source.replace('0.652833', '0.652834', 1)
        self.assertEqual(synchronise(changed, self.tables, write=True), self.source)

    def test_allocation_intervals_retain_signs_and_scale(self):
        table = self.tables['prospective_contrasts']
        self.assertIn(r'\times10^{', table)
        self.assertIn('$[-', table)
        self.assertIn('finite-sample coverage is unverified', table)


if __name__ == '__main__':
    unittest.main()
