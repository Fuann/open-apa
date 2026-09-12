import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from word_evaluation import load_word_evaluation, open_word_arrays

class WordEvaluationTest(unittest.TestCase):
    def test_pool_cap_select_and_exclude_failures(self):
        raw = np.array([[[12., 6., 4.], [10., 8., 6.], [1., 1., 1.], [8., 9., 10.]],
                        [[99., 99., 99.]] * 4, [[99., 99., 99.]] * 4])
        target = np.zeros((3, 4, 4))
        target[0, :, -1] = [0, 0, 1, 2]
        evaluation = {'items': {'ok': {'words': ['match', 'insert', 'substitute'],
            'score_indices': [0, 2], 'targets': [[9, 10, 8], [7, 10, 9]]}},
            'failures': {'failed_eval': 'failed'}}
        p, t = open_word_arrays(raw, target, ['ok', 'failed_asr', 'failed_eval'], evaluation, {'failed_asr': 'failed'})
        np.testing.assert_array_equal(p, [[10, 7, 5], [8, 9, 10]])
        np.testing.assert_array_equal(t, [[9, 10, 8], [7, 10, 9]])
        evaluation['items']['ok']['words'].append('extra')
        with self.assertRaisesRegex(ValueError, 'token mismatch'):
            open_word_arrays(raw, target, ['ok', 'failed_asr', 'failed_eval'], evaluation, {'failed_asr': 'failed'})

    def test_metadata_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            manifest, scores, output = d/'et.csv', d/'scores.json', d/'word.json'
            manifest.write_text('utt_id\na\n')
            scores.write_text('{}')
            data = {'protocol': 'levenshtein_match_substitution', 'fingerprint': {
                name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in [('manifest', manifest), ('scores', scores)]}}
            output.write_text(json.dumps(data))
            self.assertEqual(load_word_evaluation(output, manifest, scores)[1], ['a'])
            scores.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_word_evaluation(output, manifest, scores)

if __name__ == '__main__':
    unittest.main()
