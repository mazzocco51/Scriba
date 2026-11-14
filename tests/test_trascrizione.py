"""Regressioni per confini dei blocchi, copertura parziale e lock di ripresa."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trascrivi_lezioni as transcription


class TranscriptionTests(unittest.TestCase):
    def test_missing_confidence_preserves_openvino_words(self):
        chunk = {'core_start': 0, 'core_end': 30, 'range_start': 0, 'range_end': 30,
                 'segments': [{'start': 1, 'end': 2, 'text': ' GPU', 'avg_logprob': None,
                               'compression_ratio': None, 'no_speech_prob': None,
                               'words': [{'start': 1, 'end': 2, 'word': ' GPU', 'probability': None}]}]}
        row, = transcription.owned_segments(chunk)
        self.assertEqual(row['text'], 'GPU')
        self.assertIn('affidabilita numerica non disponibile', row['review'])

    def test_time_and_invalid_time(self):
        self.assertEqual(transcription.parse_time('01:22:05.5'), 4925.5)
        for value in ['-1', 'nan', '1:2:3:4']:
            with self.assertRaises(Exception):
                transcription.parse_time(value)

    def test_directory_filter_and_recursive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'lezione.MP4').touch()
            (root/'note.md').touch()
            (root/'altre').mkdir()
            (root/'altre'/'lezione.mp4').touch()
            self.assertEqual(len(transcription.discover([folder], False)), 1)
            self.assertEqual(len(transcription.discover([folder], True)), 2)

    def test_overlap_has_one_owner_even_for_word_across_boundary(self):
        words = [{'word': text, 'start': start, 'end': end, 'probability': 0.9}
                 for text, start, end in [(' prima', 8, 9), (' confine', 9.8, 10.2), (' dopo', 11, 12)]]
        segment = {'words': words, 'text': ' prima confine dopo', 'start': 8, 'end': 12,
                   'avg_logprob': -0.1, 'compression_ratio': 1, 'no_speech_prob': 0.1}
        left = {'core_start': 0, 'core_end': 10, 'range_start': 0, 'range_end': 20,
                'segments': [segment]}
        right = {**left, 'core_start': 10, 'core_end': 20}
        rows = transcription.owned_segments(left) + transcription.owned_segments(right)
        self.assertEqual(' '.join(r['text'] for r in rows), 'prima confine dopo')
        self.assertEqual(sum(r['text'].count('confine') for r in rows), 1)
        self.assertTrue(rows[1]['review'])
        self.assertEqual(len(segment['words']), 3)  # Il contesto grezzo resta disponibile.

    def test_partial_output_is_explicit(self):
        config = {'duration': 3600, 'start': 0, 'end': 3600, 'model': 'test', 'language': 'en'}
        chunk = {'core_start': 0, 'core_end': 600, 'decoded_fraction': 1,
                 'range_start': 0, 'range_end': 3600, 'language': 'en', 'segments': []}
        md = transcription.render_markdown(Path('lezione.mp4'), config, [chunk], 6)
        self.assertIn('Stato: PARZIALE', md)
        self.assertIn('Blocchi completati: 1/6', md)
        self.assertNotIn('Stato: COMPLETA', md)

    def test_lock_prevents_concurrent_writer_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder)/'run.lock'
            with transcription.exclusive_run(lock):
                with self.assertRaises(RuntimeError):
                    with transcription.exclusive_run(lock):
                        self.fail('Secondo writer ammesso')
            with self.assertRaises(ValueError):
                with transcription.exclusive_run(lock):
                    raise ValueError('Interruzione simulata')
            with transcription.exclusive_run(lock):
                pass


if __name__ == '__main__':
    unittest.main()
