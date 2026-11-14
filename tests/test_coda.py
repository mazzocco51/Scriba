import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coda_trascrizioni as queue


class QueueTests(unittest.TestCase):
    def test_recovery_and_frozen_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            media = root/'prima.mp4'
            media.touch()
            with patch.object(queue, 'ROOT', root/'queue'), patch.object(queue, 'wake'):
                with contextlib.redirect_stdout(io.StringIO()):
                    queue.main('avvia', str(root))
                (root/'seconda.mp4').touch()
                with contextlib.closing(queue.database()) as db:
                    job = db.execute('SELECT * FROM jobs').fetchone()
                    self.assertEqual(json.loads(job['args'])['sources'], [str(media)])
                    db.execute("UPDATE jobs SET status='in corso'")
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        queue.main('stato', job['id'])
                        queue.main('riprendi', job['id'])
                    self.assertIn('interrotto', out.getvalue())
                    with patch.object(queue.asr, 'main', return_value=0) as transcribe:
                        queue.run_worker(db)
                        queue.run_worker(db)
                    self.assertEqual(transcribe.call_count, 1)
                    self.assertEqual(db.execute('SELECT status FROM jobs').fetchone()[0], 'completato')


if __name__ == '__main__':
    unittest.main()
