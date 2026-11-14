"""Confronta CPU foreground, CPU background e Vulkan su due estratti configurabili."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from trascrivi_lezioni import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--starts', type=float, nargs=2, default=[0, 60])
    parser.add_argument('--cpu-model', required=True)
    parser.add_argument('--gpu-model', required=True)
    parser.add_argument('--profili', nargs='+', choices=['cpu8', 'cpu4', 'vulkan'],
                        default=['cpu8', 'cpu4', 'vulkan'])
    args = parser.parse_args()
    parent = HERE/'tmp/bench-arc'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='misure-', dir=parent))
    source = args.input.resolve()
    print(run, flush=True)
    with (run/'tempi.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['profilo', 'passaggio', 'inizio_s', 'audio_s', 'tempo_s', 'min_audio_al_min', 'modello'])
        for profile in args.profili:
            model = args.gpu_model if profile == 'vulkan' else args.cpu_model
            for attempt in (1, 2):
                for start in args.starts:
                    dest = run/f'{profile}-{attempt}-{start}'
                    cmd = [sys.executable, '-X', 'utf8', str(HERE/'trascrivi_lezioni.py'), str(source),
                           '--model', model, '--offline', '--threads', '8' if profile == 'cpu8' else '4',
                           '--glossary', str(HERE/'glossario_parallel.txt'), '--language', 'en',
                           '--start', str(start), '--end', str(start+45), '--chunk-minutes', '.4',
                           '--output', str(dest)]
                    if profile == 'vulkan':
                        cmd += ['--engine', 'vulkan']
                    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' and profile != 'cpu8' else 0
                    started = time.perf_counter()
                    with dest.with_suffix('.log').open('w', encoding='utf8') as log:
                        subprocess.run(cmd, creationflags=flags, stdout=log, stderr=subprocess.STDOUT, check=True)
                    elapsed = time.perf_counter()-started
                    row = [profile, attempt, start, 45, round(elapsed, 3), round(45/elapsed, 3), model]
                    writer.writerow(row)
                    handle.flush()
                    print(row[:6], flush=True)


if __name__ == '__main__':
    main()
