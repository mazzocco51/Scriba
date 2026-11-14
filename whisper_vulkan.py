"""Adattatore whisper.cpp. La mappa VAD riporta ogni timestamp al video."""
import json
import math
import os
from collections import deque
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import wave

import numpy as np
from faster_whisper.vad import get_speech_timestamps, SpeechTimestampsMap


class VulkanModel:
    def __init__(self, cli, path, threads, output):
        self.cli, self.path = cli.resolve(), path.resolve()
        for file in (self.cli, self.path):
            if not file.is_file():
                raise ValueError(f'File Vulkan assente: {file}. Specificare --whisper-cli e --model (GGML).')
        self.threads = threads
        self.process = None
        self.selected = False
        self.temp = output.resolve()
        self.temp.mkdir(parents=True, exist_ok=True)
        self.config = {'engine': 'whisper.cpp-vulkan', 'cli': str(self.cli), 'threads': threads,
                       'decoder_settings': 'beam-and-best-of-v1',
                       'vad_timeline': 'silero-original-v1',
                       'files': {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                                 for p in (self.cli, self.path)}}

    def transcribe(self, audio, *, language, initial_prompt, vad_filter, **options):
        rate = 16000
        timeline = None
        if vad_filter:
            speech = get_speech_timestamps(audio, **options['vad_parameters'])
            if not speech:
                return [], SimpleNamespace(language=language or 'indeterminata')
            timeline = SpeechTimestampsMap(speech, rate)
            audio = np.concatenate([audio[s['start']:s['end']] for s in speech])

        def times(offsets):
            start = min(len(audio)/rate, max(0, offsets['from']/1000))
            end = min(len(audio)/rate, max(start, offsets['to']/1000))
            if timeline:
                start = timeline.get_original_time(start)
                end = max(start, timeline.get_original_time(end, is_end=True))
            return start, end

        with tempfile.TemporaryDirectory(dir=self.temp) as temp:
            wav = Path(temp)/'audio.wav'
            with wave.open(str(wav), 'wb') as handle:
                handle.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                handle.writeframes(np.clip(audio * 32768, -32768, 32767).astype('<i2').tobytes())
            streaming = self.cli.name == 'whisper-stream-cli.exe'
            output = wav if streaming else Path(temp)/'out'
            cmd = [str(self.cli), '-m', str(self.path), '-f', str(wav), '-of', str(output),
                   '-ojf', '-l', language or 'auto', '-t', str(self.threads),
                   '-bs', str(options['beam_size']), '-bo', str(options['beam_size'])]
            if initial_prompt:
                cmd.extend(['--prompt', initial_prompt])
            failed = False
            tail = deque(maxlen=35)
            if self.process is None:
                self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE if streaming else subprocess.DEVNULL,
                        env={**os.environ, 'PARALLEL_ASR_STREAM': '1'} if streaming else None,
                        encoding='utf8', errors='replace')
            else:
                self.process.stdin.write(str(wav) + '\n')
                self.process.stdin.flush()
            process = self.process
            try:
                ready = False
                for line in process.stderr:
                    tail.append(line)
                    if streaming and line.strip() == 'PARALLEL_ASR_CHUNK_READY':
                        ready = True
                        break
                    if re.search(r'using Vulkan\d+ backend', line, re.I):
                        self.selected = True
                        print(line.strip(), flush=True)
                    failed |= 'failed to initialize' in line
                code = process.poll() if ready else process.wait()
                if code or not self.selected or failed or (streaming and not ready):
                    raise RuntimeError(f'Inferenza Vulkan non confermata ({code}): {"".join(tail)}')
                data = json.loads(Path(str(output)+'.json').read_text(encoding='utf8'))
            except BaseException:
                self.close()
                raise
            if not streaming:
                self.close()

        segments = []
        for segment in data['transcription']:
            words = []
            tokens = [t for t in segment['tokens'] if not t['text'].startswith('[_')]
            for token in tokens:
                if 'offsets' not in token:
                    words = []
                    break
                start, end = times(token['offsets'])
                if words and not token['text'][:1].isspace():
                    words[-1].word += token['text']
                    words[-1].end = end
                    words[-1].probability = min(words[-1].probability, token['p'])
                else:
                    words.append(SimpleNamespace(start=start, end=end, word=token['text'], probability=token['p']))
            if ''.join(w.word for w in words).strip() != segment['text'].strip():
                # Conserva tutto il parlato anche con JSON privo di tempi per alcuni token.
                words = []
            start, end = times(segment['offsets'])
            avg = sum(math.log(max(t['p'], 1e-10)) for t in tokens) / max(1, len(tokens))
            segments.append(SimpleNamespace(start=start, end=end, text=segment['text'], words=words,
                                            avg_logprob=avg, compression_ratio=None, no_speech_prob=None))
        return segments, SimpleNamespace(language=data['result']['language'])

    def close(self):
        if self.process:
            process, self.process = self.process, None
            if process.stdin:
                process.stdin.close()
                process.stdin = None
            process.communicate()
            process.stderr.close()
            self.selected = False
