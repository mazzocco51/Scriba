"""Whisper su Intel GPU con modello persistente e timestamp originali."""
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
import zlib

import numpy as np
from faster_whisper.vad import get_speech_timestamps, SpeechTimestampsMap


class OpenVINOModel:
    def __init__(self, path, streams):
        import openvino as ov
        import openvino_genai as genai

        self.path = path.resolve()
        if not self.path.is_dir():
            raise ValueError(f'Modello OpenVINO assente: {self.path}')
        core = ov.Core()
        print('GPU OpenVINO:', core.get_property('GPU', 'FULL_DEVICE_NAME'), flush=True)
        cache = self.path.parent/'compiled-cache'
        self.pipe = genai.WhisperPipeline(
            str(self.path), 'GPU', CACHE_DIR=str(cache), word_timestamps=True,
            PERFORMANCE_HINT='LATENCY', NUM_STREAMS=str(streams))
        self.config = {
            'engine': 'openvino', 'openvino': ov.__version__,
            'genai': importlib.metadata.version('openvino-genai'),
            'streams': streams, 'vad_timeline': 'silero-original-v1',
            'files': {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                      for p in sorted(self.path.iterdir()) if p.suffix in ('.xml', '.bin', '.json')}}

    def prepare_audio(self, audio, vad_filter):
        timeline = None
        if vad_filter:
            speech = get_speech_timestamps(audio, min_silence_duration_ms=1000)
            if not speech:
                return np.array([], dtype=np.float32), None
            timeline = SpeechTimestampsMap(speech, 16000)
            audio = np.concatenate([audio[s['start']:s['end']] for s in speech])
        return audio, timeline

    def transcribe(self, audio, *, language, initial_prompt, beam_size, vad_filter, **options):
        audio, timeline = audio if isinstance(audio, tuple) else self.prepare_audio(audio, vad_filter)
        if not len(audio):
            return [], SimpleNamespace(language=language or 'indeterminata')
        config = self.pipe.get_generation_config()
        config.num_beams = beam_size
        config.return_timestamps = True
        config.word_timestamps = True
        config.do_sample = False
        if language:
            config.language = f'<|{language}|>'
        config.task = 'transcribe'
        if initial_prompt:
            config.initial_prompt = initial_prompt
        result = self.pipe.generate(audio, config)
        words = []
        for word in result.words:
            start, end = float(word.start_ts), float(word.end_ts)
            if not (0 <= start <= end <= len(audio)/16000 + 0.5):
                raise RuntimeError('Timestamp OpenVINO fuori intervallo.')
            if timeline:
                start = timeline.get_original_time(start)
                end = max(start, timeline.get_original_time(end, is_end=True))
            words.append(SimpleNamespace(start=start, end=end, word=word.word, probability=None))
        if not words and str(result).strip():
            raise RuntimeError('OpenVINO non ha restituito i timestamp delle parole.')
        # Gruppi di parole contigue mantengono la stessa assegnazione ai checkpoint.
        segments = []
        for index in range(0, len(words), 24):
            group = words[index:index+24]
            text = ''.join(w.word for w in group)
            encoded = text.encode('utf8')
            segments.append(SimpleNamespace(start=group[0].start, end=group[-1].end,
                text=text, words=group, avg_logprob=None,
                compression_ratio=len(encoded)/max(1, len(zlib.compress(encoded))),
                no_speech_prob=None))
        return segments, SimpleNamespace(language=language or 'auto')
