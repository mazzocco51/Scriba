"""Prova dei modelli FluidInference Parakeet V2 con OpenVINO Python.

Decodifica TDT greedy basata su eddy-audio (Apache-2.0).
Modelli FluidInference/parakeet-tdt-0.6b-v2-ov (CC-BY-4.0).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import av
import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from trascrivi_lezioni import audio_metadata, decode_block, timestamp


def render(tokens, device):
    words = []
    for token in tokens:
        if not words or token['text'].startswith(' '):
            words.append(dict(token))
        else:
            words[-1]['text'] += token['text']
    lines = ['# Prova Parakeet V2', '', f'Dispositivo {device}', '',
             'Trascrizione automatica sperimentale. Timestamp approssimativi.',
             'Verificare termini tecnici e confini delle finestre sovrapposte.', '']
    for index in range(0, len(words), 24):
        group = words[index:index+24]
        lines.extend([f'[{timestamp(group[0]["time"])}]', '',
                      ''.join(t['text'] for t in group).strip(), ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='NPU', choices=['NPU', 'GPU', 'CPU'])
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--output', default='trascrizioni/parakeet')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT/'models/parakeet-v2')
    parser.add_argument('--last-minutes', type=float)
    args = parser.parse_args()
    folder = Path(__file__).parent
    output = folder/args.output
    output.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    print('Dispositivo:', core.get_property(args.device, 'FULL_DEVICE_NAME'), flush=True)
    started = time.perf_counter()
    models = {}
    for name in ('melspectogram', 'encoder', 'decoder', 'joint'):
        model = core.read_model(args.model/f'parakeet_{name}.xml')
        device = 'CPU' if name == 'melspectogram' else args.device
        if device == 'NPU':
            for node in model.get_ordered_ops():
                if node.get_type_name() == 'LogSoftmax':
                    axis = node.get_axis()
                    rank = node.input(0).get_partial_shape().rank
                    if axis < 0 and rank.is_static:
                        node.set_axis(axis + rank.get_length())
            model.validate_nodes_and_infer_types()
        print('Compilazione', name, device, flush=True)
        models[name] = core.compile_model(model, device, {'CACHE_DIR': str(ROOT/'tmp/parakeet-cache')})
        print('Esecuzione effettiva', name, models[name].get_property('EXECUTION_DEVICES'), flush=True)
    load_seconds = time.perf_counter()-started
    vocab = json.loads((args.model/'parakeet_vocab.json').read_text(encoding='utf8'))
    source = args.input
    duration, origin = audio_metadata(source, av, 0)
    if args.last_minutes is not None:
        args.start = max(0, duration-args.last_minutes*60)
        args.seconds = duration-args.start
    end = duration if args.seconds is None else min(duration, args.start+args.seconds)
    if not 0 <= args.start < end or (args.last_minutes is not None and args.last_minutes <= 0):
        parser.error('Intervallo audio non valido')
    audio, fraction = decode_block(source, args.start, end, origin, 0, av, np)
    preparation = time.perf_counter()-started-load_seconds
    infer_started = time.perf_counter()
    tokens = []
    pre, enc, dec, joint = [models[k] for k in ('melspectogram', 'encoder', 'decoder', 'joint')]
    # Finestre con contesto sovrapposto e proprieta dei token per intervallo centrale.
    for core_start in range(0, len(audio), 7*16000):
        left = max(0, core_start-16000)
        right = min(len(audio), core_start+9*16000)
        samples = audio[left:right]
        mel_result = pre([samples[None, :], np.array([len(samples)], np.int32)])
        mel = mel_result[pre.output(0)]
        frames = int(mel_result[pre.output(1)].flat[0])
        padded = np.zeros(enc.input(0).shape, dtype=np.float32)
        count = min(frames, padded.shape[-1])
        padded[:, :, :count] = mel[:, :, :count]
        encoded = enc([padded, np.array([count], np.int32)])
        features = encoded[enc.output('encoder_output')]
        valid = min(features.shape[-1], int(encoded[enc.output('encoder_output_length')].flat[0]))
        h = np.zeros((2, 1, 640), np.float32)
        c = np.zeros_like(h)
        last = 1024
        cached = None
        frame = 0
        emitted = 0
        while frame < valid:
            if cached is None:
                result = dec([np.array([[last]], np.int32), h, c])
                cached = result[dec.output(0)].copy()
                next_h = result[dec.output(1)].copy()
                next_c = result[dec.output(2)].copy()
            logits = joint([features[:, :, frame:frame+1].transpose(0, 2, 1).copy(), cached])
            scores = logits[joint.output(0)].reshape(-1)
            token = int(np.argmax(scores[:1025]))
            advance = int(np.argmax(scores[1025:1030]))
            if token != 1024:
                offset = left/16000 + frame*0.08
                if core_start/16000 <= offset < min(len(audio), core_start+7*16000)/16000:
                    tokens.append({'time': args.start+offset, 'id': token,
                                   'text': vocab.get(str(token), '').replace('▁', ' ')})
                last, h, c, cached = token, next_h, next_c, None
                emitted += 1
            # I token senza avanzamento mantengono il frame, con un limite anti-loop.
            if token == 1024 or emitted >= 10:
                advance = max(advance, 1)
            if advance:
                emitted = 0
            frame += advance
        print('Audio elaborato:', round(min(len(audio)/16000, core_start/16000+7), 2), flush=True)
    elapsed = time.perf_counter()-infer_started
    report = {'source': str(source.resolve()), 'start': args.start, 'end': end,
              'device': args.device, 'audio_seconds': end-args.start, 'load_seconds': load_seconds,
              'preparation_seconds': preparation, 'inference_seconds': elapsed,
              'total_seconds': time.perf_counter()-started, 'decoded_fraction': fraction,
              'speed': (end-args.start)/elapsed, 'tokens': tokens}
    (output/'risultato.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    (output/'trascrizione.md').write_text(render(tokens, args.device), encoding='utf8')
    print({k:v for k,v in report.items() if k!='tokens'}, flush=True)


if __name__ == '__main__':
    main()
