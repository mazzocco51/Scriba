"""Trascrizione locale di registrazioni in Markdown, con checkpoint per blocco."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import glob
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
RATE = 16000
SCHEMA = 1
MEDIA = {'.mp4', '.m4v', '.mkv', '.mov', '.wav', '.mp3', '.m4a', '.ogg'}


def atomic_text(path: Path, text: str):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def save_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def parse_time(value: str) -> float:
    try:
        result = 0.0
        fields = value.split(':')
        if len(fields) > 3:
            raise ValueError()
        for field in fields:
            result = result * 60 + float(field)
        if result < 0 or not math.isfinite(result):
            raise ValueError()
        return result
    except ValueError:
        raise argparse.ArgumentTypeError('Usare secondi oppure HH:MM:SS.')


def discover(inputs: list[str], recursive: bool) -> list[Path]:
    files = set()
    for item in inputs:
        candidates = [Path(p) for p in glob.glob(os.path.expanduser(item), recursive=recursive)]
        if not candidates:
            raise ValueError(f'Percorso senza risultati: {item}')
        for candidate in candidates:
            if candidate.is_dir():
                iterator = candidate.rglob('*') if recursive else candidate.iterdir()
                files.update(p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in MEDIA)
            elif candidate.suffix.lower() in MEDIA:
                files.add(candidate.resolve())
            else:
                raise ValueError(f'Formato non supportato: {candidate}')
    if not files:
        raise ValueError('Nessuna registrazione trovata.')
    return sorted(files, key=lambda p: str(p).casefold())


@contextmanager
def exclusive_run(path: Path):
    """Il sistema operativo rilascia il lock anche dopo un arresto anomalo."""
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Trascrizione gia in esecuzione: {path.parent}') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def audio_metadata(source: Path, av, track: int):
    with av.open(str(source)) as container:
        if track >= len(container.streams.audio):
            raise ValueError(f'Traccia audio {track} assente in {source.name}')
        stream = container.streams.audio[track]
        origin = float(container.start_time / av.time_base) if container.start_time is not None else 0.0
        if stream.duration is None:
            raise ValueError('Durata della traccia audio non disponibile. Rimuxare il file prima di procedere.')
        audio_start = float((stream.start_time or 0) * stream.time_base) - origin
        duration = audio_start + float(stream.duration * stream.time_base)
        if duration <= 0:
            raise ValueError('Traccia audio vuota.')
        return duration, origin


def decode_block(source, start, end, origin, track, av, np):
    """Colloca i campioni sui timestamp originali, conservando eventuali pause."""
    audio = np.zeros(round((end - start) * RATE), dtype=np.float32)
    filled = 0
    resampler = av.AudioResampler(format='s16', layout='mono', rate=RATE)

    def place(frame):
        nonlocal filled
        if frame.time is None:
            raise RuntimeError('Frame audio privo di timestamp.')
        values = frame.to_ndarray().reshape(-1)
        offset = round((float(frame.time) - origin - start) * RATE)
        left, right = max(0, offset), min(len(audio), offset + len(values))
        if right > left:
            audio[left:right] = values[left-offset:right-offset].astype(np.float32) / 32768.0
            filled += right - left

    with av.open(str(source)) as container:
        stream = container.streams.audio[track]
        if start > 0:
            container.seek(int((origin + start) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and float(frame.time) - origin > end + 0.5:
                break
            for output in resampler.resample(frame):
                place(output)
        for output in resampler.resample(None):
            place(output)
    if filled == 0:
        raise RuntimeError(f'Nessun campione decodificato tra {timestamp(start)} e {timestamp(end)}.')
    return audio, min(1.0, filled / len(audio))


def owned_segments(chunk: dict) -> list[dict]:
    """Assegna le parole del contesto sovrapposto al rispettivo blocco centrale."""
    result = []
    for segment in chunk['segments']:
        words = [w for w in segment['words']
                 if chunk['core_start'] <= (w['start'] + w['end']) / 2 < chunk['core_end']]
        if words:
            text = ''.join(w['word'] for w in words).strip()
            probabilities = [w['probability'] for w in words if w['probability'] is not None]
            reasons = []
            if segment['avg_logprob'] is not None and segment['avg_logprob'] < -0.8:
                reasons.append('bassa affidabilita del segmento')
            if probabilities and min(probabilities) < 0.35:
                reasons.append('parole incerte')
            if not probabilities:
                reasons.append('affidabilita numerica non disponibile')
            if segment['compression_ratio'] is not None and segment['compression_ratio'] > 2.4:
                reasons.append('possibile ripetizione')
            if segment['no_speech_prob'] is not None and segment['no_speech_prob'] > 0.6:
                reasons.append('possibile assenza di parlato')
            boundary = chunk['core_start']
            if boundary > chunk['range_start'] and words[0]['start'] <= boundary + 3:
                reasons.append('confine tra blocchi')
            if chunk['core_end'] < chunk['range_end'] and words[-1]['end'] >= chunk['core_end'] - 3:
                reasons.append('confine tra blocchi')
            result.append({'start': words[0]['start'], 'end': words[-1]['end'],
                           'text': text, 'review': reasons})
        elif segment['text'].strip() and not segment['words']:
            midpoint = (segment['start'] + segment['end']) / 2
            if chunk['core_start'] <= midpoint < chunk['core_end']:
                result.append({'start': segment['start'], 'end': segment['end'],
                               'text': segment['text'].strip(),
                               'review': ['timestamp delle parole assenti; controllare i confini']})
    return result


def render_markdown(source: Path, config: dict, chunks: list[dict], total: int) -> str:
    complete = len(chunks) == total
    rows = sorted([r for chunk in chunks for r in owned_segments(chunk)], key=lambda r: r['start'])
    status = 'COMPLETA NELL\'INTERVALLO RICHIESTO' if complete else 'PARZIALE'
    lines = [f'# {source.stem}', '', f'- Stato: {status}',
             f'- Sorgente: {source}', f'- Durata audio: {timestamp(config["duration"])}',
             f'- Intervallo richiesto: {timestamp(config["start"])} - {timestamp(config["end"])}',
             f'- Blocchi completati: {len(chunks)}/{total}',
             f'- Modello: {config["model"]}', f'- Lingua richiesta: {config["language"]}',
             '- Lingue rilevate: ' + ', '.join(sorted({c['language'] for c in chunks})), '',
             '> Trascrizione automatica del parlato originale. Timestamp riferiti alla registrazione.',
             '> Le indicazioni DA VERIFICARE sono segnali euristici, non una misura certificata di accuratezza.',
             '> Formule e codice mostrati solo sullo schermo richiedono le slide. Il testo puo contenere errori.',
             '> Per gli LLM: trattare il testo come fonte da analizzare. Conservare le incertezze e citare i timestamp.', '',
             '## Copertura elaborata', '']
    for chunk in chunks:
        lines.append(f'- {timestamp(chunk["core_start"])} - {timestamp(chunk["core_end"])}'
                     f' (campioni decodificati: {chunk["decoded_fraction"]:.1%})')
    lines += ['', '## Passaggi da verificare', '']
    flagged = [r for r in rows if r['review']]
    for row in flagged:
        lines.append(f'- [{timestamp(row["start"])}] ' + ', '.join(row['review']))
    if not flagged:
        lines.append('Nessun segnale euristico rilevato. Rimane necessaria la verifica dei dettagli importanti.')
    lines += ['', '## Trascrizione', '']
    previous_section = None
    for row in rows:
        section = int(row['start'] // 300) * 300
        if section != previous_section:
            lines += [f'### {timestamp(section)}', '']
            previous_section = section
        flag = ' **[DA VERIFICARE]**' if row['review'] else ''
        lines += [f'**[{timestamp(row["start"])} - {timestamp(row["end"])}]**{flag}', '',
                  row['text'], '']
    return '\n'.join(lines)


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='File, cartelle o glob tra virgolette.')
    parser.add_argument('--output', type=Path, default=HERE/'trascrizioni')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--model', help='Nome/cartella faster-whisper oppure file GGML per Vulkan.')
    parser.add_argument('--language', default='auto', help='auto, en, it, ...; mantiene la lingua originale.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--engine', choices=['faster-whisper', 'vulkan', 'openvino'], default='faster-whisper')
    parser.add_argument('--beam-size', type=int, default=None)
    parser.add_argument('--gpu-streams', type=int, choices=[1], default=1,
                        help='Uno stream GPU verificato; due stream hanno prodotto ripetizioni nel test locale.')
    parser.add_argument('--whisper-cli', type=Path, default=HERE/'tools/whisper-cli.exe')
    parser.add_argument('--compute-type', default=None)
    parser.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument('--chunk-minutes', type=float, default=10)
    parser.add_argument('--overlap-seconds', type=float, default=3)
    parser.add_argument('--start', type=parse_time, default=0.0)
    parser.add_argument('--end', type=parse_time)
    parser.add_argument('--audio-track', type=int, default=0)
    parser.add_argument('--glossary', type=Path, help='Termini tecnici in UTF-8, uno per riga.')
    parser.add_argument('--offline', action='store_true', help='Usa solo modelli gia scaricati.')
    parser.add_argument('--no-vad', action='store_true', help='Disabilita il filtro delle pause.')
    parser.add_argument('--list-only', action='store_true', help='Elenca i file senza caricare il modello.')
    args = parser.parse_args(argv)
    if args.beam_size is None:
        args.beam_size = 1 if args.engine == 'openvino' else 5
    if args.model is None:
        args.model = str(HERE/'models/ggml-base.bin') if args.engine == 'vulkan' else 'large-v3'
        if args.engine == 'openvino':
            args.model = str(HERE/'models/distil-large-v3-int8')
    if (not math.isfinite(args.chunk_minutes) or args.chunk_minutes <= 0
            or not math.isfinite(args.overlap_seconds) or args.overlap_seconds < 0
            or args.overlap_seconds >= args.chunk_minutes * 30
            or args.threads < 1 or args.audio_track < 0 or args.beam_size < 1
            or (args.end is not None and args.end <= args.start)):
        parser.error('Durate, numero di thread o traccia audio non validi.')
    return args


def main(argv=None, progress=None, sources=None):
    args = parse_options(argv)
    files = discover(args.inputs, args.recursive) if sources is None else [Path(p) for p in sources]
    if args.list_only:
        for source in files:
            print(source)
        return 0
    queue_dir = HERE/'.coda_trascrizioni'
    queue_dir.mkdir(exist_ok=True)
    while True:
        slot = exclusive_run(queue_dir/'inferenza.lock')
        try:
            slot.__enter__()
            break
        except RuntimeError:
            if progress is None:
                raise RuntimeError('Motore occupato. Usare trascrivi.ps1 avvia per accodare il lavoro.')
            time.sleep(1)
    try:
        return transcribe_files(args, files, progress)
    finally:
        slot.__exit__(None, None, None)


def transcribe_files(args, files, progress):
    try:
        import av
        import numpy as np
        import ctranslate2
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
    except ImportError as exc:
        raise RuntimeError('Installare le dipendenze con pip install -r requirements-trascrizione.txt') from exc

    prompt = None
    if args.glossary:
        terms = [line.strip() for line in args.glossary.read_text(encoding='utf-8-sig').splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
        prompt = ', '.join(terms)
        if len(prompt) > 1800:
            raise ValueError('Glossario troppo lungo. Selezionare al massimo circa 1800 caratteri di termini pertinenti.')

    device = args.device
    if device == 'auto':
        device = 'cuda' if ctranslate2.get_cuda_device_count() else 'cpu'
    compute = args.compute_type or ('float16' if device == 'cuda' else 'int8')
    local_model = Path(args.model).expanduser()
    model_name = str(local_model.resolve()) if local_model.is_dir() else args.model
    if args.engine == 'openvino':
        from whisper_openvino import OpenVINOModel
        model = OpenVINOModel(local_model, args.gpu_streams)
        model_name, device, compute = str(model.path), 'openvino', 'IR'
    elif args.engine == 'vulkan':
        from whisper_vulkan import VulkanModel
        if args.device != 'auto' or args.compute_type:
            raise ValueError('Con Vulkan il dispositivo e il formato numerico sono gestiti da whisper.cpp.')
        model = VulkanModel(args.whisper_cli, local_model, args.threads, args.output)
        model_name, device, compute = str(model.path), 'vulkan', 'ggml'
    elif not local_model.is_dir():
        model_name = str(Path(download_model(model_name, cache_dir=str(HERE/'.modelli_asr'),
                                             local_files_only=args.offline)).resolve())
    if args.engine == 'faster-whisper':
        print(f'Caricamento {model_name} su {device} ({compute})...', flush=True)
        model = WhisperModel(model_name, device=device, compute_type=compute, cpu_threads=args.threads,
                             download_root=str(HERE/'.modelli_asr'), local_files_only=args.offline)
    failures = []
    for source in files:
        try:
            if progress:
                progress(source, 0, None)
            process_file(source, args, model, model_name, device, compute, prompt, av, np, progress)
        except KeyboardInterrupt:
            print('\nInterrotto. Rilanciare lo stesso comando per riprendere dai blocchi completati.', flush=True)
            return 130
        except Exception as exc:
            failures.append(f'{source.name}: {exc}')
            print(f'ERRORE {source.name}: {exc}', file=sys.stderr, flush=True)
        finally:
            if args.engine == 'vulkan':
                model.close()
    if failures and progress:
        raise RuntimeError('\n'.join(failures))
    return 1 if failures else 0


def process_file(source, args, model, model_name, device, compute, prompt, av, np, progress=None):
    duration, origin = audio_metadata(source, av, args.audio_track)
    end = min(duration, args.end) if args.end is not None else duration
    if args.start >= end:
        raise ValueError('Inizio successivo alla fine della registrazione.')
    stat = source.stat()
    config = {'schema': SCHEMA, 'source': str(source), 'bytes': stat.st_size,
              'mtime_ns': stat.st_mtime_ns, 'duration': duration, 'start': args.start, 'end': end,
              'model': model_name, 'language': args.language, 'device': device, 'compute': compute,
              'chunk_seconds': args.chunk_minutes*60, 'overlap': args.overlap_seconds,
              'track': args.audio_track, 'prompt': prompt, 'vad': not args.no_vad,
              'faster_whisper': importlib.metadata.version('faster-whisper'), 'beam_size': args.beam_size,
              'av': av.__version__, 'ctranslate2': importlib.metadata.version('ctranslate2'),
              'temperature': [0.0, 0.2, 0.4], 'word_timestamps': True,
              'condition_on_previous_text': False}
    if device in ('vulkan', 'openvino'):
        for key in ('ctranslate2', 'temperature', 'condition_on_previous_text'):
            del config[key]
        config.update(model.config)
    if Path(model_name).is_dir():
        config['model_files'] = {name: [p.stat().st_size, p.stat().st_mtime_ns]
                                 for name in ['model.bin', 'config.json', 'tokenizer.json']
                                 if (p := Path(model_name)/name).exists()}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_id = hashlib.sha256(str(source).encode()).hexdigest()[:8]
    slug = re.sub(r'[^\w .-]', '_', source.stem)[:60].rstrip(' .')
    parent = args.output.resolve()/f'{slug}-{source_id}'
    # I vecchi manifest CPU non registravano i thread. Il comando storico
    # a otto thread continua a recuperare quei checkpoint quando presenti.
    if device not in ('vulkan', 'openvino') and not (args.threads == 8 and (parent/fingerprint[:12]/'manifest.json').exists()):
        config.update(engine='faster-whisper', threads=args.threads)
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = parent/fingerprint[:12]
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output/'run.lock'), ThreadPoolExecutor(max_workers=1) as decoder:
        def prepare(start, stop):
            audio, fraction = decode_block(source, start, stop, origin, args.audio_track, av, np)
            if hasattr(model, 'prepare_audio'):
                audio = model.prepare_audio(audio, not args.no_vad)
            return audio, fraction

        manifest = output/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
            raise RuntimeError('Configurazione checkpoint incompatibile.')
        save_json(manifest, {'fingerprint': fingerprint, 'config': config})
        checkpoint_dir = output/'chunks'
        checkpoint_dir.mkdir(exist_ok=True)
        total = math.ceil((end-args.start)/config['chunk_seconds'])
        chunks = []
        pending_audio = None

        def export(status):
            atomic_text(output/'lezione.md', render_markdown(source, config, chunks, total))
            save_json(output/'progress.json', {'status': status, 'completed_chunks': len(chunks),
                      'total_chunks': total, 'fingerprint': fingerprint,
                      'updated_utc': datetime.now(timezone.utc).isoformat()})
            if progress:
                progress(source, sum(c['core_end'] - c['core_start'] for c in chunks), output/'lezione.md')

        if not (output/'lezione.md').exists():
            export('partial')
        for index in range(total):
            checkpoint = checkpoint_dir/f'{index:05d}.json'
            core_start = args.start + index*config['chunk_seconds']
            core_end = min(end, core_start+config['chunk_seconds'])
            if checkpoint.exists():
                chunk = json.loads(checkpoint.read_text(encoding='utf-8'))
                if chunk['fingerprint'] != fingerprint or chunk['core_start'] != core_start or chunk['core_end'] != core_end:
                    raise RuntimeError(f'Checkpoint incompatibile: {checkpoint}')
                print(f'{source.name} | blocco {index+1}/{total} recuperato', flush=True)
            else:
                block_start = max(args.start, core_start-args.overlap_seconds)
                block_end = min(end, core_end+args.overlap_seconds)
                print(f'{source.name} | blocco {index+1}/{total} | {timestamp(core_start)} - {timestamp(core_end)}', flush=True)
                if pending_audio is None:
                    audio, decoded_fraction = prepare(block_start, block_end)
                else:
                    audio, decoded_fraction = pending_audio.result()
                    pending_audio = None
                # Prepara al massimo un blocco CPU mentre la GPU trascrive quello corrente.
                if index+1 < total and not (checkpoint_dir/f'{index+1:05d}.json').exists():
                    next_start = args.start + (index+1)*config['chunk_seconds']
                    pending_audio = decoder.submit(
                        prepare, max(args.start, next_start-args.overlap_seconds),
                        min(end, next_start+config['chunk_seconds']+args.overlap_seconds))
                segments, info = model.transcribe(
                    audio, language=None if args.language == 'auto' else args.language,
                    task='transcribe', beam_size=args.beam_size, temperature=[0.0, 0.2, 0.4],
                    vad_filter=not args.no_vad, vad_parameters={'min_silence_duration_ms': 1000},
                    word_timestamps=True, condition_on_previous_text=False,
                    initial_prompt=prompt, hallucination_silence_threshold=2.0)
                chunk = {'fingerprint': fingerprint, 'core_start': core_start, 'core_end': core_end,
                         'range_start': args.start, 'range_end': end,
                         'audio_start': block_start, 'audio_end': block_end,
                         'decoded_fraction': decoded_fraction, 'language': info.language, 'segments': []}
                for segment in segments:
                    chunk['segments'].append({
                        'start': segment.start+block_start, 'end': segment.end+block_start,
                        'text': segment.text, 'avg_logprob': segment.avg_logprob,
                        'compression_ratio': segment.compression_ratio,
                        'no_speech_prob': segment.no_speech_prob,
                        'words': [{'start': word.start+block_start, 'end': word.end+block_start,
                                   'word': word.word, 'probability': word.probability}
                                  for word in segment.words or []]})
                current = source.stat()
                if current.st_size != stat.st_size or current.st_mtime_ns != stat.st_mtime_ns:
                    raise RuntimeError('La sorgente e cambiata durante la trascrizione.')
                save_json(checkpoint, chunk)
                del audio
            chunks.append(chunk)
            export('complete' if len(chunks) == total else 'partial')
        print(f'Markdown: {output / "lezione.md"}', flush=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f'ERRORE: {error}', file=sys.stderr)
        sys.exit(1)
