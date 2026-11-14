"""Coda Windows locale. SQLite coordina invii concorrenti e uscita del worker."""
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid

import trascrivi_lezioni as asr

ROOT = asr.HERE / '.coda_trascrizioni'


def database():
    ROOT.mkdir(exist_ok=True)
    db = sqlite3.connect(ROOT / 'lavori.sqlite', timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, args TEXT, cwd TEXT, status TEXT,
        source TEXT, seconds REAL DEFAULT 0, markdown TEXT, error TEXT)""")
    return db


def run_worker(db):
    db.execute('BEGIN IMMEDIATE')
    lock = asr.exclusive_run(ROOT / 'worker.lock')
    try:
        lock.__enter__()
    except RuntimeError:
        db.rollback()
        return
    try:
        db.execute("UPDATE jobs SET status='interrotto' WHERE status='in corso'")
        db.commit()
        while True:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM jobs WHERE status='in coda' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                # Libera il lock prima di consentire un nuovo INSERT: un
                # invio che arriva proprio all'uscita potrà avviare il successore.
                lock.__exit__(None, None, None)
                lock = None
                db.commit()
                return
            job = row['id']
            db.execute("UPDATE jobs SET status='in corso',error=NULL WHERE id=?", (job,))
            db.commit()

            def progress(source, seconds, markdown):
                db.execute('UPDATE jobs SET source=?,seconds=?,markdown=? WHERE id=?',
                           (str(source), seconds, str(markdown) if markdown else None, job))

            with (ROOT / f'{job}.log').open('a', encoding='utf8', buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    started = time.perf_counter()
                    print(f'Worker {os.getpid()} | avvio {time.strftime("%Y-%m-%d %H:%M:%S")}')
                    try:
                        os.chdir(row['cwd'])
                        request = json.loads(row['args'])
                        if isinstance(request, list):
                            request = {'options': request}
                        code = asr.main(request['options'], progress, request.get('sources'))
                        if code:
                            raise RuntimeError(f'Codice {code}. Vedere il log del lavoro.')
                    except (Exception, SystemExit) as exc:
                        print(f'Errore: {exc}', flush=True)
                        db.execute("UPDATE jobs SET status='errore',error=? WHERE id=?", (str(exc), job))
                    else:
                        db.execute("UPDATE jobs SET status='completato' WHERE id=?", (job,))
                    finally:
                        print(f'Tempo lavoro: {time.perf_counter()-started:.3f} s')
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


def wake():
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    with (ROOT / 'worker.log').open('ab') as log:
        subprocess.Popen([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '_worker'],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True,
                         creationflags=flags)


def main(command, *args):
    with contextlib.closing(database()) as db:
        return dispatch(db, command, args)


def dispatch(db, command, args):
    if command == '_worker':
        run_worker(db)
        return
    if command == 'avvia' and args:
        job = uuid.uuid4().hex[:16]
        options = ['--threads', '4', *args]
        # Profilo CPU leggero; il modello esplicito ha sempre precedenza.
        if not any(a.split('=')[0] in ('--model', '--engine') for a in args):
            options = ['--model', 'base', *options]
        parsed = asr.parse_options(options)
        sources = asr.discover(parsed.inputs, parsed.recursive)
        request = {'options': options, 'sources': [str(p) for p in sources]}
        db.execute("INSERT INTO jobs(id,args,cwd,status) VALUES(?,?,?,'in coda')",
                   (job, json.dumps(request), os.getcwd()))
    elif command in ('stato', 'riprendi') and len(args) == 1:
        job = args[0]
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM jobs WHERE id=?', (job,)).fetchone()
        if row is None:
            raise ValueError('ID sconosciuto')
        status = row['status']
        if status == 'in corso':
            try:
                with asr.exclusive_run(ROOT / 'worker.lock'):
                    db.execute("UPDATE jobs SET status='interrotto' WHERE id=? AND status='in corso'", (job,))
                    status = 'interrotto'
            except RuntimeError:
                pass
        db.commit()
        if command == 'stato':
            print(f"{job} | {status} | {row['seconds']/60:.2f} minuti elaborati della lezione corrente")
            for key in ('source', 'markdown', 'error'):
                if row[key]:
                    print(row[key])
            print(ROOT / f'{job}.log')
            return
        db.execute("UPDATE jobs SET status='in coda' WHERE id=? AND status IN ('errore','interrotto')", (job,))
        if status in ('completato', 'in corso'):
            print(job, status)
            return
    else:
        raise ValueError('Uso: avvia <file o cartella> [opzioni] | stato <ID> | riprendi <ID>')
    print(f'ID lavoro: {job}', flush=True)
    wake()


if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except Exception as error:
        sys.exit(str(error))
