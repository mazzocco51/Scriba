# Scriba

Trascrizione locale di lezioni e registrazioni da MP4 a Markdown con timestamp.
Funziona da terminale, senza un assistente LLM, account AI o API a pagamento.
Il download iniziale dei modelli richiede Internet; l'inferenza avviene sul PC.

## Funzionalità e stato

| Motore | Dispositivo | Coda e checkpoint | Lingue |
| --- | --- | --- | --- |
| faster-whisper | CPU o NVIDIA CUDA | Sì | Dipendono dal modello; `base` e `large-v3` multilingui |
| Whisper OpenVINO | Intel GPU | Sì | Profilo incluso distil-large-v3: inglese |
| whisper.cpp Vulkan | GPU compatibile | Sì | Dipendono dal modello GGML |
| Parakeet V2 OpenVINO | Intel NPU, GPU o CPU | No, prototipo separato | Inglese |

Il programma principale gestisce file, cartelle, intervalli, glossari, blocchi
sovrapposti e ripresa. La coda in background è implementata per Windows: un solo
lavoro alla volta, quattro thread CPU e priorità inferiore alla normale.
Parakeet è il percorso sperimentale più veloce provato su Intel Core Ultra 7 155H.
Non sono ancora implementati integrazione Notion, rielaborazione AI degli appunti,
notifiche o installazione del comando globale `trascrivi`.

## Installazione Windows

Usare Python 3.11 o 3.12 a 64 bit. Aprire PowerShell nella cartella Scriba:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyAV gestisce la lettura dei comuni MP4 senza richiedere un eseguibile ffmpeg
separato. Driver GPU/NPU compatibili sono necessari per i rispettivi motori.
I modelli, gli ambienti Python e gli eseguibili esterni non sono inclusi nel repository.

## Avvio rapido: CPU in background

```powershell
.\trascrivi.ps1 avvia "C:\Lezioni\lezione.mp4" --language it
.\trascrivi.ps1 stato ID
.\trascrivi.ps1 riprendi ID
```

Sostituire `ID` con il codice restituito da `avvia`. Il terminale si può chiudere.
Il modello predefinito della coda è `base`, scaricato al primo uso. Per maggiore
qualità specificare `--model large-v3`, con maggiori requisiti di memoria e tempo.
Usare `--language en` per l'inglese o omettere la lingua per il rilevamento automatico.

Se PowerShell blocca gli script, è possibile usare direttamente Python:

```powershell
.\.venv\Scripts\python.exe coda_trascrizioni.py avvia "C:\Lezioni\lezione.mp4" --language it
.\.venv\Scripts\python.exe coda_trascrizioni.py stato ID
```

La coda e i log sono in `.coda_trascrizioni`. Dopo un riavvio usare `riprendi ID`.
Lo stato mostra i minuti dei blocchi completati, non un aggiornamento parola per parola.
Gli eventuali errori sono registrati nel log del lavoro.

## Esecuzione in primo piano e intervalli

```powershell
.\trascrivi.ps1 "C:\Lezioni\lezione.mp4" --model base --language it
.\trascrivi.ps1 "C:\Lezioni" --recursive --model base --language en
.\trascrivi.ps1 "C:\Lezioni\lezione.mp4" --start 00:20:00 --end 00:25:00 --model base
.\trascrivi.ps1 "C:\Lezioni" --list-only
```

In primo piano il modello predefinito è `large-v3`; indicare `base` per una prova
leggera. `--glossary glossario_parallel.txt` aggiunge termini tecnici al prompt.
`--output` cambia la destinazione e `--chunk-minutes 5` riduce la quantità di lavoro
da ripetere dopo un'interruzione. `--offline` richiede modelli già scaricati.
CUDA richiede hardware NVIDIA e le librerie compatibili con faster-whisper.

## Intel GPU: OpenVINO

Installazione e download una tantum:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-openvino.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('OpenVINO/distil-whisper-large-v3-int8-ov', local_dir='models/distil-large-v3-int8', max_workers=4)"
.\trascrivi.ps1 avvia "C:\Lezioni\lecture.mp4" --engine openvino --language en
```

Il runtime registra la GPU effettiva, mantiene il modello in memoria e salva la
cache di compilazione. Il primo avvio può essere più lento. La preparazione del
blocco successivo sulla CPU si sovrappone all'inferenza GPU. Si usa un solo stream:
nel test locale due stream hanno peggiorato sia velocità sia risultato.

## Intel NPU: Parakeet V2 sperimentale

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-parakeet.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('FluidInference/parakeet-tdt-0.6b-v2-ov', local_dir='models/parakeet-v2', max_workers=4)"
.\.venv\Scripts\python.exe parakeet.py --input "C:\Lezioni\lecture.mp4" --output trascrizioni/lecture
```

Per l'ultima mezz'ora aggiungere `--last-minutes 30`. Per un estratto usare
`--start 600 --seconds 120` (secondi). Senza intervallo viene trascritto tutto il
file. `--model` accetta una cartella alternativa; `--device NPU` è il default.
La preparazione dello spettrogramma usa CPU, encoder/decoder/joint usano il
dispositivo richiesto e ne stampano il nome effettivo. La compilazione iniziale
NPU può richiedere minuti; gli avvii successivi riutilizzano la cache.

Il prototipo lavora in primo piano, non ha checkpoint e decodifica in RAM tutto
l'intervallo richiesto. I timestamp sono approssimativi; le finestre possono
perdere parti di parole ai confini. Specificare una destinazione diversa per ogni
lezione: `trascrizione.md` e `risultato.json` vengono sovrascritti nella stessa cartella.

## Vulkan (avanzato)

Servono un modello GGML e una build Vulkan di whisper.cpp:

```powershell
.\trascrivi.ps1 avvia "C:\Lezioni\lecture.mp4" --engine vulkan --whisper-cli "C:\Tools\whisper-cli.exe" --model "C:\Models\ggml-base.bin" --language en
```

I percorsi predefiniti sono `tools/whisper-cli.exe` e `models/ggml-base.bin`.
La CLI ordinaria ricarica il modello per ogni blocco. Per mantenere il modello
caricato è inclusa la patch locale `third_party/whisper-stream.patch` contro
whisper.cpp v1.9.5 (vedere `third_party/README.md`). Non sono distribuiti binari.

## Output e uso per lo studio

Il motore principale scrive in `trascrizioni/<file-id>/<configurazione>/`:

- `lezione.md`: testo, timestamp, metadati e passaggi da verificare.
- `chunks/*.json`: segmenti e parole riconosciuti, usati anche per riprendere.
- `manifest.json`: modello, versioni e parametri.
- `progress.json`: copertura e completamento.

Rilanciare lo stesso comando recupera i blocchi completati. Parametri o sorgenti
diversi generano una configurazione separata. Il glossario è un suggerimento:
nessun motore garantisce trascrizioni perfette. Verificare numeri, formule, termini
tecnici e informazioni d'esame. Non viene analizzato il contenuto visivo del video.
I Markdown si possono fornire a NotebookLM o a un LLM; il programma non li carica
automaticamente su servizi esterni e non produce autonomamente appunti rielaborati.

## Misure indicative

Su Intel Core Ultra 7 155H, per lo stesso estratto inglese di 10 minuti:

| Configurazione | Tempo totale |
| --- | --- |
| Whisper OpenVINO distil-large-v3 INT8 | 142,2 s |
| Parakeet V2 NPU con cache compilata | 38,5 s |

Parakeet ha inoltre trascritto un intervallo di 30 minuti in circa 119,2 s.
Sono misure locali con modelli diversi, senza valutazione WER su una trascrizione
di riferimento; non dimostrano equivalenza di qualità o prestazioni su altri PC.
Il primo test NPU di due minuti ha richiesto circa 163 s per caricamento/compilazione.

## Test e struttura

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`trascrivi_lezioni.py` contiene pipeline ed esportazione; `coda_trascrizioni.py`
gestisce i lavori; `whisper_openvino.py` e `whisper_vulkan.py` sono gli adattatori;
`parakeet.py` è il prototipo NPU. `benchmark_trascrizione.py --help` mostra le
opzioni per confrontare CPU e Vulkan su estratti dello stesso file.

## Pubblicazione su GitHub

Il progetto non include registrazioni o trascrizioni personali. `.gitignore`
esclude modelli, cache, ambiente, coda e media. Controllare sempre `git status`
prima di pubblicare, soprattutto se vengono aggiunti altri formati di file.
Creare un repository vuoto chiamato `scriba`, quindi dalla cartella del progetto:

```powershell
git init -b main
git add .
git commit -m "Initial Scriba project"
git remote add origin https://github.com/USERNAME/scriba.git
git push -u origin main
```

Sostituire `USERNAME`. La licenza del codice originale Scriba resta da scegliere
prima di distribuirlo come progetto open source. Le dipendenze e i modelli
mantengono le rispettive licenze; attribuzioni in `third_party/README.md`.
