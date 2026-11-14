# Scriba

Local lecture transcription from MP4 to timestamped Markdown. Runs from your terminal, with background jobs and resumable checkpoints on Windows. No AI subscription or API key required.

## Quick start

Requires 64-bit Python 3.11 or 3.12. In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\trascrivi.ps1 avvia "C:\Lectures\lecture.mp4" --language en
.\trascrivi.ps1 stato JOB_ID
.\trascrivi.ps1 riprendi JOB_ID
```

`avvia` queues a job and returns its ID; `stato` checks progress; `riprendi` resumes an interrupted job. You can close the terminal after submitting a job.

The background default is the multilingual Whisper `base` model, downloaded on first use. Add `--model large-v3` for a larger model, or use `--language it` for Italian. Transcripts and checkpoints are saved under `trascrizioni/`.

To transcribe an interval in the foreground:

```powershell
.\trascrivi.ps1 "C:\Lectures\lecture.mp4" --model base --start 00:20:00 --end 00:25:00
```

## Hardware acceleration

| Engine | Hardware | Background / resume |
| --- | --- | --- |
| faster-whisper | CPU or NVIDIA CUDA | Yes |
| Whisper OpenVINO | Intel GPU | Yes |
| whisper.cpp Vulkan | Compatible GPU | Yes |
| Parakeet V2 (experimental) | Intel NPU, GPU or CPU | No |

### Intel GPU

Install the optional runtime and download the English model once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-openvino.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('OpenVINO/distil-whisper-large-v3-int8-ov', local_dir='models/distil-large-v3-int8')"
.\trascrivi.ps1 avvia "C:\Lectures\lecture.mp4" --engine openvino --language en
```

### Intel NPU: Parakeet

Experimental English-only transcription, running in the foreground:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-parakeet.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('FluidInference/parakeet-tdt-0.6b-v2-ov', local_dir='models/parakeet-v2')"
.\.venv\Scripts\python.exe parakeet.py --input "C:\Lectures\lecture.mp4" --output trascrizioni/lecture
```

Add `--last-minutes 30` to transcribe only the final half hour. Use a different output directory for each recording. Initial NPU compilation can take several minutes; later runs reuse the cache. This prototype has approximate timestamps and no checkpoint recovery.

Vulkan requires a separate whisper.cpp build and GGML model, passed through `--whisper-cli` and `--model`. See [third-party notes](third_party/README.md) for the optional persistent-model patch and attribution.

## Notes

- Inference runs locally after model downloads. Compatible drivers are required for GPU/NPU acceleration.
- Output preserves the spoken language. Check technical terms, formulas and numbers against the recording; Scriba does not read slides or the board.
- Markdown is ready to use with study tools or LLMs. Automatic note rewriting and cloud uploads are not included.

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
