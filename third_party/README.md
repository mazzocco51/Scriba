# Componenti esterni

- faster-whisper: https://github.com/SYSTRAN/faster-whisper (MIT).
- OpenVINO / OpenVINO GenAI: https://github.com/openvinotoolkit (Apache-2.0).
- Il decoder TDT di `parakeet.py` è basato su eddy-audio:
  https://github.com/FluidInference/eddy-audio (Apache-2.0).
  Questa implementazione Python adatta la decodifica greedy, la normalizzazione
  dell'asse LogSoftmax e la gestione delle finestre. Non riproduce tutte le
  funzioni del progetto originale. Copia della licenza in `eddy-audio-LICENSE`.
- Modelli Parakeet: https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov
  (CC-BY-4.0 secondo la distribuzione del modello; pesi scaricati separatamente).
- whisper.cpp v1.9.5: https://github.com/ggml-org/whisper.cpp (MIT), licenza in
  `whisper.cpp-LICENSE`.

## Patch whisper.cpp

`whisper-stream.patch` conserva le modifiche locali alla v1.9.5:
include standard, log del dispositivo e protocollo stdin per riutilizzare il
modello fra blocchi. Applicare nella checkout di quella versione:

```powershell
git apply C:\percorso\Scriba\third_party\whisper-stream.patch
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release --target whisper-cli
```

Servono toolchain C++, CMake e Vulkan SDK compatibili. Copiare l'eseguibile
compilato e le relative DLL in una cartella locale e rinominare l'eseguibile
`whisper-stream-cli.exe`. Passarlo a Scriba con `--whisper-cli`: il nome attiva
il protocollo persistente. Non rinominare una build non modificata.
La build da sorgente non viene eseguita automaticamente dall'installazione Python.
