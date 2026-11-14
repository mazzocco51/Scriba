# Usa .venv del progetto, oppure Python dal PATH.
$ErrorActionPreference = 'Stop'
$transcriptionPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $transcriptionPython)) {
    $transcriptionPython = (Get-Command python -ErrorAction Stop).Source
}
$entry = 'trascrivi_lezioni.py'
if ($args.Count -gt 0 -and $args[0] -in @('avvia','stato','riprendi')) {
    $entry = 'coda_trascrizioni.py'
}
& $transcriptionPython -X utf8 (Join-Path $PSScriptRoot $entry) @args
exit $LASTEXITCODE
