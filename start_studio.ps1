#requires -Version 5.1
<#
.SYNOPSIS
    On-demand launcher for Pierre's isolated Voice Studio (Qwen3-TTS) and Video Studio (MoneyPrinterTurbo).
.DESCRIPTION
    Spawns both studios manually in separate loopback-only console windows pinned strictly to GPU 1 (cuda:1).
    Leaves 0 MiB VRAM allocated when windows are closed; zero auto-start or scheduled tasks are registered.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = $PSScriptRoot
$VoiceDir = Join-Path $RootDir "voice_studio"
$VideoDir = Join-Path $RootDir "video_studio"
$VoicePython = Join-Path $VoiceDir ".venv\Scripts\python.exe"
$VoiceScript = Join-Path $VoiceDir "tts_server.py"
$VideoPython = Join-Path $VideoDir ".venv\Scripts\python.exe"
$VideoConfig = Join-Path $VideoDir "config.toml"
$VideoExampleConfig = Join-Path $VideoDir "config.example.toml"

if (-not (Test-Path $VoicePython)) {
    throw "Voice Studio venv Python not found at '$VoicePython'. Create voice_studio\.venv first."
}
if (-not (Test-Path $VideoPython)) {
    throw "Video Studio venv Python not found at '$VideoPython'. Create video_studio\.venv first."
}
if (-not (Test-Path $VideoConfig) -and (Test-Path $VideoExampleConfig)) {
    Copy-Item -Path $VideoExampleConfig -Destination $VideoConfig
    Write-Host "[Init] Copied config.example.toml -> video_studio\config.toml" -ForegroundColor Yellow
}

# Ensure WinGet FFmpeg binary folder is on PATH for both Voice Studio (MP3 transcode) and Video Studio.
$WingetFfmpegBin = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin"
if (-not (Test-Path $WingetFfmpegBin)) {
    $FfmpegExe = Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg*") -Filter "ffmpeg.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $FfmpegExe) {
        $WingetFfmpegBin = $FfmpegExe.DirectoryName
    }
}
if ((Test-Path $WingetFfmpegBin) -and ($env:PATH -split ";" | Where-Object { $_ -eq $WingetFfmpegBin }).Count -eq 0) {
    $env:PATH = "$WingetFfmpegBin;$env:PATH"
}

$PrevCudaVisible = $env:CUDA_VISIBLE_DEVICES
$PrevQwenDevice = $env:QWEN_TTS_DEVICE

try {
    # 1. Launch Voice Studio (Gradio 127.0.0.1:7860 + OpenAI-compatible TTS shim 127.0.0.1:7861) directly on cuda:1.
    if (Test-Path Env:CUDA_VISIBLE_DEVICES) {
        Remove-Item Env:CUDA_VISIBLE_DEVICES
    }
    $env:QWEN_TTS_DEVICE = "cuda:1"

    $VoiceProc = Start-Process `
        -FilePath $VoicePython `
        -ArgumentList @("`"$VoiceScript`"") `
        -WorkingDirectory $VoiceDir `
        -PassThru

    # 2. Launch Video Studio (MoneyPrinterTurbo 127.0.0.1:8501) masked strictly to physical GPU 1 for faster-whisper.
    $env:CUDA_VISIBLE_DEVICES = "1"

    $VideoProc = Start-Process `
        -FilePath $VideoPython `
        -ArgumentList @("-m", "streamlit", "run", ".\webui\Main.py", "--server.address=127.0.0.1", "--server.port=8501") `
        -WorkingDirectory $VideoDir `
        -PassThru
}
finally {
    if ($null -ne $PrevCudaVisible) {
        $env:CUDA_VISIBLE_DEVICES = $PrevCudaVisible
    } else {
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    }
    if ($null -ne $PrevQwenDevice) {
        $env:QWEN_TTS_DEVICE = $PrevQwenDevice
    } else {
        Remove-Item Env:QWEN_TTS_DEVICE -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "=== Pierre Training & Short-Form Video Studio (GPU 1 Isolated) ===" -ForegroundColor Cyan
Write-Host ("Voice Studio UI (Gradio):      http://127.0.0.1:7860        (PID {0})" -f $VoiceProc.Id) -ForegroundColor Green
Write-Host ("OpenAI-Compatible TTS API:     http://127.0.0.1:7861/v1/audio/speech (PID {0})" -f $VoiceProc.Id) -ForegroundColor Green
Write-Host ("Video Studio UI (Streamlit):   http://127.0.0.1:8501        (PID {0})" -f $VideoProc.Id) -ForegroundColor Green
Write-Host ("Local Ollama LLM Backend:      http://127.0.0.1:11434/v1    (on-demand)" ) -ForegroundColor DarkGray
Write-Host ""
Write-Host "Stop both windows when finished to release 100% of GPU 1 VRAM (0 MiB idle; zero auto-start)." -ForegroundColor Yellow
