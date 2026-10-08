# AI Video Studio - diagnostico do Windows (Fase 1)
#
# Uso (PowerShell, nao precisa ser administrador):
#   powershell -ExecutionPolicy Bypass -File scripts\diagnose.ps1
#
# Verifica Windows, CPU, RAM, GPU, disco, virtualizacao, WSL2 e ferramentas.
# Apenas le informacoes; nao instala nem altera nada.

$ErrorActionPreference = "SilentlyContinue"
$env:WSL_UTF8 = "1"  # faz o wsl.exe responder em UTF-8 em vez de UTF-16
$tips = New-Object System.Collections.Generic.List[string]

function Show-Line([string]$label, [string]$value) {
    Write-Host ("{0,-14}{1}" -f $label, $value)
}

function Get-ToolVersion([string]$name, [string]$arg) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cmd) { return $null }
    if ($cmd.Source -like "*WindowsApps*") { return "atalho da Microsoft Store (nao instalado de verdade)" }
    $out = & $cmd.Source $arg 2>&1 | Select-Object -First 1
    return ("" + $out).Trim()
}

Write-Host ""
Write-Host "AI Video Studio - diagnostico do Windows" -ForegroundColor Cyan
Write-Host ""

# --- Sistema --------------------------------------------------------------
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$build = [int]$os.BuildNumber
Show-Line "Windows" ("{0} (build {1})" -f $os.Caption, $os.BuildNumber)
if ($build -lt 19041) {
    $tips.Add("Windows antigo para WSL2: atualize para Windows 10 2004 (build 19041) ou mais novo.")
}

Show-Line "CPU" ("{0} - {1} threads" -f $cpu.Name.Trim(), $cpu.NumberOfLogicalProcessors)

$ramTotal = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
$ramFree = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
Show-Line "RAM" ("{0} GB total, {1} GB livres" -f $ramTotal, $ramFree)
if ($ramTotal -lt 8) {
    $tips.Add("Pouca RAM ($ramTotal GB): use clipes de ate 3 s em 480p.")
} elseif ($ramTotal -lt 16) {
    $tips.Add("$ramTotal GB de RAM: o WSL2 usa ate metade por padrao; clipes de 5 s em 480p devem funcionar.")
}

# --- GPU ------------------------------------------------------------------
$gpus = Get-CimInstance Win32_VideoController
foreach ($g in $gpus) { Show-Line "GPU" $g.Name }
$nvidia = $gpus | Where-Object { $_.Name -match "NVIDIA" }
$vram = 0
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $smi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($smi) {
        $parts = $smi.Split(",")
        $vram = [math]::Round([double]$parts[1].Trim() / 1024, 1)
        Show-Line "VRAM" ("{0} GB ({1})" -f $vram, $parts[0].Trim())
    }
}
if (-not $nvidia) {
    $tips.Add("Sem GPU NVIDIA: SAM 2.1 Tiny roda em CPU para clipes curtos; o Wan 2.2 Animate precisara de GPU remota gratuita (ex.: Kaggle).")
} elseif ($vram -gt 0 -and $vram -lt 16) {
    $tips.Add("GPU NVIDIA com $vram GB: boa para o SAM 2; o Wan 2.2 Animate deve rodar remotamente.")
} elseif ($vram -ge 16) {
    $tips.Add("GPU NVIDIA com $vram GB: da para testar o Wan 2.2 Animate quantizado localmente.")
}

# --- Disco ----------------------------------------------------------------
$drive = Get-PSDrive -Name ($env:SystemDrive.TrimEnd(":"))
$freeGb = [math]::Round($drive.Free / 1GB, 1)
Show-Line "Disco" ("{0} GB livres em {1}" -f $freeGb, $env:SystemDrive)
if ($freeGb -lt 40) {
    $tips.Add("Pouco espaco livre ($freeGb GB): WSL + modelos das proximas fases ocupam 30-60 GB.")
}

# --- Virtualizacao --------------------------------------------------------
$virt = $cpu.VirtualizationFirmwareEnabled
$hyper = $cs.HypervisorPresent
Show-Line "Virtualizacao" ("firmware: {0}  hypervisor ativo: {1}" -f $virt, $hyper)
if (-not $virt -and -not $hyper) {
    $tips.Add("Virtualizacao desligada: ative Intel VT-x / AMD-V (SVM) na BIOS para usar o WSL2.")
}

# --- Ferramentas no Windows -----------------------------------------------
Write-Host ""
$tools = [ordered]@{
    "python" = "--version"; "node" = "--version"; "npm" = "--version";
    "git" = "--version"; "ffmpeg" = "-version"; "claude" = "--version"
}
foreach ($t in $tools.Keys) {
    $v = Get-ToolVersion $t $tools[$t]
    if (-not $v) { $v = "nao encontrado" }
    Show-Line $t $v
}

# --- WSL ------------------------------------------------------------------
Write-Host ""
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
$ubuntu = $null
if (-not $wsl) {
    Show-Line "WSL" "nao instalado"
    $tips.Add("Instale o WSL2 com Ubuntu (PowerShell como administrador): wsl --install -d Ubuntu  e reinicie o PC.")
} else {
    $list = & wsl.exe -l -v 2>&1 | ForEach-Object { ("" + $_).Replace([char]0, "").Trim() } | Where-Object { $_ }
    $distros = @()
    foreach ($line in $list | Select-Object -Skip 1) {
        $cols = ($line.TrimStart("*").Trim()) -split "\s+"
        if ($cols.Count -ge 3) { $distros += [pscustomobject]@{ Name = $cols[0]; State = $cols[1]; Version = $cols[2] } }
    }
    if ($distros.Count -eq 0) {
        Show-Line "WSL" "instalado, sem distribuicoes"
        $tips.Add("Instale o Ubuntu no WSL: wsl --install -d Ubuntu")
    } else {
        foreach ($d in $distros) { Show-Line "WSL" ("{0} (WSL{1}, {2})" -f $d.Name, $d.Version, $d.State) }
        $ubuntu = $distros | Where-Object { $_.Name -like "Ubuntu*" } | Select-Object -First 1
        if (-not $ubuntu) {
            $tips.Add("Nenhuma distribuicao Ubuntu no WSL: wsl --install -d Ubuntu")
        } elseif ($ubuntu.Version -ne "2") {
            $tips.Add("O Ubuntu esta em WSL1. Converta com: wsl --set-version $($ubuntu.Name) 2")
        }
    }
}

if ($ubuntu -and $ubuntu.Version -eq "2") {
    # Sem aspas duplas no script: o PowerShell 5.1 as remove ao chamar programas nativos.
    $check = 'for c in python3 git ffmpeg node npm nvidia-smi; do if command -v $c >/dev/null; then echo $c: ok; else echo $c: FALTA; fi; done; if python3 -c ''import venv, ensurepip'' 2>/dev/null; then echo python3-venv: ok; else echo python3-venv: FALTA; fi'
    Write-Host "Ferramentas dentro do $($ubuntu.Name):"
    & wsl.exe -d $ubuntu.Name -e bash -lc $check 2>&1 | ForEach-Object { Write-Host ("  " + $_) }
    $tips.Add("Dentro do Ubuntu, instale o que faltar: sudo apt update && sudo apt install -y python3-venv python3-pip git ffmpeg")
    $tips.Add("Depois, no Ubuntu: clone o repositorio em ~/ (nao em /mnt/c) e rode: bash scripts/setup.sh")
}

# --- Resumo ---------------------------------------------------------------
Write-Host ""
Write-Host "Recomendacoes:" -ForegroundColor Cyan
if ($tips.Count -eq 0) { Write-Host "  - nenhuma" }
foreach ($tip in $tips) { Write-Host "  - $tip" }
Write-Host ""
Write-Host "Copie esta saida e cole no Claude Code para seguirmos com a instalacao."
