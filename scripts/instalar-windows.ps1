# AI Video Studio - instalador para Windows (WSL2 + Ubuntu)
#
# Como usar (PowerShell normal, NAO precisa abrir como administrador):
#   irm https://raw.githubusercontent.com/GuiMorais33/AI-Video-Studio/HEAD/scripts/instalar-windows.ps1 -OutFile $env:TEMP\instalar-aivs.ps1
#   powershell -ExecutionPolicy Bypass -File $env:TEMP\instalar-aivs.ps1
#
# O que ele faz, pedindo confirmacao so quando o Windows exigir:
#   1. Instala o WSL2 com Ubuntu, se faltar (pede permissao de administrador e, as vezes, reinicio).
#   2. Abre o Ubuntu uma vez para voce criar usuario e senha, se ainda nao criou.
#   3. Confere o espaco livre no disco do Ubuntu (5 GB no minimo; 12 GB para usar placa NVIDIA).
#   4. Dentro do Ubuntu: baixa o projeto em ~/AI-Video-Studio e roda scripts/setup.sh
#      (instala Python, FFmpeg, PyTorch, SAM 2.1, Node.js e o painel; pede a senha do Ubuntu).
#   5. Cria o atalho "AI Video Studio" na Area de Trabalho para abrir o estudio.
# Pode rodar de novo quantas vezes quiser: ele continua de onde parou e atualiza o projeto.

param(
    [string]$Distro = "Ubuntu",
    [string]$RepoUrl = "https://github.com/GuiMorais33/AI-Video-Studio.git"
)

# "Continue": no PowerShell 5.1, "Stop" transforma qualquer texto em stderr de programas
# nativos (wsl.exe, git) em erro fatal. Os codigos de saida sao checados explicitamente.
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"

function Write-Step([string]$text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Stop-Installer([string]$text) {
    Write-Host ""
    Write-Host $text -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Pressione Enter para sair"
    exit 1
}

function Get-WslFreeGB([string]$name) {
    # Espaco livre no disco do Windows onde fica o disco virtual do Ubuntu (ext4.vhdx).
    # De dentro do Ubuntu o "df" nao enxerga esse limite. Padrao: o disco do AppData (C:).
    $path = "" + $env:LOCALAPPDATA
    $entry = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss\*" -ErrorAction SilentlyContinue |
        Where-Object { $_.DistributionName -eq $name } | Select-Object -First 1
    if ($entry -and $entry.BasePath) { $path = "" + $entry.BasePath }
    if (-not $path) { return $null }
    $letter = ($path -replace '^\\\\\?\\', '').Substring(0, 1)
    if ($letter -notmatch '^[A-Za-z]$') { return $null }
    $psDrive = Get-PSDrive -Name $letter -ErrorAction SilentlyContinue
    if (-not $psDrive -or $null -eq $psDrive.Free) { return $null }
    return [pscustomobject]@{ Letter = $letter.ToUpper(); GB = [math]::Round($psDrive.Free / 1GB, 1) }
}

function Get-WslDistros {
    $distros = @()
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { return $distros }
    $lines = & wsl.exe -l -v 2>&1 | ForEach-Object { ("" + $_).Replace([string][char]0, "").Trim() } | Where-Object { $_ }
    foreach ($line in $lines) {
        $cols = ($line.TrimStart("*").Trim()) -split "\s+"
        if ($cols.Count -ge 3 -and $cols[-1] -match "^[12]$") {
            $distros += [pscustomobject]@{ Name = $cols[0]; State = $cols[1]; Version = $cols[-1] }
        }
    }
    return $distros
}

Write-Host ""
Write-Host "AI Video Studio - instalador" -ForegroundColor Cyan
Write-Host "Projeto: $RepoUrl"

# --- 1. Windows e WSL -----------------------------------------------------
$build = [int](Get-CimInstance Win32_OperatingSystem).BuildNumber
if ($build -lt 19041) {
    Stop-Installer "Este Windows (build $build) e antigo para o WSL2. Atualize para Windows 10 2004 ou mais novo."
}

Write-Step "Verificando WSL e $Distro"
# Atencao: o PowerShell nao diferencia maiusculas de minusculas nos nomes de variaveis.
# $wslEntry nao pode se chamar $distro, senao apagaria o parametro $Distro ("Ubuntu").
$wslEntry = Get-WslDistros | Where-Object { $_.Name -eq $Distro } | Select-Object -First 1
if (-not $wslEntry) {
    Write-Host "O $Distro no WSL nao esta instalado. O Windows vai pedir permissao de administrador."
    Write-Host "Se uma janela do $Distro abrir pedindo usuario e senha, crie (guarde a senha) e volte aqui."
    $proc = Start-Process -FilePath "wsl.exe" -ArgumentList "--install", "-d", $Distro -Verb RunAs -Wait -PassThru
    Read-Host "Quando a instalacao terminar (e o usuario do $Distro tiver sido criado, se pedido), pressione Enter"
    $wslEntry = Get-WslDistros | Where-Object { $_.Name -eq $Distro } | Select-Object -First 1
    if (-not $wslEntry) {
        Stop-Installer ("A instalacao do WSL terminou (codigo $($proc.ExitCode)), mas o $Distro ainda nao aparece. " +
            "Reinicie o computador e rode este instalador de novo. Se pedir, ative a virtualizacao (Intel VT-x / AMD-V) na BIOS.")
    }
}
if ($wslEntry.Version -ne "2") {
    Write-Host "Convertendo $Distro para WSL2..."
    & wsl.exe --set-version $Distro 2
    if ($LASTEXITCODE -ne 0) { Stop-Installer "Nao foi possivel converter o $Distro para WSL2." }
}

# --- 2. Usuario do Ubuntu -------------------------------------------------
Write-Step "Verificando o usuario do $Distro"
$user = (& wsl.exe -d $Distro -e bash -lc "id -un" 2>$null | Select-Object -Last 1)
if ($user) { $user = ("" + $user).Trim() }
if (-not $user -or $user -eq "root") {
    Write-Host "Agora o $Distro vai abrir para voce criar um usuario e uma senha (guarde a senha)."
    Write-Host "Quando aparecer o prompt do Ubuntu (algo como usuario@PC:~$), digite: exit"
    Read-Host "Pressione Enter para abrir o $Distro"
    # Distribuicoes instaladas pela Microsoft Store criam o usuario no proprio lancador (ubuntu.exe).
    $launcherExe = Get-Command ("{0}.exe" -f $Distro.ToLower().Replace("-", "").Replace(".", "")) -ErrorAction SilentlyContinue
    if ($launcherExe) { & $launcherExe.Source } else { & wsl.exe -d $Distro }
    $user = (& wsl.exe -d $Distro -e bash -lc "id -un" 2>$null | Select-Object -Last 1)
    if ($user) { $user = ("" + $user).Trim() }
    if (-not $user -or $user -eq "root") {
        Stop-Installer "O usuario do $Distro ainda nao foi criado. Abra o '$Distro' pelo menu Iniciar, crie o usuario e rode este instalador de novo."
    }
}
Write-Host "Usuario do ${Distro}: $user"

# --- 3. Espaco em disco ----------------------------------------------------
# So-CPU: ~5 GB. Com a placa NVIDIA, o PyTorch baixa ~3 GB e ocupa ~7 GB instalado.
Write-Step "Verificando espaco em disco"
$torchEnv = ""
$space = Get-WslFreeGB $Distro
if ($space) {
    Write-Host "Livre no disco $($space.Letter): (onde fica o $Distro): $($space.GB) GB"
    if ($space.GB -lt 5) {
        Stop-Installer ("Pouco espaco no disco $($space.Letter): ($($space.GB) GB livres). O AI Video Studio precisa de pelo menos 5 GB " +
            "(12 GB para usar a placa NVIDIA). Libere espaco (Lixeira, Downloads, Configuracoes > Sistema > Armazenamento) " +
            "e rode este instalador de novo.")
    }
    if ($space.GB -lt 12) {
        Write-Host "Menos de 12 GB livres: instalando o PyTorch so-CPU (~200 MB), mesmo se houver placa NVIDIA."
        $torchEnv = "TORCH=cpu "
    }
} else {
    Write-Host "Nao foi possivel medir o espaco livre; seguindo assim mesmo."
}

# --- 4. Projeto e dependencias dentro do Ubuntu ---------------------------
Write-Step "Baixando o projeto e instalando dependencias (pode levar de 10 a 30 minutos)"
Write-Host "Quando pedir [sudo] password, digite a senha do Ubuntu (ela nao aparece enquanto voce digita)."
$setup = "set -e; " +
    "if ! command -v git >/dev/null; then sudo apt-get update && sudo apt-get install -y git; fi; " +
    "if [ -d ~/AI-Video-Studio/.git ]; then git -C ~/AI-Video-Studio pull --ff-only; " +
    "else git clone $RepoUrl ~/AI-Video-Studio; fi; " +
    "cd ~/AI-Video-Studio && " + $torchEnv + "bash scripts/setup.sh"
# Chamado direto (fora de funcao e sem capturar a saida): o progresso aparece na tela e o
# "[sudo] password" funciona no console. Sem aspas duplas em $setup: o PowerShell 5.1 as remove.
& wsl.exe -d $Distro -e bash -lc $setup
$code = $LASTEXITCODE
if ($code -ne 0) {
    Stop-Installer "A instalacao dentro do $Distro falhou (codigo $code). Veja as mensagens acima e rode o instalador de novo."
}

# --- 5. Atalho na Area de Trabalho ----------------------------------------
Write-Step "Criando o atalho 'AI Video Studio' na Area de Trabalho"
$appDir = Join-Path $env:LOCALAPPDATA "AIVideoStudio"
New-Item -ItemType Directory -Force -Path $appDir | Out-Null
$launcher = Join-Path $appDir "abrir-estudio.ps1"
$launcherBody = @'
# Abre o AI Video Studio: sobe motor + painel no WSL e abre o navegador quando estiver pronto.
param([string]$Distro = "__DISTRO__")
$env:WSL_UTF8 = "1"
Start-Job -ScriptBlock {
    for ($i = 0; $i -lt 240; $i++) {
        try {
            Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 2 | Out-Null
            Start-Process "http://localhost:3000"
            return
        } catch { Start-Sleep -Seconds 1 }
    }
} | Out-Null
Write-Host "AI Video Studio rodando. Feche esta janela (ou Ctrl+C) para encerrar."
& wsl.exe -d $Distro -e bash -lc "cd ~/AI-Video-Studio && bash scripts/start.sh"
'@
Set-Content -Path $launcher -Value $launcherBody.Replace("__DISTRO__", $Distro) -Encoding ASCII

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop "AI Video Studio.lnk"))
$lnk.TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$lnk.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`""
$lnk.WorkingDirectory = $appDir
$lnk.Description = "Abrir o AI Video Studio"
$lnk.Save()

Write-Host ""
Write-Host "Pronto! Use o atalho 'AI Video Studio' na Area de Trabalho." -ForegroundColor Green
Write-Host "O navegador abre sozinho em http://localhost:3000 quando o estudio estiver pronto."
Write-Host ""
Read-Host "Pressione Enter para sair"
