# Teste do fluxo inteiro de scripts/instalar-windows.ps1 com o Windows simulado.
# Roda no PowerShell 7 (inclusive no Linux): pwsh -NoProfile -File scripts/tests/instalar-windows.test.ps1
# Funcoes globais com o nome de comandos (wsl.exe, Start-Process, Read-Host...) tem prioridade sobre
# os comandos reais, entao o instalador conversa com o "Windows" falso abaixo.

$ErrorActionPreference = "Stop"
$installer = Join-Path (Split-Path $PSScriptRoot -Parent) "instalar-windows.ps1"
$script:failures = 0

function Check([bool]$condition, [string]$message) {
    if ($condition) { Write-Host "  ok: $message" }
    else { Write-Host "  FALHOU: $message" -ForegroundColor Red; $script:failures++ }
}

function global:Get-CimInstance { param($ClassName) [pscustomobject]@{ BuildNumber = "22631" } }

function global:wsl.exe {
    $S = $global:S
    $S.Calls += ,($args -join " ")
    $global:LASTEXITCODE = 0
    if ($args[0] -eq "-l") {
        $z = [char]0
        $lines = @("  NAME      STATE      VERSION")
        if ($S.Installed) { $lines += "* Ubuntu    Stopped    $($S.Version)" }
        else { $lines = @("Windows Subsystem for Linux has no installed distributions.") }
        return $lines | ForEach-Object { $_.ToCharArray() -join $z }  # saida UTF-16 como no wsl.exe antigo
    }
    if ($args[0] -eq "--set-version") { $S.Version = "2"; return }
    if ($args[0] -eq "-d" -and $args.Count -eq 2) { $S.User = "guimo"; return }  # sessao interativa: cria usuario
    if ($args[0] -eq "-d" -and $args[2] -eq "-e") {
        $command = $args[-1]
        if ($command -eq "id -un") { return $S.User }
        $S.Setup = $command
        $global:LASTEXITCODE = $S.SetupCode
        return "saida do setup"
    }
    throw "chamada inesperada: wsl.exe $($args -join ' ')"
}

function global:Start-Process {
    param($FilePath, [string[]]$ArgumentList, $Verb, [switch]$Wait, [switch]$PassThru)
    $global:S.StartProcess += ,@{ File = $FilePath; Args = $ArgumentList; Verb = $Verb }
    if ($FilePath -eq "wsl.exe" -and $ArgumentList -contains "--install" -and $global:S.InstallWorks) {
        $global:S.Installed = $true
    }
    return [pscustomobject]@{ ExitCode = 0 }
}

function global:Read-Host { param($Prompt) $global:S.Prompts += ,$Prompt; return "" }

function global:New-Object {
    param([string]$TypeName, [string]$ComObject)
    $shell = [pscustomobject]@{}
    $shell | Add-Member ScriptMethod CreateShortcut {
        param($path)
        $lnk = [pscustomobject]@{ Path = $path; TargetPath = ""; Arguments = ""; WorkingDirectory = ""; Description = "" }
        $lnk | Add-Member ScriptMethod Save { $global:S.Shortcut = $this }
        return $lnk
    }
    return $shell
}

function Invoke-Scenario([string]$name, [hashtable]$state) {
    Write-Host "Cenario: $name" -ForegroundColor Cyan
    $root = Join-Path ([IO.Path]::GetTempPath()) ("aivs-test-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Force -Path (Join-Path $root "Desktop") | Out-Null
    $env:HOME = $root
    $env:LOCALAPPDATA = Join-Path $root "AppData"
    $env:SystemRoot = $root  # no Windows real: C:\Windows
    $defaults = @{
        Installed = $false; InstallWorks = $true; Version = "2"; User = "root"; SetupCode = 0
        Calls = @(); StartProcess = @(); Prompts = @(); Setup = $null; Shortcut = $null
    }
    foreach ($key in $state.Keys) { $defaults[$key] = $state[$key] }
    $global:S = [pscustomobject]$defaults
    $global:LASTEXITCODE = 0
    $output = & $installer 6>&1 | Out-String
    return @{ Output = $output; Exit = $global:LASTEXITCODE; Root = $root }
}

# 1) Windows sem Ubuntu: instala (como administrador), cria o usuario, roda o setup e cria o atalho.
$r = Invoke-Scenario "instalacao do zero" @{}
$install = $global:S.StartProcess | Where-Object { $_.File -eq "wsl.exe" } | Select-Object -First 1
Check ($null -ne $install) "pede a instalacao do WSL"
Check (($install.Args -join " ") -eq "--install -d Ubuntu") "argumentos do wsl --install: '$($install.Args -join ' ')'"
Check ($install.Verb -eq "RunAs") "pede permissao de administrador"
Check ($r.Output -notmatch "O  no WSL") "mensagens com o nome da distribuicao"
Check ($global:S.Calls -contains "-d Ubuntu") "abre o Ubuntu para criar o usuario"
Check ($global:S.Setup -match "git clone https://github.com/GuiMorais33/AI-Video-Studio.git ~/AI-Video-Studio") "clona o projeto"
Check ($global:S.Setup -match "bash scripts/setup.sh") "roda o setup"
Check ($global:S.Setup -notmatch '"') "comando do setup sem aspas duplas (PowerShell 5.1 as remove)"
Check ($null -ne $global:S.Shortcut -and $global:S.Shortcut.Path -like "*AI Video Studio.lnk") "cria o atalho na Area de Trabalho"
$launcher = Join-Path $env:LOCALAPPDATA "AIVideoStudio/abrir-estudio.ps1"
Check (Test-Path $launcher) "grava o lancador"
Check ((Get-Content $launcher -Raw) -match 'Distro = "Ubuntu"') "lancador usa o Ubuntu"
Check ($global:S.Shortcut.Arguments -match [regex]::Escape($launcher)) "atalho aponta para o lancador"
Check ($r.Output -match "Pronto!") "termina com sucesso"
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile($launcher, [ref]$null, [ref]$errors) | Out-Null
Check ($errors.Count -eq 0) "lancador gerado e PowerShell valido"

# 2) Ubuntu ja instalado em WSL1 e com usuario: converte, nao pede admin, setup falha -> sai com erro.
$r = Invoke-Scenario "setup com falha" @{ Installed = $true; Version = "1"; User = "guimo"; SetupCode = 3 }
Check ($global:S.StartProcess.Count -eq 0) "nao pede administrador quando o Ubuntu ja existe"
Check ($global:S.Calls -contains "--set-version Ubuntu 2") "converte para WSL2"
Check ($r.Exit -eq 1) "sai com codigo 1"
Check ($r.Output -match "falhou \(codigo 3\)") "mostra o codigo do setup"
Check ($null -eq $global:S.Shortcut) "nao cria atalho quando o setup falha"

# 3) Instalacao do WSL precisa de reinicio: o Ubuntu nao aparece depois do wsl --install.
$r = Invoke-Scenario "precisa reiniciar" @{ InstallWorks = $false }
Check ($r.Exit -eq 1) "sai com codigo 1"
Check ($r.Output -match "Reinicie o computador") "pede para reiniciar"

if ($script:failures) { Write-Host "$($script:failures) verificacao(oes) falharam" -ForegroundColor Red; exit 1 }
Write-Host "Todas as verificacoes passaram" -ForegroundColor Green
