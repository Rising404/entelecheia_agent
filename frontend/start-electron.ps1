param(
  [switch]$HiddenLauncher
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StateRoot = if ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA "Entelecheia"
} else {
  Join-Path ([System.IO.Path]::GetTempPath()) "Entelecheia"
}
$LogDir = Join-Path $StateRoot "Logs"
$LogFile = Join-Path $LogDir "launcher.log"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Write-LauncherLog {
  param([Parameter(Mandatory = $true)][string]$Message)

  $Line = "[{0}] {1}" -f (Get-Date -Format "o"), $Message
  Add-Content -LiteralPath $LogFile -Value $Line -Encoding UTF8
  if (-not $HiddenLauncher) {
    Write-Host $Message
  }
}

function Resolve-NodeExecutable {
  $RepoRoot = Split-Path -Parent $ScriptDir
  $Candidate = Join-Path $RepoRoot ".runtime\node\node.exe"
  if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
    return $Candidate
  }
  return $null
}

function Test-RendererBuildRequired {
  $DistIndex = Join-Path $ScriptDir "dist\index.html"
  if (-not (Test-Path -LiteralPath $DistIndex -PathType Leaf)) {
    return $true
  }

  $BuiltAt = (Get-Item -LiteralPath $DistIndex).LastWriteTimeUtc
  $Inputs = @(
    (Join-Path $ScriptDir "index.html"),
    (Join-Path $ScriptDir "package.json"),
    (Join-Path $ScriptDir "pnpm-lock.yaml"),
    (Join-Path $ScriptDir "vite.config.js")
  )
  $SourceRoot = Join-Path $ScriptDir "src"
  if (Test-Path -LiteralPath $SourceRoot -PathType Container) {
    $Inputs += Get-ChildItem -LiteralPath $SourceRoot -Recurse -File |
      ForEach-Object { $_.FullName }
  }
  foreach ($InputPath in $Inputs) {
    if (
      (Test-Path -LiteralPath $InputPath -PathType Leaf) -and
      (Get-Item -LiteralPath $InputPath).LastWriteTimeUtc -gt $BuiltAt
    ) {
      return $true
    }
  }
  return $false
}

try {
  Set-Location -LiteralPath $ScriptDir
  Write-LauncherLog "开始检查 Windows 桌面启动环境。"

  $NodeExe = Resolve-NodeExecutable
  if (-not $NodeExe) {
    throw "项目 Node 运行时不存在：仓库 .runtime\node\node.exe。"
  }

  $NodeVersionOutput = & $NodeExe --version 2>&1
  $NodeVersionExitCode = $LASTEXITCODE
  $NodeVersion = ([string]$NodeVersionOutput).Trim()
  if ($NodeVersionExitCode -ne 0 -or $NodeVersion -notmatch '^v(?<major>\d+)') {
    throw "无法确认 Node 版本。"
  }
  if ([int]$Matches["major"] -lt 20) {
    throw "Node 版本过低（$NodeVersion）；需要 Node 20+。"
  }

  $ViteBin = Join-Path $ScriptDir "node_modules\vite\bin\vite.js"
  $ElectronExe = Join-Path $ScriptDir "node_modules\electron\dist\electron.exe"
  if (-not (Test-Path -LiteralPath $ViteBin -PathType Leaf)) {
    throw "前端依赖未安装：缺少 Vite。请先按项目 lockfile 准备本地依赖。"
  }
  if (-not (Test-Path -LiteralPath $ElectronExe -PathType Leaf)) {
    throw "前端依赖未安装：缺少 Electron。请先按项目 lockfile 准备本地依赖。"
  }

  if (Test-RendererBuildRequired) {
    Write-LauncherLog "使用 $NodeVersion 构建渲染层。"
    $env:CI = "true"
    & $NodeExe $ViteBin build *>> $LogFile
    $BuildExitCode = $LASTEXITCODE
    if ($BuildExitCode -ne 0) {
      throw "前端构建失败（exit $BuildExitCode）。"
    }
  } else {
    Write-LauncherLog "渲染层未变化，复用现有构建。"
  }

  Write-LauncherLog "启动 Entelecheia 桌面应用。"
  $ElectronArguments = "`"$ScriptDir`""
  $Process = Start-Process `
    -FilePath $ElectronExe `
    -ArgumentList $ElectronArguments `
    -WorkingDirectory $ScriptDir `
    -PassThru
  Start-Sleep -Milliseconds 250
  if ($Process.HasExited -and $Process.ExitCode -ne 0) {
    throw "Electron 启动失败（exit $($Process.ExitCode)）。"
  }
  Write-LauncherLog "启动请求已交给 Electron（pid $($Process.Id)）。"
  exit 0
} catch {
  $Failure = $_.Exception.Message
  Write-LauncherLog "启动失败：$Failure"
  if (-not $HiddenLauncher) {
    Write-Error "$Failure`n日志：$LogFile"
  }
  exit 1
}
