# SynapStore installer for Windows: puts synapstore.exe on PATH, then runs
# `synapstore install` to register the MCP server and memory skill with every
# agent it finds.
#
#   irm https://raw.githubusercontent.com/vertuzz/synapstore/main/install.ps1 | iex
#
# Environment: SYNAPSTORE_INSTALL_DIR (default %LOCALAPPDATA%\synapstore\bin),
# SYNAPSTORE_VERSION (default latest), SYNAPSTORE_NO_SETUP=1, SYNAPSTORE_INSTALL_ARGS
# (arguments for `synapstore install`, e.g. "--embedder minilm").
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$repo = 'vertuzz/synapstore'
$dir = if ($env:SYNAPSTORE_INSTALL_DIR) { $env:SYNAPSTORE_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'synapstore\bin' }
$name = 'synapstore-x86_64-windows'
$base = if ($env:SYNAPSTORE_VERSION) { "https://github.com/$repo/releases/download/$env:SYNAPSTORE_VERSION" } else { "https://github.com/$repo/releases/latest/download" }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
New-Item -ItemType Directory $tmp | Out-Null
try {
    Write-Host "synapstore-install: downloading $base/$name.zip"
    Invoke-WebRequest "$base/$name.zip" -OutFile "$tmp\a.zip" -UseBasicParsing
    Invoke-WebRequest "$base/$name.zip.sha256" -OutFile "$tmp\a.sha256" -UseBasicParsing
    $want = ((Get-Content "$tmp\a.sha256" -Raw).Trim() -split '\s+')[0].ToLower()
    $got = (Get-FileHash "$tmp\a.zip" -Algorithm SHA256).Hash.ToLower()
    if ($want -ne $got) { throw "checksum mismatch for $name.zip" }
    Expand-Archive "$tmp\a.zip" -DestinationPath $tmp
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$tmp\$name\synapstore.exe" (Join-Path $dir 'synapstore.exe') -Force
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

$bin = Join-Path $dir 'synapstore.exe'
Write-Host "synapstore-install: installed $(& $bin --version) to $bin"
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not ($userPath -split ';' | Where-Object { $_ -eq $dir })) {
    $newPath = if ($userPath) { "$dir;$userPath" } else { $dir }
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Write-Host "synapstore-install: added $dir to your user PATH (open a new terminal)"
}
$env:Path = "$dir;$env:Path"

if ($env:SYNAPSTORE_NO_SETUP -ne '1') {
    [string[]]$installArgs = if ($env:SYNAPSTORE_INSTALL_ARGS) { $env:SYNAPSTORE_INSTALL_ARGS -split '\s+' | Where-Object { $_ } } else { @() }
    & $bin install @installArgs
}
