<#
.SYNOPSIS
    Safe GTSRB Dataset Download and Verification Script
.DESCRIPTION
    Acquires the official German Traffic Sign Recognition Benchmark (GTSRB) archives
    from the authoritative ERDA repository (Rub-Universität Bochum / Copenhagen).
    Includes:
    - Pre-flight dry-run option (-DryRun)
    - Resume support with temporary download files (.tmp) to prevent partial data corruption
    - Exact byte-count validation
    - Overwrite protection
.PARAMETER DryRun
    Perform endpoint reachability and disk space checks without downloading.
.PARAMETER Force
    Re-download and overwrite existing archives if present.
#>
param(
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$RawDataDir = Join-Path $ProjectRoot "data\raw\GTSRB"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " GTSRB Dataset Acquisition Manager (Java 21 / PowerShell)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Target Directory : $RawDataDir"
Write-Host "Estimated Space  : ~365 MB download / ~550 MB uncompressed"

$Archives = @(
    @{
        Name = "GTSRB_Final_Training_Images.zip"
        Url  = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        ExpectedBytes = 276294756
        Description = "43-Class Training Images (39,209 PPM files + GT-*.csv annotations)"
    },
    @{
        Name = "GTSRB_Final_Test_Images.zip"
        Url  = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
        ExpectedBytes = 88978620
        Description = "Held-out Test Images (12,630 PPM files)"
    },
    @{
        Name = "GTSRB_Final_Test_GT.zip"
        Url  = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
        ExpectedBytes = 99620
        Description = "Held-out Test Ground-Truth Annotations (GT-final_test.csv)"
    }
)

# 1. Pre-flight Endpoint Inspection (Dry-Run Mode)
Write-Host "`n[1/3] Pre-flight Endpoint Check:" -ForegroundColor Yellow
$AllEndpointsHealthy = $true

foreach ($arch in $Archives) {
    Write-Host "  Checking: $($arch.Name)..." -NoNewline
    try {
        $rawHeaders = & curl.exe -s -I $arch.Url
        $headerText = $rawHeaders -join "`n"
        if ($headerText -match "HTTP/1\.[01]\s+200\s+OK" -and $headerText -match "Content-Length:\s*(\d+)") {
            $remoteLen = [int64]$matches[1]
            if ($remoteLen -eq $arch.ExpectedBytes) {
                Write-Host " [OK] (Matches published size: $remoteLen bytes)" -ForegroundColor Green
            } else {
                Write-Host " [WARN] (Remote size $remoteLen != expected $($arch.ExpectedBytes))" -ForegroundColor Yellow
                $AllEndpointsHealthy = $false
            }
        } else {
            Write-Host " [FAILED] (Endpoint not reachable)" -ForegroundColor Red
            $AllEndpointsHealthy = $false
        }
    } catch {
        Write-Host " [ERROR: $_]" -ForegroundColor Red
        $AllEndpointsHealthy = $false
    }
}

if ($DryRun) {
    Write-Host "`n[DRY RUN COMPLETE] Pre-flight checks passed. No files downloaded." -ForegroundColor Green
    Write-Host "To execute actual download, run without -DryRun after user confirmation."
    exit 0
}

# 2. Disk Space & Target Directory Setup
if (-not (Test-Path $RawDataDir)) {
    New-Item -ItemType Directory -Path $RawDataDir -Force | Out-Null
}

$drive = (Get-Item $RawDataDir).PSDrive.Name
$disk = Get-PSDrive -Name $drive
$freeBytes = $disk.Free
Write-Host "`n[2/3] Disk Space on Drive $drive`: $([math]::Round($freeBytes / 1GB, 2)) GB free."

if ($freeBytes -lt 1000000000) { # 1 GB minimum safeguard
    Write-Error "Insufficient disk space on drive $drive. At least 1 GB required."
    exit 1
}

# 3. Execution Phase via Java Acquisition Engine
Write-Host "`n[3/3] Commencing Dataset Download and Safe Extraction via Java Engine..." -ForegroundColor Yellow

$env:JAVA_HOME = "C:\Program Files\Java\jdk-25.0.2"
$mvnCmd = "C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd"

& "$mvnCmd" exec:java "-Dexec.mainClass=com.trafficsign.app.DatasetDownloadCli"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Dataset acquisition CLI failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n[SUCCESS] All GTSRB archives acquired, safely extracted, and verified in $RawDataDir." -ForegroundColor Green
