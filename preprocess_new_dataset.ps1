$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceCsv = Join-Path $Root "ukdale\UKDALE_House1_Final_6s.csv"
$OutDir = Join-Path $Root "new_dataset"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Splits = @(
    @{ name = "train"; date = "2014-09-11"; prefix = "09/11/2014" },
    @{ name = "validation"; date = "2014-11-09"; prefix = "11/09/2014" },
    @{ name = "test"; date = "2014-12-19"; prefix = "12/19/2014" }
)

$ExpectedRows = 14400

function Convert-Timestamp {
    param([string]$Value)

    $dt = [datetime]::ParseExact(
        $Value,
        "MM/dd/yyyy hh:mm:ss tt",
        [Globalization.CultureInfo]::InvariantCulture
    )
    return $dt.ToString("yyyy-MM-dd HH:mm:ss+00:00")
}

$writers = @{}
$counts = @{}
$dateToSplit = @{}

foreach ($split in $Splits) {
    $outPath = Join-Path $OutDir "UKDALE_HF_$($split.name).csv"
    $writer = [System.IO.StreamWriter]::new($outPath, $false, [System.Text.Encoding]::UTF8)
    $writer.WriteLine("timestamp,aggregate,dishwasher,fridge,microwave,washing_machine")
    $writers[$split.prefix] = $writer
    $counts[$split.name] = 0
    $dateToSplit[$split.prefix] = $split.name
}

Write-Output "Streaming source -> $SourceCsv"

$reader = [System.IO.StreamReader]::new($SourceCsv)
try {
    $header = $reader.ReadLine()
    if ($null -eq $header) {
        throw "Source CSV is empty: $SourceCsv"
    }

    $columns = $header.Split(",")
    $index = @{}
    for ($i = 0; $i -lt $columns.Count; $i++) {
        $index[$columns[$i]] = $i
    }

    foreach ($required in @("time", "Total_Power", "Dishwasher", "Fridge_Freezer", "Microwave", "Washer_Dryer")) {
        if (-not $index.ContainsKey($required)) {
            throw "Missing required column '$required' in $SourceCsv"
        }
    }

    while (($line = $reader.ReadLine()) -ne $null) {
        if ($line.Length -lt 10) {
            continue
        }

        $prefix = $line.Substring(0, 10)
        if (-not $writers.ContainsKey($prefix)) {
            continue
        }

        $parts = $line.Split(",")
        if ($parts.Count -lt $columns.Count) {
            continue
        }

        $timestamp = Convert-Timestamp -Value $parts[$index.time]
        $outLine = "{0},{1},{2},{3},{4},{5}" -f `
            $timestamp,
            $parts[$index.Total_Power],
            $parts[$index.Dishwasher],
            $parts[$index.Fridge_Freezer],
            $parts[$index.Microwave],
            $parts[$index.Washer_Dryer]

        $writers[$prefix].WriteLine($outLine)
        $splitName = $dateToSplit[$prefix]
        $counts[$splitName] += 1
    }
}
finally {
    $reader.Close()
    foreach ($writer in $writers.Values) {
        $writer.Close()
    }
}

$readmePath = Join-Path $OutDir "README.txt"
@"
UK-DALE NILM preprocessed split set

Source:
  ukdale/UKDALE_House1_Final_6s.csv

House:
  1

Sampling:
  6 seconds

Splits:
  train      2014-09-11
  validation 2014-11-09
  test       2014-12-19

Output columns:
  timestamp
  aggregate        <- Total_Power
  dishwasher       <- Dishwasher
  fridge           <- Fridge_Freezer
  microwave        <- Microwave
  washing_machine  <- Washer_Dryer
"@ | Set-Content -Path $readmePath -Encoding UTF8

Write-Output ""
Write-Output "Created files:"
foreach ($split in $Splits) {
    $path = Join-Path $OutDir "UKDALE_HF_$($split.name).csv"
    $count = $counts[$split.name]
    $status = if ($count -eq $ExpectedRows) { "OK" } else { "CHECK" }
    Write-Output ("  {0} rows={1} [{2}]" -f $path, $count, $status)
}
Write-Output "  $readmePath"
