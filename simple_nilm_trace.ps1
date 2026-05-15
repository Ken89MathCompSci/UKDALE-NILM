$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms.DataVisualization

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $Root "dataset"
$TrainPath = Join-Path $DataDir "UKDALE_HF_train.csv"
$TargetPath = Join-Path $DataDir "UKDALE_HF_test.csv"
$OutPath = Join-Path $DataDir "simple_nilm_trace.png"

$Appliances = @("dishwasher", "fridge", "microwave", "washing_machine")
$Colors = @{
    aggregate = [System.Drawing.Color]::FromArgb(31, 119, 180)
    estimated_total = [System.Drawing.Color]::FromArgb(34, 34, 34)
    dishwasher = [System.Drawing.Color]::FromArgb(214, 39, 40)
    fridge = [System.Drawing.Color]::FromArgb(44, 160, 44)
    microwave = [System.Drawing.Color]::FromArgb(255, 127, 14)
    washing_machine = [System.Drawing.Color]::FromArgb(148, 103, 189)
}

function Get-Median {
    param([double[]]$Values)

    if ($Values.Count -eq 0) {
        return 0.0
    }

    $sorted = @($Values | Sort-Object)
    $mid = [int]($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) {
        return [double]$sorted[$mid]
    }

    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

function Load-CsvRows {
    param([string]$Path)
    return @(Import-Csv $Path)
}

function Learn-ApplianceSignatures {
    param([object[]]$Rows)

    $thresholds = @{
        dishwasher = 50.0
        fridge = 20.0
        microwave = 100.0
        washing_machine = 50.0
    }

    $signatures = @{}
    foreach ($app in $Appliances) {
        $active = New-Object System.Collections.Generic.List[double]
        foreach ($row in $Rows) {
            $value = [double]$row.$app
            if ($value -gt $thresholds[$app]) {
                $active.Add($value)
            }
        }

        $median = Get-Median -Values $active.ToArray()
        if ($median -le 0) {
            $median = $thresholds[$app]
        }
        $signatures[$app] = $median
    }

    return $signatures
}

function Invoke-SimpleNilmTrace {
    param(
        [object[]]$Rows,
        [hashtable]$Signatures
    )

    $states = @{}
    foreach ($app in $Appliances) {
        $states[$app] = $false
    }

    $eventThreshold = 90.0
    $maxRelativeError = 0.65
    $estimated = @()
    $events = New-Object System.Collections.Generic.List[object]

    $previousAggregate = [double]$Rows[0].aggregate
    for ($i = 0; $i -lt $Rows.Count; $i++) {
        $aggregate = [double]$Rows[$i].aggregate
        $delta = $aggregate - $previousAggregate

        if ([math]::Abs($delta) -ge $eventThreshold) {
            $bestApp = $null
            $bestError = [double]::PositiveInfinity

            foreach ($app in $Appliances) {
                $signature = [double]$Signatures[$app]
                $error = [math]::Abs([math]::Abs($delta) - $signature)
                if ($error -lt $bestError) {
                    $bestError = $error
                    $bestApp = $app
                }
            }

            $allowedError = [math]::Max(120.0, [double]$Signatures[$bestApp] * $maxRelativeError)
            if ($bestError -le $allowedError) {
                $turnOn = $delta -gt 0
                $states[$bestApp] = $turnOn
                $events.Add([pscustomobject]@{
                    Index = $i
                    Time = $Rows[$i].timestamp
                    Delta = [math]::Round($delta, 1)
                    Appliance = $bestApp
                    State = $(if ($turnOn) { "ON" } else { "OFF" })
                })
            }
        }

        $estimateRow = [ordered]@{
            timestamp = $Rows[$i].timestamp
            aggregate = $aggregate
            estimated_total = 0.0
        }

        foreach ($app in $Appliances) {
            $estimatedPower = if ($states[$app]) { [double]$Signatures[$app] } else { 0.0 }
            $estimateRow[$app] = $estimatedPower
            $estimateRow.estimated_total += $estimatedPower
        }

        $estimated += [pscustomobject]$estimateRow
        $previousAggregate = $aggregate
    }

    return @{
        Trace = $estimated
        Events = $events
    }
}

function Add-LineSeries {
    param(
        $Chart,
        [string]$AreaName,
        [string]$Name,
        [object[]]$Rows,
        [string]$Column,
        [int]$Step = 1,
        [int]$Width = 2
    )

    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($Name)
    $series.ChartArea = $AreaName
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::FastLine
    $series.Color = $Colors[$Column]
    $series.BorderWidth = $Width

    for ($i = 0; $i -lt $Rows.Count; $i += $Step) {
        $x = ($i * 6.0) / 3600.0
        $y = [double]$Rows[$i].$Column
        [void]$series.Points.AddXY($x, $y)
    }

    $Chart.Series.Add($series)
}

function New-ChartArea {
    param(
        $Chart,
        [string]$Name,
        [string]$Title,
        [double]$Y,
        [double]$Height
    )

    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea($Name)
    $area.Position.X = 7
    $area.Position.Y = $Y
    $area.Position.Width = 88
    $area.Position.Height = $Height
    $area.AxisX.Title = "Hours from start"
    $area.AxisY.Title = "W"
    $area.AxisX.Minimum = 0
    $area.AxisX.Maximum = 24
    $area.AxisX.Interval = 3
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.IsStartedFromZero = $true
    $area.BackColor = [System.Drawing.Color]::White
    $Chart.ChartAreas.Add($area)

    $titleObj = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $titleObj.Text = $Title
    $titleObj.DockedToChartArea = $Name
    $titleObj.IsDockedInsideChartArea = $false
    $titleObj.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
    $Chart.Titles.Add($titleObj)
}

function Save-NilmPlot {
    param(
        [object[]]$Trace,
        [hashtable]$Signatures,
        [string]$Path
    )

    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = 1700
    $chart.Height = 1150
    $chart.BackColor = [System.Drawing.Color]::White
    $chart.AntiAliasing = [System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles]::All
    $chart.TextAntiAliasingQuality = [System.Windows.Forms.DataVisualization.Charting.TextAntiAliasingQuality]::High

    $mainTitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $mainTitle.Text = "Simple NILM Trace on UK-DALE Test Split"
    $mainTitle.Font = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
    $chart.Titles.Add($mainTitle)

    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend("legend")
    $legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Bottom
    $legend.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $chart.Legends.Add($legend)

    New-ChartArea -Chart $chart -Name "total" -Title "Aggregate vs Estimated Labelled Total" -Y 8 -Height 24
    Add-LineSeries -Chart $chart -AreaName "total" -Name "aggregate" -Rows $Trace -Column "aggregate" -Step 1 -Width 2
    Add-LineSeries -Chart $chart -AreaName "total" -Name "estimated labelled total" -Rows $Trace -Column "estimated_total" -Step 1 -Width 2

    $y = 39
    foreach ($app in $Appliances) {
        New-ChartArea -Chart $chart -Name $app -Title "$app estimate, signature $([math]::Round($Signatures[$app], 1)) W" -Y $y -Height 12
        Add-LineSeries -Chart $chart -AreaName $app -Name $app -Rows $Trace -Column $app -Step 1 -Width 2
        $chart.Series[$app].IsVisibleInLegend = $false
        $y += 13
    }

    $chart.SaveImage($Path, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $chart.Dispose()
}

$trainRows = Load-CsvRows -Path $TrainPath
$targetRows = Load-CsvRows -Path $TargetPath

$signatures = Learn-ApplianceSignatures -Rows $trainRows
$result = Invoke-SimpleNilmTrace -Rows $targetRows -Signatures $signatures
Save-NilmPlot -Trace $result.Trace -Signatures $signatures -Path $OutPath

$eventsPath = Join-Path $DataDir "simple_nilm_events.csv"
$result.Events | Export-Csv -Path $eventsPath -NoTypeInformation

Write-Output "Learned signatures:"
foreach ($app in $Appliances) {
    Write-Output ("  {0,-16} {1,8:N1} W" -f $app, $signatures[$app])
}
Write-Output "Detected events: $($result.Events.Count)"
Write-Output $OutPath
Write-Output $eventsPath
