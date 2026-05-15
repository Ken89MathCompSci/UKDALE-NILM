$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms.DataVisualization

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $Root "dataset"
$TrainPath = Join-Path $DataDir "UKDALE_HF_train.csv"
$TargetPath = Join-Path $DataDir "UKDALE_HF_validation.csv"
$OutPath = Join-Path $DataDir "nilm_combo_validation_trace.png"
$MetricsPath = Join-Path $DataDir "nilm_combo_validation_metrics.csv"

$Appliances = @("dishwasher", "fridge", "microwave", "washing_machine")
$Colors = @{
    aggregate = [System.Drawing.Color]::FromArgb(31, 119, 180)
    reconstructed = [System.Drawing.Color]::FromArgb(30, 30, 30)
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

function Learn-Signatures {
    param([object[]]$Rows)

    $thresholds = @{
        dishwasher = 50.0
        fridge = 20.0
        microwave = 100.0
        washing_machine = 50.0
    }

    $signatures = @{}
    foreach ($app in $Appliances) {
        $values = New-Object System.Collections.Generic.List[double]
        foreach ($row in $Rows) {
            $value = [double]$row.$app
            if ($value -gt $thresholds[$app]) {
                $values.Add($value)
            }
        }

        $signatures[$app] = [math]::Max($thresholds[$app], (Get-Median -Values $values.ToArray()))
    }

    $baseValues = New-Object System.Collections.Generic.List[double]
    foreach ($row in $Rows) {
        $anyActive = $false
        foreach ($app in $Appliances) {
            if ([double]$row.$app -gt $thresholds[$app]) {
                $anyActive = $true
                break
            }
        }
        if (-not $anyActive) {
            $baseValues.Add([double]$row.aggregate)
        }
    }

    $signatures["_base"] = Get-Median -Values $baseValues.ToArray()
    return $signatures
}

function New-StateSpace {
    param([hashtable]$Signatures)

    $states = @()
    for ($mask = 0; $mask -lt [math]::Pow(2, $Appliances.Count); $mask++) {
        $sum = 0.0
        $state = [ordered]@{
            Mask = $mask
            LabelledPower = 0.0
        }

        for ($i = 0; $i -lt $Appliances.Count; $i++) {
            $app = $Appliances[$i]
            $isOn = (($mask -band (1 -shl $i)) -ne 0)
            $power = if ($isOn) { [double]$Signatures[$app] } else { 0.0 }
            $state[$app] = $power
            $sum += $power
        }

        $state.LabelledPower = $sum
        $states += [pscustomobject]$state
    }

    return $states
}

function Get-HammingDistance {
    param([int]$A, [int]$B)

    $x = $A -bxor $B
    $count = 0
    while ($x -gt 0) {
        $count += ($x -band 1)
        $x = $x -shr 1
    }
    return $count
}

function Invoke-ComboNilm {
    param(
        [object[]]$Rows,
        [hashtable]$Signatures
    )

    $states = New-StateSpace -Signatures $Signatures
    $switchPenalty = 120.0
    $expectedPower = New-Object double[] $states.Count
    $transitionCost = New-Object 'double[,]' $states.Count, $states.Count
    for ($s = 0; $s -lt $states.Count; $s++) {
        $expectedPower[$s] = [double]$Signatures["_base"] + [double]$states[$s].LabelledPower
        for ($p = 0; $p -lt $states.Count; $p++) {
            $transitionCost[$p, $s] = $switchPenalty * (Get-HammingDistance -A $states[$p].Mask -B $states[$s].Mask)
        }
    }

    $previousCosts = New-Object double[] $states.Count
    $backPointers = New-Object 'int[,]' $Rows.Count, $states.Count

    for ($s = 0; $s -lt $states.Count; $s++) {
        $previousCosts[$s] = [math]::Abs([double]$Rows[0].aggregate - $expectedPower[$s])
        $backPointers[0, $s] = -1
    }

    for ($t = 1; $t -lt $Rows.Count; $t++) {
        $currentCosts = New-Object double[] $states.Count
        for ($s = 0; $s -lt $states.Count; $s++) {
            $emission = [math]::Abs([double]$Rows[$t].aggregate - $expectedPower[$s])
            $bestCost = [double]::PositiveInfinity
            $bestPrev = 0

            for ($p = 0; $p -lt $states.Count; $p++) {
                $cost = $previousCosts[$p] + $emission + $transitionCost[$p, $s]
                if ($cost -lt $bestCost) {
                    $bestCost = $cost
                    $bestPrev = $p
                }
            }

            $currentCosts[$s] = $bestCost
            $backPointers[$t, $s] = $bestPrev
        }
        $previousCosts = $currentCosts
    }

    $bestLast = 0
    $bestLastCost = [double]::PositiveInfinity
    for ($s = 0; $s -lt $states.Count; $s++) {
        if ($previousCosts[$s] -lt $bestLastCost) {
            $bestLastCost = $previousCosts[$s]
            $bestLast = $s
        }
    }

    $path = New-Object int[] $Rows.Count
    $path[$Rows.Count - 1] = $bestLast
    for ($t = $Rows.Count - 1; $t -gt 0; $t--) {
        $path[$t - 1] = $backPointers[$t, $path[$t]]
    }

    $trace = @()
    for ($t = 0; $t -lt $Rows.Count; $t++) {
        $state = $states[$path[$t]]
        $row = [ordered]@{
            timestamp = $Rows[$t].timestamp
            aggregate = [double]$Rows[$t].aggregate
            reconstructed = [double]$Signatures["_base"] + [double]$state.LabelledPower
            actual_labelled = 0.0
            estimated_labelled = [double]$state.LabelledPower
        }

        foreach ($app in $Appliances) {
            $actual = [double]$Rows[$t].$app
            $estimate = [double]$state.$app
            $row["actual_$app"] = $actual
            $row["estimated_$app"] = $estimate
            $row.actual_labelled += $actual
        }

        $trace += [pscustomobject]$row
    }

    return $trace
}

function Add-Series {
    param(
        $Chart,
        [string]$AreaName,
        [string]$Name,
        [object[]]$Rows,
        [string]$Column,
        [System.Drawing.Color]$Color,
        [string]$DashStyle = "Solid",
        [int]$Width = 2
    )

    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($Name)
    $series.ChartArea = $AreaName
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::FastLine
    $series.Color = $Color
    $series.BorderWidth = $Width
    $series.BorderDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::$DashStyle

    for ($i = 0; $i -lt $Rows.Count; $i++) {
        $x = ($i * 6.0) / 3600.0
        $y = [double]$Rows[$i].$Column
        [void]$series.Points.AddXY($x, $y)
    }

    $Chart.Series.Add($series)
}

function Add-Area {
    param($Chart, [string]$Name, [string]$AreaTitle, [double]$Y, [double]$Height)

    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea($Name)
    $area.Position.X = 7
    $area.Position.Y = $Y
    $area.Position.Width = 88
    $area.Position.Height = $Height
    $area.AxisX.Minimum = 0
    $area.AxisX.Maximum = 24
    $area.AxisX.Interval = 3
    $area.AxisX.Title = "Hours"
    $area.AxisY.Title = "W"
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.IsStartedFromZero = $true
    $Chart.ChartAreas.Add($area)

    $titleObj = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $titleObj.Text = $AreaTitle
    $titleObj.DockedToChartArea = $Name
    $titleObj.IsDockedInsideChartArea = $false
    $titleObj.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
    $Chart.Titles.Add($titleObj)
}

function Save-TracePlot {
    param([object[]]$Trace, [hashtable]$Signatures)

    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = 1800
    $chart.Height = 1250
    $chart.BackColor = [System.Drawing.Color]::White
    $chart.AntiAliasing = [System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles]::All

    $mainTitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $mainTitle.Text = "Combinatorial NILM on UK-DALE Validation Split"
    $mainTitle.Font = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
    $chart.Titles.Add($mainTitle)

    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend("legend")
    $legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Bottom
    $legend.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $chart.Legends.Add($legend)

    Add-Area -Chart $chart -Name "total" -AreaTitle "Aggregate and Reconstructed Aggregate" -Y 8 -Height 22
    Add-Series -Chart $chart -AreaName "total" -Name "aggregate" -Rows $Trace -Column "aggregate" -Color $Colors.aggregate -Width 2
    Add-Series -Chart $chart -AreaName "total" -Name "reconstructed aggregate" -Rows $Trace -Column "reconstructed" -Color $Colors.reconstructed -DashStyle "Dash" -Width 2

    Add-Area -Chart $chart -Name "labelled" -AreaTitle "Actual vs Estimated Labelled Appliance Total" -Y 35 -Height 17
    Add-Series -Chart $chart -AreaName "labelled" -Name "actual labelled total" -Rows $Trace -Column "actual_labelled" -Color $Colors.aggregate -Width 2
    Add-Series -Chart $chart -AreaName "labelled" -Name "estimated labelled total" -Rows $Trace -Column "estimated_labelled" -Color $Colors.reconstructed -DashStyle "Dash" -Width 2

    $y = 58
    foreach ($app in $Appliances) {
        $title = "$app actual vs estimated, learned signature $([math]::Round($Signatures[$app], 1)) W"
        Add-Area -Chart $chart -Name $app -AreaTitle $title -Y $y -Height 9
        Add-Series -Chart $chart -AreaName $app -Name "actual $app" -Rows $Trace -Column "actual_$app" -Color $Colors[$app] -Width 2
        Add-Series -Chart $chart -AreaName $app -Name "estimated $app" -Rows $Trace -Column "estimated_$app" -Color $Colors[$app] -DashStyle "Dash" -Width 2
        $y += 10
    }

    $chart.SaveImage($OutPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $chart.Dispose()
}

function Save-Metrics {
    param([object[]]$Trace)

    $metrics = @()
    foreach ($app in $Appliances) {
        $absError = 0.0
        $actualEnergy = 0.0
        $estimatedEnergy = 0.0
        foreach ($row in $Trace) {
            $actual = [double]$row."actual_$app"
            $estimated = [double]$row."estimated_$app"
            $absError += [math]::Abs($actual - $estimated)
            $actualEnergy += $actual * 6.0 / 3600.0
            $estimatedEnergy += $estimated * 6.0 / 3600.0
        }

        $metrics += [pscustomobject]@{
            appliance = $app
            mae_watts = [math]::Round($absError / $Trace.Count, 2)
            actual_wh = [math]::Round($actualEnergy, 2)
            estimated_wh = [math]::Round($estimatedEnergy, 2)
            energy_error_wh = [math]::Round($estimatedEnergy - $actualEnergy, 2)
        }
    }

    $metrics | Export-Csv -Path $MetricsPath -NoTypeInformation
    return $metrics
}

$trainRows = @(Import-Csv $TrainPath)
$targetRows = @(Import-Csv $TargetPath)
$signatures = Learn-Signatures -Rows $trainRows
$trace = Invoke-ComboNilm -Rows $targetRows -Signatures $signatures
Save-TracePlot -Trace $trace -Signatures $signatures
$metrics = Save-Metrics -Trace $trace

Write-Output "Learned signatures:"
Write-Output ("  {0,-16} {1,8:N1} W" -f "base", $signatures["_base"])
foreach ($app in $Appliances) {
    Write-Output ("  {0,-16} {1,8:N1} W" -f $app, $signatures[$app])
}
Write-Output ""
Write-Output "Validation metrics:"
$metrics | Format-Table -AutoSize | Out-String | Write-Output
Write-Output $OutPath
Write-Output $MetricsPath
