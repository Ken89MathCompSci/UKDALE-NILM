$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms.DataVisualization

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $Root "dataset"
$Splits = @(
    @{ Name = "train"; Label = "Train - House 1" },
    @{ Name = "validation"; Label = "Validation - House 1" },
    @{ Name = "test"; Label = "Test - House 5" }
)
$Channels = @("aggregate", "dishwasher", "fridge", "microwave", "washing_machine")
$Colors = @{
    aggregate = [System.Drawing.Color]::FromArgb(31, 119, 180)
    dishwasher = [System.Drawing.Color]::FromArgb(214, 39, 40)
    fridge = [System.Drawing.Color]::FromArgb(44, 160, 44)
    microwave = [System.Drawing.Color]::FromArgb(255, 127, 14)
    washing_machine = [System.Drawing.Color]::FromArgb(148, 103, 189)
}

function New-LineChart {
    param(
        [int]$Width,
        [int]$Height,
        [string]$Title
    )

    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = $Width
    $chart.Height = $Height
    $chart.BackColor = [System.Drawing.Color]::White
    $chart.AntiAliasing = [System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles]::All
    $chart.TextAntiAliasingQuality = [System.Windows.Forms.DataVisualization.Charting.TextAntiAliasingQuality]::High

    $mainTitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $mainTitle.Text = $Title
    $mainTitle.Font = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
    $chart.Titles.Add($mainTitle)

    return $chart
}

function Add-GridChartArea {
    param(
        $Chart,
        [string]$AreaName,
        [string]$Title,
        [int]$Row,
        [int]$Col,
        [int]$Rows,
        [int]$Cols
    )

    $leftMargin = 4.0
    $topMargin = 8.0
    $cellWidth = 92.0 / $Cols
    $cellHeight = 86.0 / $Rows

    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea($AreaName)
    $area.Position.X = $leftMargin + ($Col * $cellWidth)
    $area.Position.Y = $topMargin + ($Row * $cellHeight)
    $area.Position.Width = $cellWidth - 1.4
    $area.Position.Height = $cellHeight - 1.6
    $area.BackColor = [System.Drawing.Color]::White
    $area.AxisX.Title = "Hours"
    $area.AxisY.Title = "W"
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisX.LabelStyle.Font = New-Object System.Drawing.Font("Segoe UI", 7)
    $area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font("Segoe UI", 7)
    $area.AxisX.TitleFont = New-Object System.Drawing.Font("Segoe UI", 7)
    $area.AxisY.TitleFont = New-Object System.Drawing.Font("Segoe UI", 7)
    $area.AxisX.Minimum = 0
    $area.AxisX.Maximum = 24
    $area.AxisX.Interval = 6
    $area.AxisY.IsStartedFromZero = $true

    $Chart.ChartAreas.Add($area)

    $titleObj = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $titleObj.Text = $Title
    $titleObj.DockedToChartArea = $AreaName
    $titleObj.IsDockedInsideChartArea = $false
    $titleObj.Font = New-Object System.Drawing.Font("Segoe UI", 8, [System.Drawing.FontStyle]::Bold)
    $Chart.Titles.Add($titleObj)
}

function Add-Series {
    param(
        $Chart,
        [string]$AreaName,
        [string]$SeriesName,
        [object[]]$Rows,
        [string]$Channel,
        [int]$Step = 1
    )

    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($SeriesName)
    $series.ChartArea = $AreaName
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::FastLine
    $series.BorderWidth = 1
    $series.Color = $Colors[$Channel]

    for ($i = 0; $i -lt $Rows.Count; $i += $Step) {
        $x = ($i * 6.0) / 3600.0
        $y = [double]$Rows[$i].$Channel
        [void]$series.Points.AddXY($x, $y)
    }

    $Chart.Series.Add($series)
}

function Save-OverviewPlot {
    param($DataBySplit)

    $chart = New-LineChart -Width 2200 -Height 1150 -Title "UK-DALE HF Power Signals by Split"

    for ($row = 0; $row -lt $Splits.Count; $row++) {
        $splitName = $Splits[$row].Name
        $splitLabel = $Splits[$row].Label
        $rows = $DataBySplit[$splitName]

        for ($col = 0; $col -lt $Channels.Count; $col++) {
            $channel = $Channels[$col]
            $areaName = "area_${row}_${col}"
            $title = "$splitLabel - " + ($channel -replace "_", " ")
            Add-GridChartArea -Chart $chart -AreaName $areaName -Title $title -Row $row -Col $col -Rows $Splits.Count -Cols $Channels.Count
            Add-Series -Chart $chart -AreaName $areaName -SeriesName "${splitName}_${channel}" -Rows $rows -Channel $channel
        }
    }

    $out = Join-Path $DataDir "ukdale_power_overview_ps.png"
    $chart.SaveImage($out, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $chart.Dispose()
    Write-Output $out
}

function Save-FirstHourPlot {
    param($DataBySplit)

    $chart = New-LineChart -Width 1700 -Height 1000 -Title "UK-DALE First Hour Detail"
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend("legend")
    $legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Bottom
    $legend.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $chart.Legends.Add($legend)

    for ($row = 0; $row -lt $Splits.Count; $row++) {
        $splitName = $Splits[$row].Name
        $splitLabel = $Splits[$row].Label
        $rows = @($DataBySplit[$splitName] | Select-Object -First 600)
        $areaName = "detail_$row"
        Add-GridChartArea -Chart $chart -AreaName $areaName -Title $splitLabel -Row $row -Col 0 -Rows $Splits.Count -Cols 1
        $chart.ChartAreas[$areaName].AxisX.Maximum = 1
        $chart.ChartAreas[$areaName].AxisX.Interval = 0.25

        foreach ($channel in $Channels) {
            Add-Series -Chart $chart -AreaName $areaName -SeriesName "${splitName}_${channel}" -Rows $rows -Channel $channel
            $series = $chart.Series["${splitName}_${channel}"]
            $series.LegendText = $channel
            $series.IsVisibleInLegend = ($row -eq 0)
            if ($channel -ne "aggregate") {
                $series.BorderWidth = 2
            }
        }
    }

    $out = Join-Path $DataDir "ukdale_first_hour_detail_ps.png"
    $chart.SaveImage($out, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $chart.Dispose()
    Write-Output $out
}

function Save-EnergySummary {
    param($DataBySplit)

    $chart = New-LineChart -Width 1300 -Height 650 -Title "UK-DALE Labelled Appliance Energy"
    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea("energy")
    $area.AxisX.MajorGrid.Enabled = $false
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.Title = "Wh"
    $area.AxisY.IsStartedFromZero = $true
    $chart.ChartAreas.Add($area)

    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend("legend")
    $legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Bottom
    $legend.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $chart.Legends.Add($legend)

    foreach ($app in $Channels[1..($Channels.Count - 1)]) {
        $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($app)
        $series.ChartArea = "energy"
        $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Column
        $series.Color = $Colors[$app]
        $series.BorderWidth = 0

        foreach ($split in $Splits) {
            $rows = $DataBySplit[$split.Name]
            $sum = 0.0
            foreach ($row in $rows) {
                $sum += [double]$row.$app
            }
            $wh = $sum * 6.0 / 3600.0
            [void]$series.Points.AddXY($split.Name, $wh)
        }

        $chart.Series.Add($series)
    }

    $out = Join-Path $DataDir "ukdale_energy_summary_ps.png"
    $chart.SaveImage($out, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $chart.Dispose()
    Write-Output $out
}

$DataBySplit = @{}
foreach ($split in $Splits) {
    $path = Join-Path $DataDir "UKDALE_HF_$($split.Name).csv"
    $DataBySplit[$split.Name] = @(Import-Csv $path)
}

Save-OverviewPlot -DataBySplit $DataBySplit
Save-FirstHourPlot -DataBySplit $DataBySplit
Save-EnergySummary -DataBySplit $DataBySplit
