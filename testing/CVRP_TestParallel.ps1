<#
CVRP_TestParallel.ps1
Parallel test suite for the CVRP solver using PowerShell 7 ThreadJobs.
Runs a matrix of crossover x mutation x dataset combinations in parallel.
Validates exit codes and parses fitness from JSON output.

Requires: PowerShell 7+ (for Start-ThreadJob)
Usage: pwsh testing/CVRP_TestParallel.ps1
       pwsh testing/CVRP_TestParallel.ps1 -MaxParallel 8 -Gens 1000
#>

param(
    [string]$Python = "python",
    [string]$Driver = "driver.py",
    [int]$Gens = 500,
    [int]$Pop = 50,
    [int]$MaxParallel = 4
)

$ErrorActionPreference = "Stop"

# --- Build test matrix ---
$crossovers = @(
    @{ Flag = "-B"; Name = "best_route_xo" },
    @{ Flag = "-C"; Name = "cycle_xo" },
    @{ Flag = "-E"; Name = "edge_recomb_xo" },
    @{ Flag = "-O"; Name = "order_xo" },
    @{ Flag = "-X"; Name = "pmx_xo" }
)

$mutations = @(
    @{ Flag = "-I"; Name = "inversion_mut" },
    @{ Flag = "-W"; Name = "swap_mut" },
    @{ Flag = "-K"; Name = "scramble_mut" },
    @{ Flag = "-T"; Name = "two_opt_mut" },
    @{ Flag = "-F"; Name = "or_opt_mut" },
    @{ Flag = "-D"; Name = "displacement_mut" }
)

$datasets = @(
    "data/test_set.ocvrp",
    "data/F-n45-k4.ocvrp",
    "data/A-n54-k7.ocvrp"
)

# Build all test cases
$tests = @()

# Full crossover x mutation matrix on test_set (fast)
foreach ($cx in $crossovers) {
    foreach ($mt in $mutations) {
        $tests += @{
            Name = "$($cx.Name) + $($mt.Name) [test_set]"
            Args = "data/test_set.ocvrp -g $Gens -p $Pop $($cx.Flag) $($mt.Flag)"
        }
    }
}

# Each dataset with a couple of algorithm combos
foreach ($ds in $datasets) {
    $dsName = Split-Path $ds -Leaf
    $tests += @{
        Name = "order_xo + two_opt_mut [$dsName]"
        Args = "$ds -g $Gens -p $Pop -O -T"
    }
    $tests += @{
        Name = "pmx_xo + displacement_mut [$dsName]"
        Args = "$ds -g $Gens -p $Pop -X -D"
    }
}

# Multi-run and CLI flag tests
$tests += @{ Name = "multi-run (r=3)"; Args = "data/test_set.ocvrp -g 200 -p 30 -r 3" }
$tests += @{ Name = "verbose routes (-R)"; Args = "data/test_set.ocvrp -g 200 -p 30 -R" }
$tests += @{ Name = "save + output dir"; Args = "data/test_set.ocvrp -g 200 -p 30 -S -o ./results/test_parallel" }

Write-Host "`n=== CVRP Parallel Test Suite ($($tests.Count) tests, $MaxParallel workers) ===`n"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# --- Launch all jobs ---
$jobs = @()
foreach ($test in $tests) {
    $jobs += Start-ThreadJob -Name $test.Name -ThrottleLimit $MaxParallel -ScriptBlock {
        param($py, $drv, $argStr)
        $output = & $py $drv $argStr.Split(" ") 2>&1 | Out-String
        return @{
            ExitCode = $LASTEXITCODE
            Output   = $output
        }
    } -ArgumentList $Python, $Driver, $test.Args
}

# --- Collect results ---
$jobs | Wait-Job | Out-Null

$passed = 0
$failed = 0
$failures = @()

foreach ($job in $jobs) {
    $result = Receive-Job $job
    $name = $job.Name
    $ok = ($result.ExitCode -eq 0) -and ($result.Output -match '"best_individual_fitness":\s*\d+')

    if ($ok) {
        Write-Host "  PASS  $name" -ForegroundColor Green
        $passed++
    } else {
        $reason = if ($result.ExitCode -ne 0) { "exit $($result.ExitCode)" } else { "no fitness" }
        Write-Host "  FAIL  $name ($reason)" -ForegroundColor Red
        $failed++
        $failures += $name
    }
    Remove-Job $job
}

$sw.Stop()
Write-Host "`n=== Results ===" -ForegroundColor Cyan
Write-Host "Passed: $passed / $($passed + $failed)" -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })
if ($failures.Count -gt 0) {
    Write-Host "Failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
}
Write-Host "Time: $([math]::Round($sw.Elapsed.TotalSeconds, 1))s`n"

exit $failed