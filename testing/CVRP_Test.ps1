<#
CVRP_Test.ps1
Sequential test suite for the CVRP solver.
Runs smoke tests across all crossover/mutation combos and data sets.
Validates exit codes and checks that fitness values appear in output.

Usage: pwsh testing/CVRP_Test.ps1
       pwsh testing/CVRP_Test.ps1 -Gens 1000 -Pop 100
#>

param(
    [string]$Python = "python",
    [string]$Driver = "driver.py",
    [int]$Gens = 500,
    [int]$Pop = 50
)

$ErrorActionPreference = "Stop"
$passed = 0
$failed = 0
$failures = @()

function Run-Test {
    param([string]$Name, [string]$Args)

    Write-Host -NoNewline "  $Name ... "
    $output = & $Python $Driver $Args.Split(" ") 2>&1 | Out-String

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL (exit code $LASTEXITCODE)" -ForegroundColor Red
        $script:failed++
        $script:failures += $Name
        return
    }

    if ($output -notmatch '"best_individual_fitness":\s*\d+') {
        Write-Host "FAIL (no fitness in output)" -ForegroundColor Red
        $script:failed++
        $script:failures += $Name
        return
    }

    Write-Host "PASS" -ForegroundColor Green
    $script:passed++
}

Write-Host "`n=== CVRP Test Suite (sequential) ===`n"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# --- Crossover tests (test_set, small and fast) ---
Write-Host "Crossover operators:" -ForegroundColor Cyan
$crossovers = @(
    @("-B", "best_route_xo"),
    @("-C", "cycle_xo"),
    @("-E", "edge_recomb_xo"),
    @("-O", "order_xo"),
    @("-X", "pmx_xo")
)
foreach ($cx in $crossovers) {
    Run-Test $cx[1] "data/test_set.ocvrp -g $Gens -p $Pop $($cx[0])"
}

# --- Mutation tests (test_set) ---
Write-Host "`nMutation operators:" -ForegroundColor Cyan
$mutations = @(
    @("-I", "inversion_mut"),
    @("-W", "swap_mut"),
    @("-G", "gvr_scramble_mut"),
    @("-K", "scramble_mut"),
    @("-T", "two_opt_mut"),
    @("-F", "or_opt_mut"),
    @("-D", "displacement_mut")
)
foreach ($mt in $mutations) {
    Run-Test $mt[1] "data/test_set.ocvrp -g $Gens -p $Pop $($mt[0])"
}

# --- Data set tests (default algorithms, quick runs) ---
Write-Host "`nData sets:" -ForegroundColor Cyan
$datasets = @(
    "data/test_set.ocvrp",
    "data/F-n45-k4.ocvrp",
    "data/F-n72-k4.ocvrp",
    "data/A-n54-k7.ocvrp",
    "data/A-n80-k10.ocvrp",
    "data/B-n78-k10.ocvrp"
)
foreach ($ds in $datasets) {
    $name = Split-Path $ds -Leaf
    Run-Test $name "$ds -g $Gens -p $Pop"
}

# --- Combo tests (crossover + mutation pairs on a mid-size set) ---
Write-Host "`nCombination tests (F-n45-k4):" -ForegroundColor Cyan
$combos = @(
    @("-O -T", "order_xo + two_opt_mut"),
    @("-X -D", "pmx_xo + displacement_mut"),
    @("-C -K", "cycle_xo + scramble_mut"),
    @("-E -F", "edge_recomb_xo + or_opt_mut"),
    @("-B -W", "best_route_xo + swap_mut")
)
foreach ($cb in $combos) {
    Run-Test $cb[1] "data/F-n45-k4.ocvrp -g $Gens -p $Pop $($cb[0])"
}

# --- Multi-run test ---
Write-Host "`nMulti-run:" -ForegroundColor Cyan
Run-Test "3 runs with -r 3" "data/test_set.ocvrp -g 200 -p 30 -r 3"

# --- CLI flag tests ---
Write-Host "`nCLI flags:" -ForegroundColor Cyan
Run-Test "verbose routes (-R)" "data/test_set.ocvrp -g 200 -p 30 -R"
Run-Test "save results (-S)" "data/test_set.ocvrp -g 200 -p 30 -S -o ./results/test_output"
Run-Test "print gens (-P -A)" "data/test_set.ocvrp -g 200 -p 30 -P -A"

$sw.Stop()
Write-Host "`n=== Results ===" -ForegroundColor Cyan
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
if ($failures.Count -gt 0) {
    Write-Host "Failed tests:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
}
Write-Host "Time: $([math]::Round($sw.Elapsed.TotalSeconds, 1))s`n"

exit $failed
