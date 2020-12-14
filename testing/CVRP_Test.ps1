<#
CVRP_Test.ps1
PowerShell Job script template to run multiple processes on Windows PowerShell

For more information on PowerShell jobs, visit:
https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/start-job?view=powershell-7.1
#>
Write-Host "STARTING PS1 SCRIPT..."

# Your python command or python file here
    <#
        EXAMPLE 1: python driver.py -p 800 -s 5 -g 100000 -m 0.15 -c 0.85 -i 2 -r 10 -SM -BI -f .\data\F-n72-k4.ocvrp
        Example 2: python custom_script.py
    #>

if ($LastExitCode -ne 0){
    Write-Host "FAILED PyCMD1"
    exit 1
}

# Your python command or python file here

if ($LastExitCode -ne 0){
    Write-Host "FAILED PyCMD2"
    exit 1
}
