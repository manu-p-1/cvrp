<#
CVRP_TestThreadedJob.ps1
PowerShell Threaded Job script template to run multiple processes on Windows PowerShell

This version is a multi-threaded and requires PowerShell 7. For more information visit:
https://docs.microsoft.com/en-us/powershell/module/threadjob/start-threadjob?view=powershell-7.1
#>
Write-Host "STARTING PS1 Threaded SCRIPT..."

# -ThrottleLimit is an optional argument to limit the number of jobs to 4
$j = Start-ThreadJob -Name PyCMD1 -ScriptBlock {
    # Your python command or python file here
    <#
        EXAMPLE 1: python driver.py -p 800 -s 5 -g 100000 -m 0.15 -c 0.85 -i 2 -r 10 -SM -BI -f .\data\F-n72-k4.ocvrp
        Example 2: python custom_script.py
    #>
    if($LASTEXITCODE -ne 0){ Write-Host "`nPyCMD1 FAILED..." exit 1}
     
} -ThrottleLimit 4

$h = Start-ThreadJob -Name PyCMD2 -ScriptBlock { 
    # Your python command or python file here
    if($LASTEXITCODE -ne 0){ Write-Host "`nPyCMD2 FAILED..." exit 1}
}

$i = Start-ThreadJob -Name PyCMD3 -ScriptBlock {
    # Your python command or python file here
    if($LASTEXITCODE -ne 0){ Write-Host "`nPyCMD3 FAILED..." exit 1}
     
}

$k = Start-ThreadJob -Name PyCMD4 -ScriptBlock { 
    # Your python command or python file here
    if($LASTEXITCODE -ne 0){ Write-Host "`nPyCMD4 FAILED..." exit 1}
}

Start-Sleep -s 5
Get-Job