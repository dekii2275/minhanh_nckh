# Chay lan luot E1 va E2. Log: outputs\e2_pose_x_retrain_20261124\run_log.txt
$ErrorActionPreference = 'Continue'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..') -ErrorAction Stop).Path
Set-Location -LiteralPath $repoRoot -ErrorAction Stop
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Khong tim thay Python tai $python. Tao moi truong .venv truoc khi chay."
}

$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$outputDir = 'outputs\e2_pose_x_retrain_20261124'
$log = Join-Path $outputDir 'run_log.txt'
$done = Join-Path $outputDir 'DONE.txt'
$failed = Join-Path $outputDir 'FAILED.txt'
New-Item -ItemType Directory -Path $outputDir -Force -ErrorAction Stop | Out-Null
Remove-Item -LiteralPath $done, $failed -ErrorAction SilentlyContinue

$steps = @(
    @('E1 infer test x_base x_relaxed', 'src\gt_baselines\e1_redetect_pose_20261124.py', 'infer', '--all-test', '--configs', 'x_base', 'x_relaxed'),
    @('E1 evaluate test (frozen)', 'src\gt_baselines\e1_redetect_pose_20261124.py', 'evaluate', '--all-test'),
    @('E2 infer trainval x_relaxed', 'src\gt_baselines\e2_pose_x_retrain_20261124.py', 'infer', '--split', 'trainval'),
    @('E2 fix test', 'src\gt_baselines\e2_pose_x_retrain_20261124.py', 'fix', '--split', 'test'),
    @('E2 fix trainval', 'src\gt_baselines\e2_pose_x_retrain_20261124.py', 'fix', '--split', 'trainval'),
    @('E2 features', 'src\gt_baselines\e2_pose_x_retrain_20261124.py', 'features'),
    @('E2 train', 'src\gt_baselines\e2_pose_x_retrain_20261124.py', 'train'),
    @('E2 evaluate', 'src\gt_baselines\e2_pose_x_retrain_20261124.py', 'evaluate')
)

for ($index = 0; $index -lt $steps.Count; $index++) {
    $step = $steps[$index]
    $name = "$($index + 1)/$($steps.Count) $($step[0])"
    $commandArgs = $step[1..($step.Length - 1)]
    "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') BAT DAU $name" | Tee-Object -FilePath $log -Append
    & $python @commandArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        "LOI o buoc $name (exit $exitCode)" | Tee-Object -FilePath $log -Append
        $name | Out-File -FilePath $failed
        exit $exitCode
    }
}

"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') XONG TAT CA" | Tee-Object -FilePath $log -Append
Get-Date | Out-File -FilePath $done
