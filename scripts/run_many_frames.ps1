param(
  [string]$ScenesRoot = "F:\dataset_practice\work\vggt_scenes\CoreView_390",
  [int]$Start = 0
  #[int]$End = 19
)

$logDir = Join-Path $ScenesRoot "_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

for ($i = $Start; $i -le $End; $i++) {
  $frameName = ("frame_{0:d6}" -f $i)
  $sceneDir = Join-Path $ScenesRoot $frameName
  $outSparse = Join-Path $sceneDir "sparse"

  if (!(Test-Path $sceneDir)) {
    Write-Host "[skip] missing $sceneDir"
    continue
  }

  # 如果已经有points3D.bin就跳过（可断点续跑）
  if (Test-Path (Join-Path $outSparse "points3D.bin")) {
    Write-Host "[skip] done $frameName"
    continue
  }

  $logPath = Join-Path $logDir ($frameName + ".log")
  Write-Host "[run] $frameName"

  python demo_colmap.py `
    --scene_dir "$sceneDir" `
    --use_ba `
    --query_frame_num 12 `
    --max_query_pts 1024 `
    --no_fine_tracking `
    *> $logPath

  if ($LASTEXITCODE -ne 0) {
    Write-Host "[fail] $frameName exit=$LASTEXITCODE (see $logPath)"
  }
  else {
    Write-Host "[ok] $frameName"
  }
}
