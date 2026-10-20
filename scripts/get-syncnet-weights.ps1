# Downloads the SyncNet weights (Oxford VGG) into weights\ for the optional lip-sync metric.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\get-syncnet-weights.ps1

$base = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data"
# full sha256 of the files verified; any other content is rejected
$expected = @{
    "syncnet_v2.model" = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
    "sfd_face.pth"     = "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491"
}

New-Item -ItemType Directory -Force -Path "weights" | Out-Null
$failed = $false
foreach ($f in $expected.Keys) {
    $out = Join-Path "weights" $f
    Write-Host "Downloading $f ..."
    curl.exe -L --fail -o $out "$base/$f"
    if ($LASTEXITCODE -ne 0) { Write-Error "Download failed: $f"; exit 1 }
    $hash = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
    "{0,-18} {1,12} bytes  sha256 {2}" -f $f, (Get-Item $out).Length, $hash
    if ($hash -ne $expected[$f]) {
        Write-Host "sha256 mismatch for ${f}: expected $($expected[$f])" -ForegroundColor Red
        Remove-Item $out -Force
        $failed = $true
    }
}
if ($failed) { Write-Host "Weights rejected (removed). Do not use them." -ForegroundColor Red; exit 1 }
Write-Host "All sha256 match."
