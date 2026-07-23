# ============================================================
#  NASDAQ-100 로고 수집기 v6  (Claude 생성)
#  위키류(나무위키/위키피디아) 전부 제외. 종목당 20장 목표.
#  소스: Clearbit 공식 로고 + DuckDuckGo + Bing (제목에 브랜드명 필수)
#  중복: MD5(완전 동일) + 지각해시 dHash(거의 동일한 리사이즈본) 제거
#  품질: 콜라주/파트너로고모음 등 키워드 배제 강화 +
#        다운로드 후 화질 검사(가로세로비, 투명배경/단색배경 여부)로
#        "로고 하나만 깔끔하게" 아닌 이미지 자동 제거
#  주의: 여러 로고가 같이 있는지(멀티 오브젝트)는 픽셀 휴리스틱만으로는
#        완전히 못 잡음 — 키워드 필터로 상당수 걸러내는 수준. 더 정확히
#        하려면 수집 후 VLM(Qwen2.5-VL)으로 2차 검수하는 걸 추천.
# ============================================================
$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.Drawing
$UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
$BAD = 'history|evolution|evoluc|timeline|through the years|over the years|old and new|all logos|logos of|brand logos|logo collection|collection of|logo pack|bundle|set of|comparison|versus| vs |infographic|chart|banner|wallpaper|mockup|collage|compilation|grid|sprite sheet|icon set|icon pack|top \d+|ranking|alternative(s)?|competitor|portfolio|showcase|our (client|partner|sponsor)|client list|sponsor|screenshot|storefront|building|sign(age)?|store front|변천|역사|모음|로고 모음|브랜드 모음|파트너사|고객사|스크린샷|매장|간판|건물'
$Root = Join-Path (Split-Path $PSScriptRoot -Parent) 'logos'
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Log = Join-Path (Split-Path $PSScriptRoot -Parent) 'collect_log.txt'
"===== v6 시작(단일 로고 필터 + 화질 검사)(위키 전체 제외, k=20): $(Get-Date) =====" | Out-File $Log -Encoding utf8 -Append
$K = 20
$md5 = [System.Security.Cryptography.MD5]::Create()
function LogLine([string]$m) { $m | Out-File $Log -Append -Encoding utf8 }
function Get-Ext([string]$u) {
  $p = ($u -split '\?')[0]
  if ($p -match '\.(svg|png|webp|jpg|jpeg|gif)$') { return $Matches[1].ToLower() }
  return 'png'
}
function Valid-Image([string]$path) {
  try {
    $len = (Get-Item $path).Length
    if ($len -lt 3072) { return $false }
    $head = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($path)[0..14])
    if ($head -match '<!DOC|<html') { return $false }
    return $true
  } catch { return $false }
}
function Test-CleanLogo([string]$path) {
  # 로고 하나만 깔끔하게 나오는지 검사: 1) 가로세로비가 콜라주/배너처럼 극단적이지 않은지
  # 2) 투명 배경이거나(알파채널) 테두리 색이 거의 단색(플랫 배경)인지
  $ext = [IO.Path]::GetExtension($path).ToLower()
  if ($ext -eq '.svg') { return $true }  # 벡터 로고(공식 브랜드 에셋)는 신뢰
  try {
    $img = [System.Drawing.Image]::FromFile($path)
    $w = $img.Width; $h = $img.Height; $pf = $img.PixelFormat
    $bmp = New-Object System.Drawing.Bitmap $img
    $img.Dispose()
  } catch { return $false }

  if ($w -lt 32 -or $h -lt 32) { $bmp.Dispose(); return $false }
  $ratio = [double]$w / [double]$h
  if ($ratio -gt 5.0 -or $ratio -lt 0.2) { $bmp.Dispose(); return $false }  # 배너/콜라주형 배제

  $pts = @(
    @(1, 1), @(($w - 2), 1), @(1, ($h - 2)), @(($w - 2), ($h - 2)),
    @([int]($w / 2), 1), @([int]($w / 2), ($h - 2)), @(1, [int]($h / 2)), @(($w - 2), [int]($h / 2))
  )
  $hasAlpha = [System.Drawing.Image]::IsAlphaPixelFormat($pf)
  if ($hasAlpha) {
    $transparent = 0
    foreach ($pt in $pts) { if ($bmp.GetPixel($pt[0], $pt[1]).A -lt 20) { $transparent++ } }
    $bmp.Dispose()
    return ($transparent -ge 4)  # 테두리 샘플 절반 이상 투명 = 로고만 분리되어 있음
  } else {
    $rs = @(); $gs = @(); $bs = @()
    foreach ($pt in $pts) { $c = $bmp.GetPixel($pt[0], $pt[1]); $rs += $c.R; $gs += $c.G; $bs += $c.B }
    $bmp.Dispose()
    $avgR = ($rs | Measure-Object -Average).Average
    $avgG = ($gs | Measure-Object -Average).Average
    $avgB = ($bs | Measure-Object -Average).Average
    $maxDiff = 0
    for ($i = 0; $i -lt $rs.Count; $i++) {
      $d = [Math]::Abs($rs[$i] - $avgR) + [Math]::Abs($gs[$i] - $avgG) + [Math]::Abs($bs[$i] - $avgB)
      if ($d -gt $maxDiff) { $maxDiff = $d }
    }
    return ($maxDiff -lt 45)  # 테두리 색상이 거의 균일 = 깔끔한 단색 배경 (사진 배경이면 편차 큼)
  }
}
function Get-DHash([string]$path) {
  try {
    $img = [System.Drawing.Image]::FromFile($path)
    $bmp = New-Object System.Drawing.Bitmap 9, 8
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.DrawImage($img, 0, 0, 9, 8); $g.Dispose(); $img.Dispose()
    $bits = New-Object System.Collections.BitArray 64
    $n = 0
    for ($y = 0; $y -lt 8; $y++) {
      for ($x = 0; $x -lt 8; $x++) {
        $c1 = $bmp.GetPixel($x, $y); $c2 = $bmp.GetPixel($x + 1, $y)
        $g1 = ($c1.R * 3 + $c1.G * 6 + $c1.B) / 10; $g2 = ($c2.R * 3 + $c2.G * 6 + $c2.B) / 10
        $bits[$n] = ($g1 -gt $g2); $n++
      }
    }
    $bmp.Dispose()
    return $bits
  } catch { return $null }
}
function Hamming($a, $b) {
  $d = 0
  for ($i = 0; $i -lt 64; $i++) { if ($a[$i] -ne $b[$i]) { $d++ } }
  return $d
}
function Try-Add([string]$url, [string]$path, [hashtable]$hashes, [System.Collections.ArrayList]$dhashes) {
  if (Test-Path $path) { return $false }
  try {
    Invoke-WebRequest -Uri $url -Headers @{ 'User-Agent' = $UA; 'Accept' = 'image/*,*/*;q=0.8' } -OutFile $path -TimeoutSec 25 -UseBasicParsing | Out-Null
  } catch { Remove-Item $path -Force -ErrorAction SilentlyContinue; return $false }
  if (-not (Valid-Image $path)) { Remove-Item $path -Force -ErrorAction SilentlyContinue; return $false }
  if (-not (Test-CleanLogo $path)) { Remove-Item $path -Force -ErrorAction SilentlyContinue; return $false }
  $hh = [BitConverter]::ToString($md5.ComputeHash([IO.File]::ReadAllBytes($path)))
  if ($hashes[$hh]) { Remove-Item $path -Force; return $false }
  $dh = Get-DHash $path
  if ($dh) {
    foreach ($e in $dhashes) { if ((Hamming $e $dh) -le 2) { Remove-Item $path -Force; return $false } }
    [void]$dhashes.Add($dh)
  }
  $hashes[$hh] = $true
  return $true
}
function Get-Html([string]$url) {
  try { return (Invoke-WebRequest -Uri $url -Headers @{ 'User-Agent' = $UA } -TimeoutSec 25 -UseBasicParsing).Content }
  catch { return $null }
}

$DATA = @"
NVDA|NVIDIA|nvidia|nvidia.com|Nvidia|https://i.namu.wiki/i/95-LYTrgyFBfmW2HUr9nwcA_JtfYQozFIk85rkCZX3sCb0msmlpgR6sI3BuyKpDoTebBiOqa1bItUgFGmHehiK2eQqBHcLImm0PJxDg5AfI8aUtO69-oGlBso5aWzH3pf1Mi3NQ3d0z-ZgER52t5Ig.svg
AAPL|Apple|apple|apple.com|Apple Inc.|https://i.namu.wiki/i/Mpgzwo5tgMGALoDJBROM6ww1jqAkK5qvSCkyrfdPCEVIrqklUknK83QPRwUQuqd5L1yQp6OUvbGpl5NiploO-oXnCffdHdBs1Mm3jdoFsJR2ZLwtIqgO_xocLE6Y4DUvOUqyoKAYpDjJ6Ruh1goLkQ.svg
MSFT|Microsoft|microsoft|microsoft.com|Microsoft|https://i.namu.wiki/i/cpNCK3zlRmB_BY9bUMab2dCO5WV7qsm9QiaQivuNWM8b_y14QeoGq7H0pKL_G-zVbZw3SxL8fgYnyfwIki1SmJsn0TfQ1QM1HvCmFdwq2-Cl_Rb7DlERbtMNc1YAf2c9mJ91aMWibjtCi4mSEZlUuQ.svg
AMZN|Amazon|amazon|amazon.com|Amazon (company)|https://i.namu.wiki/i/THsrCOQmnTucXC5uJTVP37pUcUL6PE4KskxA3ZSFoHf9rgskfXUgiBhJOpq1pJVqSy5t8J6rd_zSX5MnZqqT4Q.svg
GOOGL|Alphabet_Google|google|google.com|Google|https://i.namu.wiki/i/rL8ZjmAlBIzE9UbG50OK2eA7X5JG_pDC8pvGTptHxLfiqGtbuqEo1kP9jg03XkacgHL1g8WOeGdgcLUSfuGXX0iklA1a63k7vDSC3fbasdW4flhrS8MQmDgYKoSRNWc8kfzcos0vdDh_gic1fFFa5w.svg
AVGO|Broadcom|broadcom|broadcom.com|Broadcom|https://i.namu.wiki/i/IcqEcH8uGZwpuGhPVfEGNCVKQFPSq3AO2hn1MhPb2n9d3-HcWzqDVOdnvvJhtZuBwl0qd_HZ-gjpVLGpdZZT1WLh2bqZErV4t2c03FxHC0qXSH1StJ1jA2Lx6jw0Rae5swFHj3nW04q5RPjLgfWEvA.svg
META|Meta|meta platforms|meta.com|Meta Platforms|https://i.namu.wiki/i/oSD5di-TslvdS0DSrsk3K3svwE4dUqn8q1DN5369CfNYsvTvNwFpFvWRjnZd_FB5039-oMhn120UqLT5qRzbw7fCK0EMiCDlZUzak0rosmNo1_8DP1iVpMTjbVBvwe9VR-bOwORq4zvNm-wvb06cXA.svg
SPCX|SpaceX|spacex|spacex.com|SpaceX|
TSLA|Tesla|tesla|tesla.com|Tesla, Inc.|https://i.namu.wiki/i/aSlMx22fnoVjJrRyBo74tOHgDFEVtFqhSmB0STHh75A9tlVMavMHGfK_3d6KKn39Pxqt7-EF_P7o-WKEkE5qrnq7Vn1r4-OrB7HB7c4evXfsUqWZYqYFmLVTLSdVXRGmnGu9B9Ye9OmaorTbgwxlrw.svg
MU|Micron|micron|micron.com|Micron Technology|
WMT|Walmart|walmart|walmart.com|Walmart|https://i.namu.wiki/i/CWuWTeqRrplFRpzA-TbxeWK5KZD2H9nQMb54oz3AIevqWboZBmbJ-rDFLJ6deePA3p2QycAwvlX5BC1wnA9H5uNEIOwcQdsiCOTVF9qy0EUd_QZL_gb29OVribNhqeJzqSaWiCrUV9I5B0t12K4Zdg.svg
AMD|AMD|amd|amd.com|AMD|https://i.namu.wiki/i/gOcDZsltu-iJjjIkxmlvIIfnO9xIDIZqy30_HolWcn6cY6FzqGkrQ9Fr-7uQ9oxwgBFTkoFU2ggIu_O4pjhbEV0nHa6-nee6nwaRTNxDjr5d4cIzHE_hY_SAqvBPKuBjA_6CNjyIki6qdDM4eGhuJg.svg
ASML|ASML|asml|asml.com|ASML Holding|https://i.namu.wiki/i/rIBC4U-LHdP5-23Ao-UOD20POUT-LAMbWW3d85OmHFPFGjIREavOprd6-8nuZfsIyQ3J0sFGOBEe4Mu3_1aSHwRox1hnvb3FyeLsjJqwcZokzjlequt-0WoWqn0o69kHERxUtorWEwIPob4SYx5VZg.svg
INTC|Intel|intel|intel.com|Intel|https://i.namu.wiki/i/-rhiygt49BT6Jab0D-Ud9cn-XUOC4G8JHQkIDS2Xs0MjshiH9uHUlpck3lu8YP0pR_5zwl-LveD9jbeufWrcV-9F8Qf4X_l8VvlqDvsyBUmFiaaBWoUw7CBEOyZiHR0Y4qVqZrwkf1YU4qMddtOptA.svg
CSCO|Cisco|cisco|cisco.com|Cisco|https://i.namu.wiki/i/IPT9k7VbOAfNA4TN_evS0oBy9Wo-U-OSiwhpq82wKWgZ8pbO2QPMzcz_uTPblGG7Qgz__0rh7pVH7NLk9TGQtiDu0HJN09HkFT2GkfbLH-Z6RGKcuqS-glI30bUimBPKyQA-LMBNuEB7TbKy987WLw.svg
AMAT|Applied_Materials|applied materials|appliedmaterials.com|Applied Materials|https://i.namu.wiki/i/Kd2gQ6GoJ9Obhhr6ys3qilrIZ_ZYZ1y0T00TS3RJLDPGoO4f61kQ4yzia3IRME_nGdr3fuMmUeS8v7hBgmNsXw.svg
COST|Costco|costco|costco.com|Costco|https://i.namu.wiki/i/nc4ibgyxuaSTUhTUaoaWD1g_PgJry8KbjXIXJhJzK7nEc9PwnIQJ7MQlJaHBe_X48KluVTuCB0xemaDND0oNMbUc9AplTcAaypHj-RzGYFEKazsoI_QnGNqxvbwi6Ym9Ti-KYH7Im6GDNRTkCdgEAQ.svg
LRCX|Lam_Research|lam research|lamresearch.com|Lam Research|https://i.namu.wiki/i/1831mQUPgnGGJkO1uvCHpw5CxsjNuwhG1Wtv3O7gE558h33c8hTaSMc8SI_eftSe5rus50JfoZNjn2eJb2twbB4lmiO6dJTSvNR_1HX1bpoV5UrrNnXKoAEK4umhDCY1yK8Cpf0PHaosMFG5NPNJKQ.svg
PLTR|Palantir|palantir|palantir.com|Palantir Technologies|
PANW|Palo_Alto_Networks|palo alto networks|paloaltonetworks.com|Palo Alto Networks|
NFLX|Netflix|netflix|netflix.com|Netflix|https://i.namu.wiki/i/6rQOiUYmL4VJ4oWvOkxLIYwiRzjYhYTiYSFndgBHyYYSC6GZl4z2vSptbd_kqf7rBWqc3kAvvHxG_VrezomH7A.svg
ARM|Arm|arm holdings|arm.com|Arm Holdings|
KLAC|KLA|kla|kla.com|KLA Corporation|https://i.namu.wiki/i/pOjQSDiduJKRGq9Gl_KXmcrfmby3_kaa1eucXlxGpGpl2l0WzFua8qUYnsVWlITt6FBhDeAbNOZJgRHHHytR0R4zdcroPuWZGAoriEpy7ZX8STJp49CaxLM9WNvHEQSPxVR-om02QQkKWH8L8KqpdQ.svg
TXN|Texas_Instruments|texas instruments|ti.com|Texas Instruments|https://i.namu.wiki/i/_i5LwbK-f4c0EG5XbZmEjEkOTU8wf-D2N19YaTuk6BRV1_lzAOkjDHeLXZ6l-L30cBDESCBYcn_3kRfE2AwMkIp7orq0FuTFL39-ClVhUebVgU1VJiG3FhXMSSrsNs-ObQUH1EvSYtQdIMjSVkcTRg.svg
LIN|Linde|linde|linde.com|Linde plc|
TMUS|T-Mobile|t-mobile|t-mobile.com|T-Mobile US|
CRWD|CrowdStrike|crowdstrike|crowdstrike.com|CrowdStrike|https://i.namu.wiki/i/baR4MiOwGTizmfftiiH20z73pLqRQFUYteRCybiEDDk8VmsUheKZqMhSLscYRlb6nXcHYPQkl2wIf1M5CjHLCQ.svg
SNDK|Sandisk|sandisk|sandisk.com|SanDisk|https://i.namu.wiki/i/5siqfjQEcvuE-P1vH2hsYiEKuih8Is4BaOrP3GYW4mCLBof-TsMhsNjas2LCURIBscvLEPzBSWxghWi0mw8oGQzGiBZJ7If2HOvEizrMD-6WnNhJpFQke-LuLGRFYmLMd6xlKRg7du4iRikbPo3gTg.svg
AMGN|Amgen|amgen|amgen.com|Amgen|https://i.namu.wiki/i/rX6tvPWGeTd6jUe8Dz_DGGj5D4lqReN7E76yzRZ-qBhBH3QR_a8Zgfja5C3eF3FTZGmBIUgJfD7ouvnRgj2hz7Vb3WVHV05uKvLykYXvMEma1vaGf5JPsAEheidMhh4ctUqmKbbgA68eRChibof3gA.svg
PEP|PepsiCo|pepsico|pepsico.com|PepsiCo|https://i.namu.wiki/i/MaodpRujsRr2O4x8ln_-hEVEHTBlU47VOFAYTtz7IWhAx2xXJLvT1ek5ZMkd4oDLmGGRgCm5JbcDd-Gzi39Xhr3oEhO2C8ey1LoPMaO2LkpOz5EIE060vBOzj0peLvuK9C5rr8fi08mtNW8mmvL9Yg.svg
ADI|Analog_Devices|analog devices|analog.com|Analog Devices|
QCOM|Qualcomm|qualcomm|qualcomm.com|Qualcomm|https://i.namu.wiki/i/TI_xfwHUBYB8VZ1VI3UYN0snVebPdO2vLFzJAWA6Ec56XfE3aldtE3EvYDVdCezBu2o9fdzcXhFALfON84NW0evfvPP5whdLWh0vCboF3XwZSZKBcpNsCVf5C7-XaVNPEEMfqDHJ3LcWTm3DCv3weA.svg
STX|Seagate|seagate|seagate.com|Seagate Technology|https://i.namu.wiki/i/HnrJuIW5NXcMVc8euhj8YXW1ZfviiRSxl8RDGeNlG47V0QeZIr6TbMyr8mdRsbWkeEwDuyo-VcTIAQtjSFzwBw.svg
MRVL|Marvell|marvell|marvell.com|Marvell Technology|https://i.namu.wiki/i/HwXWLKyKG9fsEctMozAYqnpILA-G6FHbGD-o8mLyYD_Pn3nFJ1qSTVBx0WMigt9PXAXfLnM5WsMA-uBx1mgOvmppTH_1vzh8jRDqh5JgmtGPsL2LoyyTfvpHnFJKW7EPTO6ZxeyBhdMI8co2hlP7KA.svg
GILD|Gilead|gilead|gilead.com|Gilead Sciences|https://i.namu.wiki/i/ad2v4Fs2T4u7aMkiLlL0GcTVUMnKSXrb9eBY1zGOs1uTqck3-kvXPi-QmgndYXuwcG_Q5xaPwW-2JplymndIl3WSsz5rUIsnO5JPeIpIkyUNoqWfXv8VI62rCThp_9W5_yIYToYbTX06XK4KbYhECw.svg
WDC|Western_Digital|western digital|westerndigital.com|Western Digital|
SHOP|Shopify|shopify|shopify.com|Shopify|https://i.namu.wiki/i/PXCR6FWeDwVKjxAvGpFpjVTJ1GJyZjp-hdyvGBU7n0NEKdeB0qeKFjytnzZHn2ofxP0phFdENlctqBHrNSOxFv9O7Uk0pnPZzwZO5SdorJ5PEL-DB6BfvARY8oXanvF81CJv_yE5eZH4Uzx0sfjy-Q.svg
APP|AppLovin|applovin|applovin.com|AppLovin|
BKNG|Booking|booking.com|booking.com|Booking Holdings|https://i.namu.wiki/i/QqVkWCviOwpmiyQ2o--XLwkLqJ3yBjBFqUDf299vPicDZU6YwCBxw3O-DYFv8EK7j_WBN0sjRKUMPogb6_VM3w.svg
VRTX|Vertex_Pharmaceuticals|vertex pharmaceuticals|vrtx.com|Vertex Pharmaceuticals|
ISRG|Intuitive_Surgical|intuitive surgical|intuitive.com|Intuitive Surgical|https://i.namu.wiki/i/HpA2Wad3GOXNuE54-xRSDoAidnIcBDaozwmCWCfhiaBNr5HQB8-FlV_nKS7V1Z-AcAjRX8gZUsKHanWIWkHi11bUPnutpHt6sYm3u2NI4cGC0qv4bVb25TWiRR79IvmGBThDY301iTCB4fpvY5B9_w.svg
SBUX|Starbucks|starbucks|starbucks.com|Starbucks|https://i.namu.wiki/i/Wb6wHstizOjjcTdfo5mnPllk6RHeS07Yclm9d2zn2bC4Pi6Ii7P0WH0S9XVHafaGHqZRgbAFKaoD48VB2J-xXr3wrkzlcPux1RffgSQ9PggP6dlQqLa6120yT9eVSPrNcqZ3mxkPjv-3Eq4MNnkQgA.svg
PDD|PDD_Temu|pinduoduo|pddholdings.com|PDD Holdings|https://i.namu.wiki/i/zC9kMoi2WSPxmGXu2emnUJKoPzy6ZFM0ibWpG5lXAa6oFh3gCCCostCcEwbEt71nsfMqNYBuyIrDfRCCa3B9HAafhVyancOWlruitftVwoTgD3kvVHSJYT1l8Zq2EYLUzzfZhy5CGZhFtFe1bvlCFg.svg
FTNT|Fortinet|fortinet|fortinet.com|Fortinet|
ADP|ADP|adp payroll|adp.com|ADP (company)|https://i.namu.wiki/i/jqfhnxBVSGLaDLt2pCelUkNqFwHJCND-K2HijtU95FdXhonYYjV49LalPM6f_B9QEyyjG7euyzz1EyWDPgXDhA.svg
MAR|Marriott|marriott|marriott.com|Marriott International|https://i.namu.wiki/i/6cIF3e5-5xRHuuaXr34s8tKyryVRMggo6DAEjOnJ1aeMqHR7bfiLRDM9YtD-e1mb_uM0HEPnXPXnoQpSdrc3UBS5qm-tSqxeQQRqBUlWQGYkHc3dsW9ZS_JpKliFIQatbVeeRLpUqQC1VqVpETw4rw.svg
MNST|Monster_Beverage|monster energy|monsterenergy.com|Monster Beverage|
ADBE|Adobe|adobe|adobe.com|Adobe Inc.|https://i.namu.wiki/i/57vKBVqGVlWgM5ZDgBbUnBL3L_vXkTyiO4Ev5hAp4Y-RLHmnGoBHTmCCGrT_FsYb_hlYHC3sn8l9thn0ndZzuP0U-LsHR1O3jb_0k8K5LJwrpAONDEslyLxnmubZMEkxcUSVcZer8O_geDXq-u5MYw.svg
CSX|CSX|csx|csx.com|CSX Transportation|https://i.namu.wiki/i/kZpOqqWZaiEf9N-k2566JR65PEGWKwCy6nW5j7DXCAQibwYCwFnoSOxoW8WgpR3PCLsztDs9XzV8u582Ppmb54wDZrCZi3Lmpv6vF_1IKWDjpyEl16yLaLgpAicyp7-pwJVstNBPD0MFlPyW6nhWNA.svg
DDOG|Datadog|datadog|datadoghq.com|Datadog|
MELI|MercadoLibre|mercadolibre|mercadolibre.com|MercadoLibre|
CDNS|Cadence|cadence design|cadence.com|Cadence Design Systems|https://i.namu.wiki/i/xhxYw2RNEgEnXk0saRhO3QOKXO0FQD72ZeyOWLj5T27d-jycawIThN0pu5jU18Gy83iG612XUrVssGzDoKfcEWrT_wOY3R1IQa3TJzfSq82ujDv8PbInrUgqU23GsO0Xejy-6FYRfnwJad0uR92lkA.svg
CEG|Constellation_Energy|constellation energy|constellationenergy.com|Constellation Energy|
ABNB|Airbnb|airbnb|airbnb.com|Airbnb|https://i.namu.wiki/i/RJ_JMpRywHDYoFVaGR7N5p2NQ_75SV_S0-KVoPEFspXz-8onDFMNfz3KKMuyiY6O8Bl3ybBQx_FXf-f-QZffgT2Le8DBPWRH5Lt6vaYtIo5HxP1OEIwNLZlU3V5nZkOB5JTSp4pY1PhIXJjF47rO1g.svg
CMCSA|Comcast|comcast|corporate.comcast.com|Comcast|https://i.namu.wiki/i/2EORd640GzCPzrts9XabXLQbTWtdw4ijR_lArw_jkDNz-WQhiXrDAovuPW8NHthzRuTQJSltvLJYRomanP3RXUMWcBHN_xNdNt4hifGq3M9ZzHo5w10FtwkljiXoJHIGxeaYaHeurgarHVy6rgHfow.svg
CTAS|Cintas|cintas|cintas.com|Cintas|
DASH|DoorDash|doordash|doordash.com|DoorDash|https://i.namu.wiki/i/m-8aPZ_NKhKf1WkQaGrpv7aV-UHXXS2SBpmLYDM4pELot8FABIt8tL3_D7ErJLPTKlA2XSazLhCwto9m9fOv1TwPF9ybziPgwdp4NNwIgViNMEYgyWKnJKGXeOwlo0qKxCkT2JoEBJOB5vdVjs2qAg.webp
INTU|Intuit|intuit|intuit.com|Intuit|https://i.namu.wiki/i/CjYsxy7XgT79XqDmLiu3DKE4GoBrxOwEgaGq9p6TBO16ae54wlN-ANbIeGd6jF_TQt60xGBd_dOUawgl3vdLnrty989-dFa-mJUXpnSHFELf6HpaAwyhYpHjgph2hefQRl1Qx8LppGT7ZJZYcp94BA.svg
MDLZ|Mondelez|mondelez|mondelezinternational.com|Mondelez International|https://i.namu.wiki/i/-NFauNBdwTS6HyT5niU3fE15Susai47Zs3XayuwpOoMwm09HLE1ULDZIyMrcTUlGyeitpEH5GukKUEXAcLjcnTi0zdqUMCt5abOygS8ZElcKc5yfaJuHwehdQN7DbtP_u4CdY1WhxAdxNMcfq_WJdA.svg
ROST|Ross_Stores|ross stores|rossstores.com|Ross Stores|
SNPS|Synopsys|synopsys|synopsys.com|Synopsys|
AEP|American_Electric_Power|american electric power|aep.com|American Electric Power|
ORLY|OReilly_Auto|oreilly auto parts|oreillyauto.com|O'Reilly Auto Parts|
HON|Honeywell|honeywell|honeywell.com|Honeywell|
REGN|Regeneron|regeneron|regeneron.com|Regeneron Pharmaceuticals|
WBD|Warner_Bros_Discovery|warner bros discovery|wbd.com|Warner Bros. Discovery|https://i.namu.wiki/i/kGSMC543gMWAERRxY8_0zcThycIl5gBjF2QZT9_1UEXl_POe3BrhDxCWLPbIRhmT12lA_LKODQE36K-5faON9e1lIS6gBV83Q932F5z-qPozjz0EXbiD8z1EDPfFB3qj_YWy-hRiigEZk2YNCDVbuw.svg
NXPI|NXP|nxp semiconductors|nxp.com|NXP Semiconductors|https://i.namu.wiki/i/mTQbs12BR3ISS1zcKTektxmQ_w4pxCQZxJ_9Vw89Xa-V3RbNUd2gPSvEletGOCTPxwaQ4CTl5M5Sf8LCpuKQbA.svg
PCAR|PACCAR|paccar|paccar.com|Paccar|
MPWR|Monolithic_Power|monolithic power systems|monolithicpower.com|Monolithic Power Systems|
LITE|Lumentum|lumentum|lumentum.com|Lumentum Holdings|
BKR|Baker_Hughes|baker hughes|bakerhughes.com|Baker Hughes|https://i.namu.wiki/i/Lmw5Fv_laIts2DzNVtvHUoEWrNUXxX0OGMysPigPBjVx0NvjXD63vrHtKjpaWJZ4JIaZLIN0TDkaj1Tiuv-Y5GMQb5oBzSt_pBJyXysGdOcZOcnSqmxisrAyAX7_9Gnaeu895_ipvnBP4baA2achyQ.webp
FANG|Diamondback_Energy|diamondback energy|diamondbackenergy.com|Diamondback Energy|
EA|Electronic_Arts|electronic arts|ea.com|Electronic Arts|https://i.namu.wiki/i/LRQslHXK76xgxvE3qIL2kl4Vz_aGC5cSWIM-D28VPvmaCEnyq5czu6m5cdIYwLd_-SYjZOmQBeYSJFTwgg-A_TcGPrF15kZWAuxcXUt2v6bRL_9Bq33cH7lLC93IDWDaF1xwJn2xNzRF6MGjDPNGEw.svg
FAST|Fastenal|fastenal|fastenal.com|Fastenal|
ALAB|Astera_Labs|astera labs|asteralabs.com|Astera Labs|
TER|Teradyne|teradyne|teradyne.com|Teradyne|
PYPL|PayPal|paypal|paypal.com|PayPal|https://i.namu.wiki/i/8k_zwcCF5yM0jgREwj4Fl0MFFAn1H2Coq17jS5_f93Q8-E7ArRs9Yz8vYWw9jKZlUiU-2OjWpDQYFSmbscjDzGZZVVXQaNKHr0-xrhTOTqoCOuiPRzDZmEVdmUplc3d2wLeXuWbQyIt_ryq5bYJJfg.svg
XEL|Xcel_Energy|xcel energy|xcelenergy.com|Xcel Energy|
ODFL|Old_Dominion|old dominion freight|odfl.com|Old Dominion Freight Line|
EXC|Exelon|exelon|exeloncorp.com|Exelon|
CCEP|CocaCola_Europacific|coca-cola europacific|cocacolaep.com|Coca-Cola Europacific Partners|
ADSK|Autodesk|autodesk|autodesk.com|Autodesk|https://i.namu.wiki/i/qSVZk8aIeUq9viPWcovA_OP6QiNoBkgbSx-X6px8tjvcKqGbXNziTLSiEie7Au8-gBFJ8PUtxW3cqw5c_FirVQ.svg
FER|Ferrovial|ferrovial|ferrovial.com|Ferrovial|
NBIS|Nebius|nebius|nebius.com|Nebius Group|
IDXX|IDEXX|idexx|idexx.com|Idexx Laboratories|
MCHP|Microchip|microchip technology|microchip.com|Microchip Technology|https://i.namu.wiki/i/gfoEICCQiEBgvkZ3wBzXNqZXeds9xdxy2D4nGlDmn0yMwVXQvZBFdlfTzSSHnk55Il_B6HmoNoBENhsKyPrs4FuZbh_3kCv6baDWl9zR-jNjHgXXfmianK-9H1Lnv4AUrHfOM-YaGi8XElEeAOWB_g.svg
TTWO|TakeTwo|take-two interactive|take2games.com|Take-Two Interactive|
RKLB|Rocket_Lab|rocket lab|rocketlabusa.com|Rocket Lab|
KDP|Keurig_Dr_Pepper|keurig dr pepper|keurigdrpepper.com|Keurig Dr Pepper|
TRI|Thomson_Reuters|thomson reuters|thomsonreuters.com|Thomson Reuters|
AXON|Axon|axon enterprise|axon.com|Axon Enterprise|
PAYX|Paychex|paychex|paychex.com|Paychex|
CRWV|CoreWeave|coreweave|coreweave.com|CoreWeave|
ROP|Roper|roper technologies|ropertech.com|Roper Technologies|
WDAY|Workday|workday|workday.com|Workday, Inc.|
ALNY|Alnylam|alnylam|alnylam.com|Alnylam Pharmaceuticals|
MSTR|Strategy_MicroStrategy|microstrategy|strategy.com|MicroStrategy|https://i.namu.wiki/i/gOaUBDaNNPUcW-Fo-aCSQ_13h_qMYnqblKk_V3YYOYhyh5lKdUqvznH8FubjDpylvLbSZ2DNQtA8xR5s-OQ2_-f4e5jRutCsi_9ahQRuaNvyMLnpJ45Pp2eK6c1yEeVU1dSSb0ifCbinsfxK4dR3Dw.webp
KHC|Kraft_Heinz|kraft heinz|kraftheinzcompany.com|Kraft Heinz|https://i.namu.wiki/i/t-QrUQfHi22hz3YCVC369vHjCwhlk40OxsnXc4QdNHrh-AplOTl0H8tM1Q1fQ07QMNUWUksalgUeHGcyABToiQ-WK94rgK8MMvFHUX7v7SWegiCDtbhUex3e6AKdlFU0W84Nk5W32a5z4Zt1TiGPmA.svg
DXCM|Dexcom|dexcom|dexcom.com|Dexcom|
GEHC|GE_HealthCare|ge healthcare|gehealthcare.com|GE HealthCare|https://i.namu.wiki/i/WDbeopLjrDH7A5V4jXDMkCPpkEFH__XsGWNTKO6vM3ELq99qXgpYI5JQ8rLiq_K5_f3ws4d6SBocyG3LVNPDhA.svg
CPRT|Copart|copart|copart.com|Copart|
"@

# ---------- 이전 수집물 전체 삭제 (위키류 포함 클린 스타트) ----------
if (Test-Path $Root) { Get-ChildItem $Root -Recurse -File | Remove-Item -Force; Get-ChildItem $Root -Directory | Remove-Item -Recurse -Force }
LogLine "이전 수집물 전체 삭제"

$total = 0
foreach ($line in ($DATA -split "`n")) {
  $line = $line.Trim()
  if (-not $line) { continue }
  $f = $line -split '\|'
  $tick = $f[0]; $brand = $f[1]; $tok = $f[2]; $dom = $f[3]
  $tok0 = ($tok -split ' ')[0].ToLower()
  $dir = Join-Path $Root ("{0}_{1}" -f $tick, $brand)
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Write-Host "=== [$tick] $brand ===" -ForegroundColor Cyan
  $hashes = @{}
  $dhashes = New-Object System.Collections.ArrayList
  $count = 0

  # 1) Clearbit 공식 로고
  $p = Join-Path $dir ("{0}_{1}_clearbit.png" -f $tick, $brand)
  if (Try-Add ("https://logo.clearbit.com/{0}?size=512" -f $dom) $p $hashes $dhashes) { $count++; Write-Host "  + clearbit" -ForegroundColor Green }

  # 2) DuckDuckGo 이미지 (투명배경 쿼리 우선, 일반 쿼리는 부족할 때만 보충)
  foreach ($q in @("$tok logo png transparent", "$tok logo transparent background", "$tok logo")) {
    if ($count -ge $K) { break }
    $h1 = Get-Html ("https://duckduckgo.com/?q=" + [uri]::EscapeDataString($q) + "&iax=images&ia=images")
    $vqd = ''
    if ($h1) { $m = [regex]::Match($h1, "vqd=['\`"]?([\d-]+)"); if ($m.Success) { $vqd = $m.Groups[1].Value } }
    if (-not $vqd) { LogLine "[$tick] DDG vqd 실패 ($q)"; continue }
    $jd = $null
    try { $jd = Invoke-RestMethod -Uri ("https://duckduckgo.com/i.js?l=us-en&o=json&q=" + [uri]::EscapeDataString($q) + "&vqd=" + $vqd + "&p=1") -Headers @{ 'User-Agent' = $UA; 'Referer' = 'https://duckduckgo.com/' } -TimeoutSec 25 } catch { LogLine "[$tick] DDG 요청 실패" }
    if (-not ($jd -and $jd.results)) { continue }
    $di = ($count + 1)
    foreach ($r in $jd.results) {
      if ($count -ge $K) { break }
      $title = ('' + $r.title).ToLower()
      if (-not $title.Contains($tok0)) { continue }
      if ($title -notmatch 'logo|로고|svg|png|vector|icon|brand|emblem|wordmark|symbol|transparent') { continue }
      if ($title -match $BAD) { continue }
      if (('' + $r.image).ToLower() -match 'history|evolution|collection|banner|wallpaper') { continue }
      $u = $r.image
      if ($u -notmatch '\.(svg|png|jpg|jpeg|webp)([?#]|$)') { continue }
      $p = Join-Path $dir ("{0}_{1}_ddg_{2:d2}.{3}" -f $tick, $brand, $di, (Get-Ext $u))
      if (Try-Add $u $p $hashes $dhashes) { $count++; $di++; Write-Host "  + ddg" -ForegroundColor DarkGray }
    }
  }

  # 3) Bing 이미지 (투명 필터, 2페이지)
  foreach ($first in @(1, 36)) {
    if ($count -ge $K) { break }
    $html2 = Get-Html ("https://www.bing.com/images/search?q=" + [uri]::EscapeDataString("$tok logo") + "&qft=+filterui:photo-transparent&first=$first")
    if (-not $html2) { continue }
    $html2 = $html2 -replace '&quot;','"'
    $items = [regex]::Matches($html2, '\{"murl":"(https?:[^"]+?)"[^\}]*?"t":"([^"]*?)"')
    $bi = ($count + 1)
    foreach ($m in $items) {
      if ($count -ge $K) { break }
      $u = $m.Groups[1].Value -replace '\\/','/'
      $title = $m.Groups[2].Value.ToLower()
      if (-not $title.Contains($tok0)) { continue }
      if ($title -match $BAD) { continue }
      if ($u.ToLower() -match 'history|evolution|collection|banner|wallpaper') { continue }
      if ($u -notmatch '\.(svg|png|jpg|jpeg|webp)([?#]|$)') { continue }
      $p = Join-Path $dir ("{0}_{1}_bing_{2:d2}.{3}" -f $tick, $brand, $bi, (Get-Ext $u))
      if (Try-Add $u $p $hashes $dhashes) { $count++; $bi++; Write-Host "  + bing" -ForegroundColor DarkGray }
    }
  }
  LogLine ("[$tick] $brand : $count 장")
  Write-Host ("  => $count 장") -ForegroundColor Yellow
  $total += $count
}
LogLine ("v5 완료: $(Get-Date)  총 $total 장")
Write-Host "`n완료! 총 $total 장. 로그: collect_log.txt" -ForegroundColor Yellow
Read-Host "엔터를 누르면 종료"
