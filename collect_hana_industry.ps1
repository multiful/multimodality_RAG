<#
.SYNOPSIS
    네이버 금융 리서치 - 산업분석 카테고리에서 특정 증권사(기본: 하나증권) 리포트만 N개 수집

.DESCRIPTION
    collect_naver_research.ps1 을 기반으로, 산업분석(industry_list.naver) 목록을
    페이지 단위로 훑으면서 지정한 증권사가 발간한 PDF만 필터링하여 다운로드한다.
    하나증권 리포트는 목록에 드물게 섞여 있으므로 MaxPages 를 넉넉히 준다.

    - 목록 페이지는 EUC-KR(cp949) → raw byte로 받아 직접 디코딩
    - 이미 받은 파일(pdf_url 기준)은 건너뛰므로 재실행(resume) 가능
    - 결과 PDF: {OutRoot}\industry\ , 메타데이터: {OutRoot}\metadata.csv 에 append

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\collect_hana_industry.ps1
    powershell -ExecutionPolicy Bypass -File .\collect_hana_industry.ps1 -Count 20 -Broker "하나"
#>

[CmdletBinding()]
param(
    # 수집할 하나증권 산업분석 리포트 개수
    [int]$Count = 20,

    # 증권사 필터 (broker 텍스트에 이 문자열이 포함되면 매칭). "하나" 로 하나증권/하나금융투자 모두 커버
    [string]$Broker = "하나",

    # PDF 저장 루트 (기본: 스크립트 위치\data\raw)
    [string]$OutRoot = "",

    # 페이지 요청 간 대기(초)
    [double]$DelaySec = 0.7,

    # 안전장치: 최대 탐색 페이지 수 (하나증권이 드물면 늘려야 함)
    [int]$MaxPages = 40
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$SleepMs = [int]($DelaySec * 1000)

if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $OutRoot = Join-Path $scriptDir "data\raw"
}

$Code    = "industry"
$CatName = "산업분석"
$Slug    = "industry_list.naver"

$BaseUrl = "https://finance.naver.com/research/"
$Headers = @{
    "User-Agent"      = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    "Referer"         = $BaseUrl
    "Accept-Language" = "ko-KR,ko;q=0.9"
}
$EucKr = [System.Text.Encoding]::GetEncoding(949)

# ---------------------------------------------------------------------------
function Get-PageHtml {
    param([string]$Url)
    $resp = Invoke-WebRequest -Uri $Url -Headers $Headers -UseBasicParsing -TimeoutSec 30
    if ($resp.Content -is [byte[]]) { return $EucKr.GetString($resp.Content) }
    if ($resp.RawContentStream)     { return $EucKr.GetString($resp.RawContentStream.ToArray()) }
    return $resp.Content
}

function Remove-HtmlTag {
    param([string]$s)
    $s = $s -replace '<[^>]+>', ''
    $s = [System.Net.WebUtility]::HtmlDecode($s)
    return $s.Trim()
}

function ConvertTo-SafeFileName {
    param([string]$s, [int]$MaxLen = 80)
    $s = $s -replace '[\\/:*?"<>|]', '_'
    $s = $s -replace '\s+', ' '
    $s = $s.Trim()
    if ($s.Length -gt $MaxLen) { $s = $s.Substring(0, $MaxLen).Trim() }
    if ([string]::IsNullOrWhiteSpace($s)) { $s = "untitled" }
    return $s
}

function Get-ReportsFromPage {
    param([string]$Html)
    $reports = @()
    $rows = $Html -split '(?i)<tr'
    foreach ($row in $rows) {
        $pdfMatch = [regex]::Match($row, 'href="(?<u>https?://(?:ssl\.)?(?:stock\.)?pstatic\.net/[^"]+?\.pdf)"', 'IgnoreCase')
        if (-not $pdfMatch.Success) {
            $pdfMatch = [regex]::Match($row, 'href="(?<u>[^"]+?\.pdf)"', 'IgnoreCase')
            if (-not $pdfMatch.Success) { continue }
        }
        $pdfUrl = $pdfMatch.Groups['u'].Value
        if ($pdfUrl -notmatch '^https?://') { $pdfUrl = "https://finance.naver.com" + $pdfUrl }

        $titleMatch = [regex]::Match($row, '<a[^>]+href="[^"]*_read\.n(?:aver|hn)\?[^"]*nid=(?<nid>\d+)[^"]*"[^>]*>(?<t>.*?)</a>', 'IgnoreCase,Singleline')
        $title = if ($titleMatch.Success) { Remove-HtmlTag $titleMatch.Groups['t'].Value } else { "" }
        $nid   = if ($titleMatch.Success) { $titleMatch.Groups['nid'].Value } else { "" }

        $itemMatch = [regex]::Match($row, '<a[^>]+href="[^"]*(?:/item/main\.n(?:aver|hn)|upjong)[^"]*"[^>]*>(?<t>.*?)</a>', 'IgnoreCase,Singleline')
        if ($itemMatch.Success) {
            $itemName = Remove-HtmlTag $itemMatch.Groups['t'].Value
            if ($itemName -and $title -notlike "$itemName*") { $title = "$itemName - $title" }
        }

        # 증권사 = 제목(_read 앵커가 있는) 칸 바로 다음 칸.
        # 산업분석 목록의 셀 순서: [분류(섹터)] [제목] [증권사] [첨부] [날짜] [조회수]
        # 분류 칸은 링크가 없는 평문이라, 단순히 '첫 평문 td'를 집으면 섹터를 오인함.
        $broker = ""
        $tdMatches = @([regex]::Matches($row, '<td[^>]*>(?<c>.*?)</td>', 'IgnoreCase,Singleline'))
        $titleIdx = -1
        for ($i = 0; $i -lt $tdMatches.Count; $i++) {
            if ($tdMatches[$i].Groups['c'].Value -match '_read\.n(?:aver|hn)') { $titleIdx = $i; break }
        }
        if ($titleIdx -ge 0 -and ($titleIdx + 1) -lt $tdMatches.Count) {
            $broker = Remove-HtmlTag $tdMatches[$titleIdx + 1].Groups['c'].Value
        }
        # 폴백: 위치 탐지 실패 시 기존 휴리스틱(링크·숫자·제목 아닌 첫 칸)
        if ([string]::IsNullOrWhiteSpace($broker)) {
            foreach ($td in $tdMatches) {
                $txt = Remove-HtmlTag $td.Groups['c'].Value
                if ($txt -and $txt -notmatch '^\d' -and $txt -ne $title -and $td.Groups['c'].Value -notmatch '<a\s') {
                    $broker = $txt; break
                }
            }
        }

        $dateMatch = [regex]::Match($row, '\b(\d{2}\.\d{2}\.\d{2})\b')
        $date = if ($dateMatch.Success) { $dateMatch.Groups[1].Value } else { "" }

        $reports += [pscustomobject]@{ PdfUrl=$pdfUrl; Title=$title; Broker=$broker; Date=$date; Nid=$nid }
    }
    return $reports
}

# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null
$catDir  = Join-Path $OutRoot $Code
New-Item -ItemType Directory -Path $catDir -Force | Out-Null

$metaPath = Join-Path $OutRoot "metadata.csv"
$allMeta  = @()
if (Test-Path $metaPath) {
    try { $allMeta = @(Import-Csv $metaPath -Encoding UTF8) } catch { $allMeta = @() }
}

Write-Host ""
Write-Host ("=== [{0}] {1} : '{2}' 발간분 {3}개 목표 ===" -f $Code, $CatName, $Broker, $Count) -ForegroundColor Cyan

$collected = 0
$scanned   = 0
$seenUrls  = New-Object 'System.Collections.Generic.HashSet[string]'
$page      = 1

while ($collected -lt $Count -and $page -le $MaxPages) {
    $listUrl = "{0}{1}?&page={2}" -f $BaseUrl, $Slug, $page
    Write-Host ("  page {0} : {1}" -f $page, $listUrl) -ForegroundColor DarkGray

    try { $html = Get-PageHtml -Url $listUrl }
    catch { Write-Warning ("  목록 페이지 요청 실패: {0}" -f $_.Exception.Message); break }

    $reports = @(Get-ReportsFromPage -Html $html)
    if ($reports.Count -eq 0) {
        $dump = Join-Path $OutRoot ("debug_{0}_p{1}.html" -f $Code, $page)
        [IO.File]::WriteAllText($dump, $html, [Text.Encoding]::UTF8)
        Write-Warning ("  PDF 링크 없음 - 구조 변경 가능. 덤프: {0}" -f $dump)
        break
    }

    foreach ($r in $reports) {
        if ($collected -ge $Count) { break }
        if (-not $seenUrls.Add($r.PdfUrl)) { continue }
        $scanned++

        # === 증권사 필터 ===
        if ($r.Broker -notlike "*$Broker*") { continue }

        $idx      = $collected + 1
        $datePart = if ($r.Date) { $r.Date -replace '\.', '' } else { "nodate" }
        $safeTitle = ConvertTo-SafeFileName $r.Title
        $fileName = "{0}_hana_{1:d2}_{2}_{3}.pdf" -f $Code, $idx, $datePart, $safeTitle
        $outFile  = Join-Path $catDir $fileName

        $already = $allMeta | Where-Object { $_.pdf_url -eq $r.PdfUrl }
        if ($already -and (Test-Path (Join-Path $OutRoot $already[0].local_path))) {
            # 기존 행 broker 라벨 교정 (구 수집기 버그로 섹터명이 들어간 경우 복구)
            if ($already[0].broker -ne $r.Broker -and $r.Broker) {
                Write-Host ("    [skip/label수정] {0}  '{1}'→'{2}'" -f $r.Title, $already[0].broker, $r.Broker) -ForegroundColor DarkYellow
                $already[0].broker = $r.Broker
            } else {
                Write-Host ("    [skip] 이미 수집됨: {0}" -f $r.Title) -ForegroundColor DarkGray
            }
            $collected++
            continue
        }

        try {
            Invoke-WebRequest -Uri $r.PdfUrl -Headers $Headers -OutFile $outFile -UseBasicParsing -TimeoutSec 60
            $bytes = [IO.File]::ReadAllBytes($outFile)
            if ($bytes.Length -lt 4 -or -not ($bytes[0] -eq 0x25 -and $bytes[1] -eq 0x50 -and $bytes[2] -eq 0x44 -and $bytes[3] -eq 0x46)) {
                Write-Warning ("    PDF 아님: {0}" -f $r.PdfUrl); Remove-Item $outFile -Force; continue
            }
            $sizeKB = [math]::Round((Get-Item $outFile).Length / 1KB, 1)
            Write-Host ("    [{0,2}/{1}] {2}  ({3} KB)  <{4}>" -f $idx, $Count, $safeTitle, $sizeKB, $r.Broker) -ForegroundColor Green

            $allMeta += [pscustomobject]@{
                category=$Code; category_name=$CatName; nid=$r.Nid; title=$r.Title
                broker=$r.Broker; report_date=$r.Date; pdf_url=$r.PdfUrl
                local_path=(Join-Path $Code $fileName); size_kb=$sizeKB
                downloaded_at=(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
            }
            $collected++
        } catch {
            Write-Warning ("    다운로드 실패: {0} → {1}" -f $r.PdfUrl, $_.Exception.Message)
        }
        Start-Sleep -Milliseconds $SleepMs
    }

    $page++
    Start-Sleep -Milliseconds $SleepMs
}

$csvEnc = if ($PSVersionTable.PSVersion.Major -ge 6) { 'utf8BOM' } else { 'UTF8' }
$allMeta | Export-Csv -Path $metaPath -NoTypeInformation -Encoding $csvEnc

Write-Host ""
Write-Host ("→ 하나증권 산업분석 {0}/{1}개 수집 (스캔 {2}건, {3}페이지)" -f $collected, $Count, $scanned, ($page-1)) -ForegroundColor Yellow
Write-Host ("메타데이터: {0} (총 {1}건)" -f $metaPath, $allMeta.Count) -ForegroundColor Cyan
Write-Host ("PDF 위치  : {0}" -f $catDir) -ForegroundColor Cyan
if ($collected -lt $Count) {
    Write-Host ("주의: 목표 미달. -MaxPages 를 늘리거나 -Broker 조건을 확인하세요." -f $collected) -ForegroundColor Red
}
