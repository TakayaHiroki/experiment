# ベクトル化を順番に実行するバッチスクリプト（PowerShell）
#
# usage（research フォルダの中で実行する）:
#    powershell -ExecutionPolicy Bypass -File .\tools\RQ1\run_embeddings.ps1
#    powershell -ExecutionPolicy Bypass -File .\tools\RQ1\run_embeddings.ps1 -Models qwen3-0.6b
#    powershell -ExecutionPolicy Bypass -File .\tools\RQ1\run_embeddings.ps1 -Models qwen3-8b,gte-qwen2-7b-instruct
#    powershell -ExecutionPolicy Bypass -File .\tools\RQ1\run_embeddings.ps1 -DryRun

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Corpus   = "bewt",
    [string]$CorpusDir = "corpus\bewt\json",
    [string]$OutRoot  = "embeddings",
    [string]$LogFile  = "logs\embed_runlog.csv",
    [string[]]$Models = @(),
    [string[]]$Variants = @("full", "title", "steps", "expect"),
    [string[]]$Apps = @(),
    [int]$Threads = 0,
    [ValidateSet("cpu", "cuda")][string]$Device = "cpu",
    [switch]$AllCores,
    [double]$MaxBusyPercent = 10,
    [int]$IdleWaitSec = 300,
    [int]$MinFreeMemMB = 1024,
    [switch]$IncludeAll,
    [switch]$DryRun,
    [switch]$Force
)

$ModelScript = [ordered]@{
    "tfidf"                   = "BASELINE:tfidf"
    "lsa"                     = "BASELINE:lsa"
    "lsa-full"                = "BASELINE:lsa-full"
    "sbert-all-mpnet-base-v2" = "tools\RQ1\embedding_models\sbertallmpnetbasev2.py"
    "bge-base-en-v1.5"        = "tools\RQ1\embedding_models\bgebaseenv1.5.py"
    "e5-base-v2"              = "tools\RQ1\embedding_models\e5basev2.py"
    "qwen3-0.6b"                  = "tools\RQ1\embedding_models\qwen3embedding06b.py"
    "qwen3-0.6b+sts"              = "tools\RQ1\embedding_models\qwen3embedding06b.py"
    "qwen3-0.6b@fp32"             = "tools\RQ1\embedding_models\qwen3embedding06b.py"
    "qwen3-0.6b+sts@fp32"         = "tools\RQ1\embedding_models\qwen3embedding06b.py"
    "gte-qwen2-1.5b-instruct"     = "tools\RQ1\embedding_models\gteqwen21.5binstruct.py"
    "gte-qwen2-1.5b-instruct+sts" = "tools\RQ1\embedding_models\gteqwen21.5binstruct.py"
    "stella-en-1.5b-v5"           = "tools\RQ1\embedding_models\stellaen1.5bv5.py"
    "stella-en-1.5b-v5+sts"       = "tools\RQ1\embedding_models\stellaen1.5bv5.py"
    "qwen3-4b"                    = "tools\RQ1\embedding_models\qwen3embedding4b.py"
    "qwen3-4b+sts"                = "tools\RQ1\embedding_models\qwen3embedding4b.py"
    "qwen3-4b@fp32"               = "tools\RQ1\embedding_models\qwen3embedding4b.py"
    "qwen3-4b+sts@fp32"           = "tools\RQ1\embedding_models\qwen3embedding4b.py"
    "gte-qwen2-7b-instruct"       = "tools\RQ1\embedding_models\gteqwen27binstruct.py"
    "gte-qwen2-7b-instruct+sts"   = "tools\RQ1\embedding_models\gteqwen27binstruct.py"
    "qwen3-8b"                    = "tools\RQ1\embedding_models\qwen3embedding8b.py"
    "qwen3-8b+sts"                = "tools\RQ1\embedding_models\qwen3embedding8b.py"
    "qwen3-8b@fp32"               = "tools\RQ1\embedding_models\qwen3embedding8b.py"
    "qwen3-8b+sts@fp32"           = "tools\RQ1\embedding_models\qwen3embedding8b.py"
}

$PromptModels = @("qwen3-0.6b", "gte-qwen2-1.5b-instruct", "stella-en-1.5b-v5", "qwen3-4b",
                  "gte-qwen2-7b-instruct", "qwen3-8b")
function Get-ExtraArgs($model) {
    $base = ($model -replace "@fp32$", "") -replace "\+sts$", ""
    $extra = ""
    if ($PromptModels -contains $base) {
        if ($model -like "*+sts*") { $extra += " --prompt sts" } else { $extra += " --prompt none" }
    }
    if ($base -like "qwen3-*") {
        if ($model -like "*@fp32") { $extra += " --dtype float32" } else { $extra += " --dtype bfloat16" }
    }
    return $extra
}

$DefaultOrder = @(
    "tfidf", "lsa", "lsa-full",
    "sbert-all-mpnet-base-v2", "bge-base-en-v1.5", "e5-base-v2",
    "qwen3-0.6b", "qwen3-0.6b+sts", "qwen3-0.6b@fp32", "qwen3-0.6b+sts@fp32",
    "gte-qwen2-1.5b-instruct", "gte-qwen2-1.5b-instruct+sts",
    "stella-en-1.5b-v5", "stella-en-1.5b-v5+sts",
    "qwen3-4b", "qwen3-4b+sts",
    "gte-qwen2-7b-instruct", "gte-qwen2-7b-instruct+sts",
    "qwen3-8b", "qwen3-8b+sts"
)

$ConflictPattern = "embedding_models|baseline_embed"

$LogHeader = "timestamp,corpus,model,app,variant,device,n,truncated,batch,threads,interop,bound_threads,dtype,affinity,params," +
             "t_warm_s,warm_bytes,t_load_s,t_first_s,t_encode_s,total_s," +
             "c_warm_s,c_load_s,c_first_s,c_encode_s,cpu_total_s," +
             "cpu_busy_before_pct,other_cpu_during_pct,peak_ws_mb,peak_rss_mb,peak_gpu_mb,mem_free_min_mb," +
             "env_file,status"

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public static class CpuSets {
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetSystemCpuSetInformation(IntPtr info, uint length, out uint returned, IntPtr process, uint flags);
    public static int[][] Get() {
        uint len;
        GetSystemCpuSetInformation(IntPtr.Zero, 0, out len, IntPtr.Zero, 0);
        IntPtr buf = Marshal.AllocHGlobal((int)len);
        try {
            if (!GetSystemCpuSetInformation(buf, len, out len, IntPtr.Zero, 0))
                throw new System.ComponentModel.Win32Exception();
            var list = new List<int[]>();
            int off = 0;
            while (off < len) {
                int size = Marshal.ReadInt32(buf, off);
                // 論理番号，効率クラス，物理コア番号．効率クラスが大きいほど性能コア
                list.Add(new int[] { Marshal.ReadByte(buf, off + 14), Marshal.ReadByte(buf, off + 18), Marshal.ReadByte(buf, off + 15) });
                off += size;
            }
            return list.ToArray();
        } finally { Marshal.FreeHGlobal(buf); }
    }
}
'@

function Get-CpuRaw {
    Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter "Name='_Total'"
}

function Get-BusyFraction($a, $b) {
    $dt = [double]($b.Timestamp_Sys100NS - $a.Timestamp_Sys100NS)
    if ($dt -le 0) { return 0.0 }
    # このカウンタは遊休時間を数えるので 1.0 から引く．反転は誤りではない
    return [math]::Max(0.0, 1.0 - ([double]($b.PercentProcessorTime - $a.PercentProcessorTime) / $dt))
}

function Get-BusyPercent {
    $a = Get-CpuRaw
    Start-Sleep -Seconds 2
    $b = Get-CpuRaw
    return [math]::Round(100 * (Get-BusyFraction $a $b), 1)
}

function Get-ConflictingProcess {
    Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
        Where-Object { $_.CommandLine -match $ConflictPattern }
}

function Get-FreeMemMB {
    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
}

function Get-ModelId($script) {
    $m = Select-String -Path $script -Pattern '^MODEL = "(.+)"' | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value }
    return $null
}

if (-not (Test-Path "tools\RQ1")) {
    Write-Host "research フォルダの中で実行すること" -ForegroundColor Red
    exit 1
}

$Models   = @($Models   | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Variants = @($Variants | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Apps     = @($Apps     | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($Models.Count -eq 0) { $Models = $DefaultOrder }

$cpus = [CpuSets]::Get()
$logical = $cpus.Count
$topClass = ($cpus | ForEach-Object { $_[1] } | Measure-Object -Maximum).Maximum
if ($AllCores) {
    $workCpus = @($cpus)
} else {
    # Pコアの各物理コアから論理番号が最小の1つだけを使う
    $workCpus = @($cpus | Where-Object { $_[1] -eq $topClass } | Group-Object { $_[2] } |
                  ForEach-Object { $_.Group | Sort-Object { $_[0] } | Select-Object -First 1 } |
                  Sort-Object { $_[0] })
}
$workIds = @($workCpus | ForEach-Object { $_[0] })
$sideCpus = @($cpus | Where-Object { $workIds -notcontains $_[0] })
$workMask = [int64]0
foreach ($c in $workCpus) { $workMask = $workMask -bor ([int64]1 -shl $c[0]) }
$sideMask = [int64]0
foreach ($c in $sideCpus) { $sideMask = $sideMask -bor ([int64]1 -shl $c[0]) }
if ($Threads -le 0) { $Threads = $workCpus.Count }
if ((-not $AllCores) -and ($Threads -gt $workCpus.Count)) {
    Write-Host "スレッド数 $Threads が固定先の論理プロセッサ数 $($workCpus.Count) を超えている" -ForegroundColor Red
    exit 1
}
$affinityLabel = if ($AllCores) { "all" } else { "0x{0:X}" -f $workMask }

Write-Host "計測に使う論理プロセッサ: $($workIds -join ',') / $logical (mask $affinityLabel) / スレッド数: $Threads"
Write-Host "計算に使う装置: $Device"

if ($DryRun) {
    foreach ($model in $Models) {
        if (-not $ModelScript.Contains($model)) { Write-Host "[警告] 未知のモデル名: $model" -ForegroundColor Yellow }
        else { Write-Host "  $model -> $($ModelScript[$model])$(Get-ExtraArgs $model)" }
    }
    Write-Host "(DryRun のため実行しない)"
    exit 0
}

# 二重起動を拒否する
$mutex = New-Object System.Threading.Mutex($false, "Global\experiment_run_embeddings")
$owned = $false
try { $owned = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
if (-not $owned) {
    Write-Host "run_embeddings.ps1 が別に実行中．終了後に再実行" -ForegroundColor Red
    exit 1
}

try {
    $conf = @(Get-ConflictingProcess)
    if ($conf.Count -gt 0) {
        Write-Host "ベクトル化の Python プロセスが別に動いている．終了後に再実行" -ForegroundColor Red
        $conf | ForEach-Object { Write-Host "    PID $($_.ProcessId): $($_.CommandLine)" }
        exit 1
    }

    $self = [System.Diagnostics.Process]::GetCurrentProcess()
    if ($sideMask -ne 0) { $self.ProcessorAffinity = [IntPtr]$sideMask }

    $env:OMP_NUM_THREADS = "$Threads"
    $env:MKL_NUM_THREADS = "$Threads"
    $env:KMP_AFFINITY = if ($AllCores) { "verbose,none" } else { "verbose,granularity=fine,compact,1,0" }

    $logDir = Split-Path $LogFile
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    if (-not (Test-Path $LogFile)) {
        $LogHeader | Out-File -FilePath $LogFile -Encoding utf8
    } elseif ((Get-Content $LogFile -TotalCount 1) -ne $LogHeader) {
        Write-Host "$LogFile の列が現在の形式と違う．-LogFile で別のファイルを指定" -ForegroundColor Red
        exit 1
    }

    $modelIds = @($Models | Where-Object { $ModelScript.Contains($_) -and ($ModelScript[$_] -notlike "BASELINE:*") } |
                  ForEach-Object { Get-ModelId $ModelScript[$_] } | Where-Object { $_ } | Sort-Object -Unique)
    $envName = "embed_env_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    $pyInfo = (& python tools\RQ1\embed_env.py --models @modelIds 2>$null) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "計測環境の取得に失敗（モデルが取得済みか確認）" -ForegroundColor Red
        exit 1
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $gitHead = (& git rev-parse HEAD 2>$null)
    $gitDirty = [bool](& git status --porcelain -- tools 2>$null)
    $envInfo = [ordered]@{
        created        = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        cpu            = ((Get-CimInstance Win32_Processor | Select-Object -First 1).Name).Trim()
        logical_procs  = $logical
        cpu_sets       = @($cpus | ForEach-Object { [ordered]@{ logical = $_[0]; efficiency_class = $_[1]; core = $_[2] } })
        work_procs     = $workIds
        affinity_mask  = $affinityLabel
        threads        = $Threads
        priority       = "AboveNormal"
        ram_total_mb   = [math]::Round($cs.TotalPhysicalMemory / 1MB)
        ram_free_mb    = Get-FreeMemMB
        os             = "$($os.Caption) $($os.Version) build $($os.BuildNumber)"
        power_scheme   = ((& powercfg /getactivescheme) -join " ").Trim()
        git_head       = $gitHead
        git_tools_dirty = $gitDirty
        args           = [ordered]@{ Models = $Models; Variants = $Variants; Apps = $Apps; Device = $Device;
                                     MaxBusyPercent = $MaxBusyPercent; MinFreeMemMB = $MinFreeMemMB;
                                     AllCores = [bool]$AllCores }
        python         = ($pyInfo | ConvertFrom-Json)
    }
    $envInfo | ConvertTo-Json -Depth 6 | Out-File -FilePath (Join-Path $logDir $envName) -Encoding utf8
    Write-Host "計測環境: $(Join-Path $logDir $envName)"
    Write-Host "電源プラン: $($envInfo.power_scheme)"
    $warmed = @{}

    if ($Apps.Count -eq 0) {
        $Apps = Get-ChildItem -Path $CorpusDir -Filter "*_full.json" |
                ForEach-Object { $_.BaseName -replace "_full$", "" } |
                # ALL_* はアプリ横断なので既定では作らない
                Where-Object { $IncludeAll -or ($_ -ne "ALL") } | Sort-Object
    }

    $total = 0; $skipped = 0; $failed = 0; $noisy = 0
    $tWarm = ""; $cWarm = ""; $warmBytes = ""

    foreach ($model in $Models) {
        if (-not $ModelScript.Contains($model)) {
            Write-Host "[警告] 未知のモデル名: $model" -ForegroundColor Yellow
            continue
        }
        $script = $ModelScript[$model]
        $isBaseline = $script -like "BASELINE:*"

        foreach ($app in $Apps) {
            foreach ($variant in $Variants) {

                $inFile  = Join-Path $CorpusDir "$($app)_$($variant).json"
                $outDir  = Join-Path $OutRoot  "$Corpus\$model"
                $outFile = Join-Path $outDir   "$($app)_$($variant).json"

                if (-not (Test-Path $inFile)) { continue }
                # 既存の出力は飛ばす．途中で止めても再開できる
                if ((Test-Path $outFile) -and (-not $Force)) { $skipped++; continue }

                $conf = @(Get-ConflictingProcess)
                if ($conf.Count -gt 0) {
                    Write-Host "ベクトル化の Python プロセスが別に起動されたため中止" -ForegroundColor Red
                    $conf | ForEach-Object { Write-Host "    PID $($_.ProcessId): $($_.CommandLine)" }
                    exit 1
                }

                $busy = Get-BusyPercent
                $waited = 0
                while (($busy -gt $MaxBusyPercent) -and ($waited -lt $IdleWaitSec)) {
                    Write-Host "  CPU使用率 $busy% のため待機中..." -ForegroundColor Yellow
                    Start-Sleep -Seconds 10
                    $waited += 12
                    $busy = Get-BusyPercent
                }

                if ((-not $isBaseline) -and (-not $warmed.ContainsKey($script))) {
                    Write-Host "[$model] モデルのファイルを先読み中..." -NoNewline
                    $warmErr = [System.IO.Path]::GetTempFileName()
                    $swWarm = [System.Diagnostics.Stopwatch]::StartNew()
                    & python tools\RQ1\embed_env.py --warm --models (Get-ModelId $script) 2>$warmErr
                    $swWarm.Stop()
                    $wl = @(Get-Content $warmErr -ErrorAction SilentlyContinue) -join " "
                    Remove-Item $warmErr -ErrorAction SilentlyContinue
                    $tWarm = [math]::Round($swWarm.Elapsed.TotalSeconds, 2)
                    $cWarm = ""; $warmBytes = ""
                    if ($wl -match "t_warm=([\d.]+)")    { $tWarm     = $Matches[1] }
                    if ($wl -match "c_warm=([\d.]+)")    { $cWarm     = $Matches[1] }
                    if ($wl -match "warm_bytes=(\d+)")   { $warmBytes = $Matches[1] }
                    $mb = if ($warmBytes -ne "") { "$([math]::Round([double]$warmBytes / 1MB)) MB" } else { "?" }
                    Write-Host " 完了 ($tWarm 秒 / $mb)"
                    $warmed[$script] = $true
                }

                New-Item -ItemType Directory -Force -Path $outDir | Out-Null
                Write-Host "[$model] $app / $variant ..." -NoNewline

                if ($isBaseline) {
                    $method = $script -replace "BASELINE:", ""
                    $argLine = "tools\RQ1\baseline_embed.py -i `"$inFile`" -o `"$outFile`" -m $method"
                } else {
                    $argLine = "`"$script`" -i `"$inFile`" -o `"$outFile`" --device $Device$(Get-ExtraArgs $model)"
                }
                $stdoutFile = [System.IO.Path]::GetTempFileName()
                $stderrFile = [System.IO.Path]::GetTempFileName()

                $selfCpu0 = $self.TotalProcessorTime.TotalSeconds
                $raw0 = Get-CpuRaw
                # 子プロセスを固定先のコアで起動するため親の割り当てを一時的に切り替える
                if (-not $AllCores) { $self.ProcessorAffinity = [IntPtr]$workMask }
                $sw = [System.Diagnostics.Stopwatch]::StartNew()
                $p = Start-Process -FilePath "python" -ArgumentList $argLine -NoNewWindow -PassThru `
                        -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
                $null = $p.Handle
                if ($sideMask -ne 0) { $self.ProcessorAffinity = [IntPtr]$sideMask }
                try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal } catch {}

                $peak = [int64]0
                $memMin = [int64]::MaxValue
                while (-not $p.HasExited) {
                    try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
                    $free = Get-FreeMemMB
                    if ($free -lt $memMin) { $memMin = $free }
                    Start-Sleep -Milliseconds 1000
                }
                $p.WaitForExit()
                $sw.Stop()
                $raw1 = Get-CpuRaw
                $sec = [math]::Round($sw.Elapsed.TotalSeconds, 2)

                $allCpuSec = (Get-BusyFraction $raw0 $raw1) * $sw.Elapsed.TotalSeconds * $logical
                $childCpuSec = $p.TotalProcessorTime.TotalSeconds
                $selfCpuSec = $self.TotalProcessorTime.TotalSeconds - $selfCpu0
                $otherPct = [math]::Round(100 * [math]::Max(0, $allCpuSec - $childCpuSec - $selfCpuSec) /
                                          ($sw.Elapsed.TotalSeconds * $logical), 1)
                if ($memMin -eq [int64]::MaxValue) { $memMin = Get-FreeMemMB }
                $peakMB = [math]::Round($peak / 1MB)

                $out = @(Get-Content $stdoutFile -ErrorAction SilentlyContinue)
                $err = @(Get-Content $stderrFile -ErrorAction SilentlyContinue)
                Remove-Item $stdoutFile, $stderrFile -ErrorAction SilentlyContinue

                $bound = @($err | Where-Object { $_ -match "KMP_AFFINITY: pid \d+ tid \d+ thread \d+ bound to OS proc set" })
                $boundProcs = @($bound | ForEach-Object { ($_ -split "OS proc set ")[-1].Trim() })
                $err = @($err | Where-Object { $_ -notmatch "^OMP: Info" })

                $tLoad = ""; $tEnc = ""; $n = ""; $trunc = ""; $batch = ""; $thr = ""; $interop = ""; $dtype = ""
                $cLoad = ""; $cEnc = ""; $dev = ""; $prm = ""; $rssMB = ""; $gpuMB = ""
                $tFirst = ""; $cFirst = ""
                $timing = $out | Select-String -Pattern "\[TIMING\]" | Select-Object -First 1
                if ($timing) {
                    if ($timing -match "t_load=([\d.]+)")   { $tLoad   = $Matches[1] }
                    if ($timing -match "t_encode=([\d.]+)") { $tEnc    = $Matches[1] }
                    if ($timing -match "\bn=(\d+)")         { $n       = $Matches[1] }
                    if ($timing -match "truncated=(\d+)")   { $trunc   = $Matches[1] }
                    if ($timing -match "batch=(\d+)")       { $batch   = $Matches[1] }
                    if ($timing -match "threads=(\d+)")     { $thr     = $Matches[1] }
                    if ($timing -match "interop=(\d+)")     { $interop = $Matches[1] }
                    if ($timing -match "dtype=(\w+)")       { $dtype   = $Matches[1] }
                    if ($timing -match "c_load=([\d.]+)")   { $cLoad   = $Matches[1] }
                    if ($timing -match "c_encode=([\d.]+)") { $cEnc    = $Matches[1] }
                    if ($timing -match "t_first=([\d.]+)")  { $tFirst  = $Matches[1] }
                    if ($timing -match "c_first=([\d.]+)")  { $cFirst  = $Matches[1] }
                    if ($timing -match "device=(\w+)")      { $dev     = $Matches[1] }
                    if ($timing -match "params=(\d+)")      { $prm     = $Matches[1] }
                    if ($timing -match "peak_rss_mb=(-?\d+)") { $rssMB = $Matches[1] }
                    if ($timing -match "peak_gpu_mb=(-?\d+)") { $gpuMB = $Matches[1] }
                }

                $uniq = @($boundProcs | Sort-Object -Unique)
                $bindingOk = $true
                if ((-not $isBaseline) -and (-not $AllCores) -and ($Device -eq "cpu")) {
                    $outside = @($boundProcs | Where-Object { $workIds -notcontains [int]$_ })
                    # OpenMP はスレッドを作り直すたびに報告するので報告の行数はスレッド数より多くなりうる
                    $bindingOk = ($thr -eq "$Threads") -and ($bound.Count -ge $Threads) -and
                                 ($uniq.Count -eq $Threads) -and ($outside.Count -eq 0)
                }

                if (($p.ExitCode -eq 0) -and (Test-Path $outFile)) {
                    $status = "ok"; $total++
                    if (-not $bindingOk) {
                        $status = "ok_unbound"; $noisy++
                        Write-Host " 完了 ($sec 秒) スレッドの固定を確認できない（threads=$thr, 固定先 $($uniq.Count) 個・報告 $($bound.Count) 行: $($uniq -join ' ')）" -ForegroundColor Yellow
                    } elseif (($busy -gt $MaxBusyPercent) -or ($otherPct -gt $MaxBusyPercent)) {
                        $status = "ok_noisy"; $noisy++
                        Write-Host " 完了 ($sec 秒) 他プロセスの負荷あり: 実行前 $busy% / 実行中 $otherPct%" -ForegroundColor Yellow
                    } elseif ($memMin -lt $MinFreeMemMB) {
                        $status = "ok_lowmem"; $noisy++
                        Write-Host " 完了 ($sec 秒) 空きメモリ不足（最小 $memMin MB）．ページングの疑い" -ForegroundColor Yellow
                    } else {
                        Write-Host " 完了 ($sec 秒)" -ForegroundColor Green
                    }
                } else {
                    $status = "failed"; $failed++
                    Write-Host " 失敗 (終了コード $($p.ExitCode))" -ForegroundColor Red
                    ($out + $err) | Select-Object -Last 5 | ForEach-Object { Write-Host "    $_" }
                }

                $stamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
                $childCpu = [math]::Round($childCpuSec, 2)
                "$stamp,$Corpus,$model,$app,$variant,$dev,$n,$trunc,$batch,$thr,$interop,$($uniq.Count),$dtype,$affinityLabel,$prm," +
                "$tWarm,$warmBytes,$tLoad,$tFirst,$tEnc,$sec," +
                "$cWarm,$cLoad,$cFirst,$cEnc,$childCpu," +
                "$busy,$otherPct,$peakMB,$rssMB,$gpuMB,$memMin,$envName,$status" |
                    Out-File -FilePath $LogFile -Append -Encoding utf8
                # 先読みの値は，それが起きた行にだけ入れる
                $tWarm = ""; $cWarm = ""; $warmBytes = ""
            }
        }
    }

    Write-Host ""
    Write-Host "===================================================="
    Write-Host " 実行 $total 件（うち計測条件の乱れ $noisy 件） / 既存のため省略 $skipped 件 / 失敗 $failed 件"
    Write-Host " ログ: $LogFile"
    if ($noisy -gt 0) {
        Write-Host " status は疑いの印．時間の採否そのものではない" -ForegroundColor Yellow
        Write-Host " ok_unbound / ok_noisy はほかのアプリを閉じ，出力ファイルを消して再実行" -ForegroundColor Yellow
        Write-Host " ok_lowmem は c_encode_s / t_encode_s が $Threads に近いかで採否を決める（研究フロー手順書 P4-3）" -ForegroundColor Yellow
    }
    Write-Host "===================================================="
    if ($failed -gt 0) { exit 1 }
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
