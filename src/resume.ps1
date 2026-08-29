# resume_seeds.ps1 — re-runs any missing (seed, tokenizer, init) combos
$seeds = @(99, 500, 12345)
$tokenizers = @("custom", "gpt2")
$inits = @("pretrained", "scratch")

$results = Get-Content ..\results\lid_results.txt -Raw

foreach ($seed in $seeds) {
    foreach ($tok in $tokenizers) {
        foreach ($init in $inits) {
            # skip if this combo already has a saved result
            $pattern = "tokenizer=$tok init=$init seed=$seed"
            if ($results -match [regex]::Escape($pattern)) {
                Write-Host "SKIP (done): $tok $init $seed" -ForegroundColor DarkGray
                continue
            }
            Write-Host "=== RUN: $tok $init $seed ===" -ForegroundColor Cyan
            python finetune_lid.py --tokenizer $tok --init $init --seed $seed
        }
    }
}
Write-Host "=== Done. Run: python aggregate_lid.py ===" -ForegroundColor Green