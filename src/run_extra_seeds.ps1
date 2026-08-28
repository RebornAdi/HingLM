# run_extra_seeds.ps1
# Adds 4 new seeds (7, 123, 2718, 31337) to reach 7 total.
# Existing seeds 1337, 42, 2024 are already in lid_results.txt - do NOT re-run them.
# Run this from your src/ directory: .\run_extra_seeds.ps1

$seeds = @(7, 123, 2718, 31337)
$tokenizers = @("custom", "gpt2")
$inits = @("pretrained", "scratch")

$total = $seeds.Count * $tokenizers.Count * $inits.Count
$done = 0

foreach ($seed in $seeds) {
    foreach ($tok in $tokenizers) {
        foreach ($init in $inits) {
            $done++
            Write-Host "`n=== [$done/$total] tokenizer=$tok init=$init seed=$seed ===" -ForegroundColor Cyan
            python finetune_lid.py --tokenizer $tok --init $init --seed $seed
            if ($LASTEXITCODE -ne 0) {
                Write-Host "!!! FAILED: tokenizer=$tok init=$init seed=$seed" -ForegroundColor Red
            }
        }
    }
}

Write-Host "`n=== All $total runs complete. Now run: python aggregate_lid.py ===" -ForegroundColor Green