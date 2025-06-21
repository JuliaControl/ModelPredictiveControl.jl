# To run benchmarks locally, BenchmarkCI should be added to root project. Then call:
# ```bash
# julia benchmark/run_benchmarks.jl
# ```

using BenchmarkCI
on_CI = haskey(ENV, "GITHUB_ACTIONS")

BenchmarkCI.judge(; baseline = "origin/main")
on_CI ? BenchmarkCI.postjudge() : BenchmarkCI.displayjudgement()