using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition

a = zeros(10)

Threads.@threads for i = 1:10
    a[i] = Threads.threadid()
end

using ModelPredictiveControl, ControlSystemsBase

f! = (ẋ, x, u, _, _) -> (ẋ .= 0.1*(2*u[]-x[]))
h! = (y, x, _, _) -> (y .= x)
model = NonLinModel(f!, h!, 2.0, 1, 1, 1)
mpc = NonLinMPC(model, nint_ym=0, Hp=5, transcription=TrapezoidalCollocation(f_threads=false))

nu = model.nu
nk = model.nk
nx̂ = mpc.estim.nx̂
Hp, Hc = mpc.Hp, mpc.Hc
geq = -1*ones(nx̂*Hp)
X̂0  = -1*ones(nx̂*Hp)
Û0  = -1*ones(nu*Hp)
K0  = -1*ones(nk*Hp)

ΔU = zeros(nu*Hc)
U0 = ones(nu*Hp)
X̂0_Z̃ = sim!(model, 6).X_data[2:end]
ϵ = 0.0
Z̃ = [ΔU; X̂0_Z̃; ϵ]

ModelPredictiveControl.con_nonlinprogeq!(
    geq, X̂0, Û0, K0, 
    mpc, model, mpc.transcription, U0, Z̃
)
println(geq)
all(geq .≤ 5e-3) || error("NOT ALL 0.0 VALUES IN GEQ VECTOR!")
#=
@profview_allocs for i in 1:10000
    ModelPredictiveControl.con_nonlinprogeq!(
    geq, X̂0, Û0, K0, 
    mpc, model, mpc.transcription, U0, Z̃
) end
 =#

using BenchmarkTools
@btime ModelPredictiveControl.con_nonlinprogeq!(
    $geq, $X̂0, $Û0, $K0, 
    $mpc, $model, $mpc.transcription, $U0, $Z̃
)


