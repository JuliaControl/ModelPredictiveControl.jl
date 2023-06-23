using JuMP, Ipopt

optim = Model(Ipopt.Optimizer)

nvar = 1 # nvar = 1 : error, nvar = 2 : works
function Jfunc(ΔŨtup...)
    ΔŨ = collect(ΔŨtup)
    return sum(ΔŨ.^2)
end

@variable(optim, ΔŨvar[1:nvar])
register(optim, :Jfunc, nvar, Jfunc, autodiff=true)
@NLobjective(optim, Min, Jfunc(ΔŨvar...))