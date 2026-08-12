"""
Makes the nonlinear solvers of the SciML ecosystem available as `compute_ac_pf`
backends.  This extension loads when a package providing SciMLBase is present,
for example NonlinearSolve.jl or SimpleNonlinearSolve.jl, e.g.

```julia
using PowerModels, NonlinearSolve

compute_ac_pf(data, solver = NewtonRaphson())
compute_ac_pf(data, solver = TrustRegion())
```
"""
module PowerModelsSciMLExt

import LinearAlgebra
import PowerModels
import SciMLBase


"""
    SciMLBase.NonlinearProblem(sys::PowerFlowSystem)

Builds a SciML `NonlinearProblem` from a `PowerFlowSystem`.  The problem
carries the analytic jacobian, its sparsity pattern and analytic
jacobian-vector products, so solvers perform no automatic differentiation
and no sparsity detection.

Together with the SciML solver cache interface (`init`, `solve!`, `reinit!`)
this supports many repeated solves of the same network efficiently, for
example a monte carlo study over the power injections,

```julia
prob = NonlinearProblem(build_pf_system(instantiate_pf_data(data)))
cache = init(prob, NewtonRaphson(linsolve = KLUFactorization(check_pattern = false)))

for injections in samples
    reinit!(cache, cache.u; p = injections)
    sol = solve!(cache)
end
```
"""
function SciMLBase.NonlinearProblem(sys::PowerModels.PowerFlowSystem)
    # jacobian-vector products from the analytic jacobian, some algorithms
    # (e.g. LevenbergMarquardt) would otherwise differentiate through `f!`
    J_buf = copy(sys.jac_prototype)
    jvp = function (Jv, v, x, p)
        sys.j!(J_buf, x, p)
        LinearAlgebra.mul!(Jv, J_buf, v)
        return nothing
    end
    vjp = function (Jtv, v, x, p)
        sys.j!(J_buf, x, p)
        LinearAlgebra.mul!(Jtv, LinearAlgebra.transpose(J_buf), v)
        return nothing
    end

    # passing the analytic jacobian and its sparsity pattern explicitly
    # prevents the solver from falling back to AD or sparsity detection
    nf = SciMLBase.NonlinearFunction{true}(sys.f!;
        jac=sys.j!,
        jac_prototype=copy(sys.jac_prototype),
        jvp=jvp,
        vjp=vjp
    )
    return SciMLBase.NonlinearProblem(nf, copy(sys.x0), copy(sys.p0))
end


"""
    PowerFlowSolution(sol::SciMLBase.AbstractNonlinearSolution)

Converts a SciML nonlinear solution into a `PowerFlowSolution`.  Useful for
returning to the PowerModels types after driving a SciML solver directly.
"""
function PowerModels.PowerFlowSolution(sol::SciMLBase.AbstractNonlinearSolution)
    # not every SciML solver reports iteration statistics or a residual
    iterations = sol.stats === nothing ? -1 : sol.stats.nsteps
    residual_norm = sol.resid === nothing ? NaN : LinearAlgebra.norm(sol.resid, Inf)

    return PowerModels.PowerFlowSolution(sol.u, SciMLBase.successful_retcode(sol), iterations, residual_norm)
end


function PowerModels._solve_nl(sys::PowerModels.PowerFlowSystem, alg::SciMLBase.AbstractNonlinearAlgorithm; abstol=1e-8, maxiters=50, kwargs...)
    prob = SciMLBase.NonlinearProblem(sys)
    sol = SciMLBase.solve(prob, alg; abstol=abstol, maxiters=maxiters, kwargs...)
    return PowerModels.PowerFlowSolution(sol)
end

end
