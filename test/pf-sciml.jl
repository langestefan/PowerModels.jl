@testset "test sciml solver extension" begin
    @testset "5-bus case, NewtonRaphson" begin
        data = PowerModels.parse_file("../test/data/matpower/case5.m")
        native = compute_ac_pf(data)
        sciml = compute_ac_pf(data, solver=NonlinearSolve.NewtonRaphson())

        @test sciml["termination_status"]
        @test sciml["iterations"] > 0

        for (i, bus) in native["solution"]["bus"]
            @test isapprox(bus["vm"], sciml["solution"]["bus"][i]["vm"]; atol=1e-7)
            @test isapprox(bus["va"], sciml["solution"]["bus"][i]["va"]; atol=1e-7)
        end
        for (i, gen) in native["solution"]["gen"]
            @test isapprox(gen["pg"], sciml["solution"]["gen"][i]["pg"]; atol=1e-6)
            @test isapprox(gen["qg"], sciml["solution"]["gen"][i]["qg"]; atol=1e-6)
        end
    end
    @testset "5-bus case, TrustRegion with flat start" begin
        data = PowerModels.parse_file("../test/data/matpower/case5.m")
        native = compute_ac_pf(data, flat_start=true)
        sciml = compute_ac_pf(data, solver=NonlinearSolve.TrustRegion(), flat_start=true)

        @test sciml["termination_status"]

        for (i, bus) in native["solution"]["bus"]
            @test isapprox(bus["vm"], sciml["solution"]["bus"][i]["vm"]; atol=1e-7)
            @test isapprox(bus["va"], sciml["solution"]["bus"][i]["va"]; atol=1e-7)
        end
    end
    @testset "14-bus case, NewtonRaphson" begin
        data = PowerModels.parse_file("../test/data/matpower/case14.m")
        native = compute_ac_pf(data)
        sciml = compute_ac_pf(data, solver=NonlinearSolve.NewtonRaphson())

        @test sciml["termination_status"]

        for (i, bus) in native["solution"]["bus"]
            @test isapprox(bus["vm"], sciml["solution"]["bus"][i]["vm"]; atol=1e-7)
            @test isapprox(bus["va"], sciml["solution"]["bus"][i]["va"]; atol=1e-7)
        end
    end
    @testset "5-bus case, LevenbergMarquardt" begin
        # exercises the analytic jacobian-vector products of the extension,
        # LevenbergMarquardt differentiates through `f!` without them
        data = PowerModels.parse_file("../test/data/matpower/case5.m")
        native = compute_ac_pf(data)
        sciml = compute_ac_pf(data, solver=NonlinearSolve.LevenbergMarquardt())

        @test sciml["termination_status"]

        for (i, bus) in native["solution"]["bus"]
            @test isapprox(bus["vm"], sciml["solution"]["bus"][i]["vm"]; atol=1e-6)
            @test isapprox(bus["va"], sciml["solution"]["bus"][i]["va"]; atol=1e-6)
        end
    end
    @testset "5-bus case, direct solver interface" begin
        data = PowerModels.parse_file("../test/data/matpower/case5.m")
        pf_data = PowerModels.instantiate_pf_data(data)
        sys = build_pf_system(pf_data)
        sol = PowerModels._solve_nl(sys, NonlinearSolve.NewtonRaphson())

        @test sol isa PowerFlowSolution
        @test sol.converged
        @test sol.iterations > 0
        @test sol.residual_norm <= 1e-8
        @test length(sol.x) == 2*length(data["bus"])
    end
    @testset "5-bus case, repeated solves with a solver cache" begin
        # the monte carlo pattern: build one NonlinearProblem, initialize a
        # solver cache once and reuse it for every sample, updating the
        # injections and warm starting through reinit!
        data = PowerModels.parse_file("../test/data/matpower/case5.m")
        pf_data = PowerModels.instantiate_pf_data(data)
        sys = build_pf_system(pf_data)

        prob = NonlinearSolve.NonlinearProblem(sys)
        cache = NonlinearSolve.init(prob, NonlinearSolve.NewtonRaphson(), abstol=1e-8)

        base = NonlinearSolve.solve!(cache)
        @test NonlinearSolve.SciMLBase.successful_retcode(base)
        x_base = copy(base.u)
        base_steps = base.stats.nsteps

        # 5% injection increase, warm started from the base solution
        NonlinearSolve.reinit!(cache, x_base; p=1.05 .* sys.p0)
        perturbed = NonlinearSolve.solve!(cache)
        @test NonlinearSolve.SciMLBase.successful_retcode(perturbed)
        @test perturbed.stats.nsteps <= base_steps
        @test !isapprox(perturbed.u, x_base; atol=1e-6)
        x_perturbed = copy(perturbed.u)

        # restoring the operating point reproduces the base solution
        NonlinearSolve.reinit!(cache, x_perturbed; p=copy(sys.p0))
        restored = NonlinearSolve.solve!(cache)
        @test NonlinearSolve.SciMLBase.successful_retcode(restored)
        @test isapprox(restored.u, x_base; atol=1e-6)

        # the sciml solution converts back to the PowerModels type
        pf_sol = PowerFlowSolution(restored)
        @test pf_sol isa PowerFlowSolution
        @test pf_sol.converged
        @test pf_sol.x == restored.u
        @test pf_sol.iterations == restored.stats.nsteps
        @test pf_sol.residual_norm <= 1e-8
    end
    @testset "5-bus case, iteration limit" begin
        data = PowerModels.parse_file("../test/data/matpower/case5.m")
        pf_data = PowerModels.instantiate_pf_data(data)
        sys = build_pf_system(pf_data, flat_start=true)
        sol = PowerModels._solve_nl(sys, NonlinearSolve.NewtonRaphson(), maxiters=1)

        @test !sol.converged
    end
end
