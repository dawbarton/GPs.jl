using GPs
using Base.Test
using StaticArrays

#--- Metric tests

@testset "Metric tests" begin

    @testset "Isotropic Euclidean" begin

        x1 = @SVector [1.3, 2.2, 4.2]
        x2 = @SVector [-0.2, 2.3, 2.1]
        iso = GPs.Metrics.Euclidean(0.1)

        @test GPs.hyperparametercount(iso) == 1
        GPs.sethyperparameters!(iso, 0.5)
        @test GPs.gethyperparameters(iso) == [0.5]

        @test GPs.Metrics.sqrdistance(iso, x1, x2) ≈ 26.68
        ans1 = GPs.Metrics.sqrdistance_dθ(iso, x1, x2)
        @test ans1 ≈ [-106.72] # check result
        @test isa(ans1, SVector) # check that types have propogated correctly
        ans2 = GPs.Metrics.sqrdistance_dx1(iso, x1, x2)
        @test ans2 ≈ [12.00, -0.80, 16.80]
        @test isa(ans2, SVector)
        ans3 = GPs.Metrics.sqrdistance_dx2(iso, x1, x2)
        @test ans3 ≈ [-12.00, 0.80, -16.80]
        @test isa(ans3, SVector)

    end

    @testset "Anisotropic Euclidean" begin

        x1 = @SVector [1.3, 2.2, 4.2]
        x2 = @SVector [-0.2, 2.3, 2.1]
        aniso = GPs.Metrics.Euclidean(MVector(0.5, 0.2, 1.2))

        @test GPs.hyperparametercount(aniso) == 3
        GPs.sethyperparameters!(aniso, [0.5, 0.2, 1.3])
        @test GPs.gethyperparameters(aniso) == MVector(0.5, 0.2, 1.3)

        @test GPs.Metrics.sqrdistance(aniso, x1, x2) ≈ 11.85946746
        ans1 = GPs.Metrics.sqrdistance_dθ(aniso, x1, x2)
        @test ans1 ≈ [-36.0, -2.5, -4.014565316]
        @test isa(ans1, SVector)
        ans2 = GPs.Metrics.sqrdistance_dx1(aniso, x1, x2)
        @test ans2 ≈ [12.00, -5.00, 2.485207100]
        @test isa(ans2, SVector)
        ans3 = GPs.Metrics.sqrdistance_dx2(aniso, x1, x2)
        @test ans3 ≈ [-12.00, 5.00, -2.485207100]
        @test isa(ans3, SVector)

    end

end

@testset "Kernel tests" begin

    @testset "Squared-exponential kernel" begin

        iso = GPs.Metrics.Euclidean(0.5)
        sqrexp = GPs.Kernels.SqrExponential(iso, 0.125)

        x1 = @SVector [1.3, 2.2, 4.2]
        x2 = @SVector [-0.2, 2.3, 2.1]

        covar = GPs.Kernels.covariance(sqrexp, x1, x2)
        @test covar ≈ 2.5138053e-08

        ans1 = GPs.Kernels.covariance_dθ(sqrexp, x1, x2, covar)
        @test ans1 ≈ [4.022088479e-07, 1.341366507e-06]
        @test isa(ans1, SVector)
        ans2 = GPs.Kernels.covariance_dx1(sqrexp, x1, x2, covar)
        @test ans2 ≈ [-1.508283182e-07, 1.005522113e-08, -2.111596459e-07]
        @test isa(ans2, SVector)
        ans3 = GPs.Kernels.covariance_dx2(sqrexp, x1, x2, covar)
        @test ans3 ≈ [1.508283182e-07, -1.005522113e-08, 2.111596459e-07]

        @test GPs.hyperparametercount(sqrexp) == 2
        GPs.sethyperparameters!(sqrexp, [0.55, 1.3])
        @test GPs.gethyperparameters(sqrexp) == [0.55, 1.3]

        @test GPs.Kernels.covariance(sqrexp, x1, x2) ≈ 0.042043656
    end

end
