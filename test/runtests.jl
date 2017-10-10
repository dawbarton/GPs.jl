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

        @test GPs.gethyperparameters(iso) == MVector(0.5)

        @test GPs.Metrics.sqrdistance(iso, x1, x2) ≈ 26.68
        @test GPs.Metrics.sqrdistance_dθ(iso, x1, x2) ≈ SVector(-106.72)
        @test GPs.Metrics.sqrdistance_dx1(iso, x1, x2) ≈ SVector(12.00, -0.80, 16.80)
        @test GPs.Metrics.sqrdistance_dx2(iso, x1, x2) ≈ SVector(-12.00, 0.80, -16.80)

    end

    @testset "Anisotropic Euclidean" begin

        x1 = @SVector [1.3, 2.2, 4.2]
        x2 = @SVector [-0.2, 2.3, 2.1]
        aniso = GPs.Metrics.Euclidean(MVector(0.5, 0.2, 1.2))

        @test GPs.hyperparametercount(aniso) == 3

        GPs.sethyperparameters!(aniso, [0.5, 0.2, 1.3])

        @test GPs.gethyperparameters(aniso) == MVector(0.5, 0.2, 1.3)

        @test GPs.Metrics.sqrdistance(aniso, x1, x2) ≈ 11.85946746
        @test GPs.Metrics.sqrdistance_dθ(aniso, x1, x2) ≈ SVector(-36.0, -2.5, -4.014565316)
        @test GPs.Metrics.sqrdistance_dx1(aniso, x1, x2) ≈ SVector(12.00, -5.00, 2.485207100)
        @test GPs.Metrics.sqrdistance_dx2(aniso, x1, x2) ≈ SVector(-12.00, 5.00, -2.485207100)

    end

end
