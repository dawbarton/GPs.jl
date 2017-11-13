using GPs
using Base.Test
using StaticArrays

#--- Metric tests

@testset "Utils tests" begin

    for x in [toSVector([1, 2, 3]),
              toSVector([SVector(1), SVector(2), SVector(3)]),
              toSVector([1 2 3]),
              toSVector(1:3)]
        @test x == [SVector(1), SVector(2), SVector(3)]
        @test isa(x, Vector{SVector{1, Int}})
    end
    for x in [toMatrix([1, 2, 3]),
              toMatrix([SVector(1), SVector(2), SVector(3)]),
              toMatrix([1 2 3])]
        @test x == [1 2 3]
        @test isa(x, Matrix)
    end

end

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

        ans1 = GPs.Kernels.covariance_dθ(sqrexp, x1, x2)
        @test ans1 ≈ [4.022088479e-07, 1.341366507e-06]
        @test isa(ans1, SVector)
        ans2 = GPs.Kernels.covariance_dx1(sqrexp, x1, x2)
        @test ans2 ≈ [-1.508283182e-07, 1.005522113e-08, -2.111596459e-07]
        @test isa(ans2, SVector)
        ans3 = GPs.Kernels.covariance_dx2(sqrexp, x1, x2)
        @test ans3 ≈ [1.508283182e-07, -1.005522113e-08, 2.111596459e-07]

        @test GPs.hyperparametercount(sqrexp) == 2
        GPs.sethyperparameters!(sqrexp, [0.55, 1.3])
        @test GPs.gethyperparameters(sqrexp) == [0.55, 1.3]

        @test GPs.Kernels.covariance(sqrexp, x1, x2) ≈ 0.042043656
    end

end

@testset "Covariance matrix tests" begin

    iso = GPs.Metrics.Euclidean(0.5)
    sqrexp = GPs.Kernels.SqrExponential(iso, 0.125)

    x1 = [SVector(1.3, 2.2, 4.2), SVector(-0.2, 1.2, -0.1)]
    x2 = [SVector(-0.2, 2.3, 2.1)]

    @test GPs.Kernels.covariance_matrix(sqrexp, x1, x2) ≈ [2.513805299e-08; 8.68673944e-08]
    M1 = GPs.Kernels.covariance_matrix(sqrexp, x1)
    @test M1 ≈ [0.015625 2.045012434e-21; 2.045012434e-21 0.015625]

    M2 = zeros(2, 2)
    GPs.Kernels.covariance_matrix!(UpperTriangular(M2), sqrexp, x1)
    M1[2, 1] = 0.0
    @test M1 == M2

end

@testset "Gaussian Process (Gaussian likelihood) tests" begin

    n = 21
    x = linspace(0, 1, n)
    y = sin.(2π*x)

    σn = 0.3
    σf = 0.1
    ℓ = 0.2

    iso = GPs.Metrics.Euclidean(ℓ)
    sqrexp = GPs.Kernels.SqrExponential(iso, σf)
    training = GPs.DataSets.TrainingSet(x, y)
    gp = GPs.GaussianLikelihoods.GP(sqrexp, σn, training)

    @test isa(gethyperparameters(gp), SVector{3, Float64})

    # Calculate the mean between the training points
    xx = 0.5*(x[2:end] + x[1:end-1])

    # From Matlab GPML toolbox (ell = log(0.2), sf = log(0.1), lik_sn = log(0.3))
    #   n = 21;
    #   x = linspace(0, 1, n);
    #   y = sin(x);
    #   xx = 0.5*(x(2:end) + x(1:end-1))
    #   meanfunc = [];                    % empty: don't use a mean function
    #   covfunc = @covSEiso;              % Squared Exponental covariance function
    #   likfunc = @likGauss;              % Gaussian likelihood
    #   hyp = struct('mean', [], 'cov', [log(0.2) log(0.1)], 'lik', log(0.1));
    #   [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xx);

    μ₀ = [0.202764557432603 0.2532874969517 0.298316602533046 0.330784847496406 0.34404680063947 0.333129086919798 0.295761669144916 0.232966591637241 0.149093996664424 0.0513161114615125 -0.0513161114615124 -0.149093996664424 -0.232966591637241 -0.295761669144916 -0.333129086919798 -0.34404680063947 -0.330784847496406 -0.298316602533047 -0.2532874969517 -0.202764557432603]
    σ₀² = [0.0968914136106435,0.0964262935050147,0.0961104866357293,0.0959299264929463,0.0958451743017393,0.0958140514271935,0.0958060359999692,0.0958050354333621,0.0958051126669153,0.0958050947257113,0.0958050947257113,0.0958051126669153,0.0958050354333621,0.0958060359999692,0.0958140514271935,0.0958451743017393,0.0959299264929463,0.0961104866357293,0.0964262935050147,0.0968914136106435]

    μ = mean(gp, xx)
    @test μ ≈ μ₀
    σ² = var(gp, xx)
    @test σ² ≈ σ₀²
    K = cov(gp, xx)
    @test diag(K) ≈ σ₀²

    # Test multiple outputs
    training2 = GPs.DataSets.TrainingSet(x, [y y]')
    gp2 = GPs.GaussianLikelihoods.GP(sqrexp, σn, training2)

    μ2 = mean(gp2, xx)
    @test μ2[1, :] == μ2[2, :]
    @test μ2[1:1, :] ≈ μ₀

    # Test likelihood calculations
    x = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    y = [0.107533 0.675794 0.136016 0.981452 1.01481 0.738462 0.864338 0.877542 1.30346 0.862904 -0.269977 0.297968 -0.442704 -0.821628 -0.808108 -1.04099 -0.975885 -0.511077 -0.305978 -0.0255785 0.134299; -0.241497 0.452465 0.913832 0.906796 1.158 1.14538 0.890368 0.867791 0.430329 0.486696 -0.229414 -0.522791 -0.749685 -1.39787 -0.66338 -0.934962 -1.10204 -0.534957 -0.930089 -0.329465 -0.0482894]
    σn = 0.1
    σf = 0.1
    ℓ = 0.2

    iso = GPs.Metrics.Euclidean(ℓ)
    sqrexp = GPs.Kernels.SqrExponential(iso, σf)
    GP = GPs.GaussianLikelihoods.GP
    trainingy = GPs.DataSets.TrainingSet(x, y)
    trainingy1 = GPs.DataSets.TrainingSet(x, y[1, :])
    trainingy2 = GPs.DataSets.TrainingSet(x, y[2, :])

    @test round(loglikelihood(GP(sqrexp, σn, trainingy1)), 2) ≈ round(136.272815830685, 2)
    @test round(loglikelihood(GP(sqrexp, σn, trainingy2)), 2) ≈ round(132.426903505274, 2)
    @test round(loglikelihood(GP(sqrexp, σn, trainingy)), 2) ≈ round(mean([136.272815830685, 132.426903505274]), 2)

    @test loglikelihood_dθ(GP(sqrexp, σn, trainingy)) ≈ [-1452.713883, -1531.4908, 306.8887837]

    # Test number type propogation
    n = 21
    x = convert(Vector{Float32}, linspace(0, 1, n))
    y = sin.(Float32(2)*π*x)

    σn = Float32(0.3)
    σf = Float32(0.1)
    ℓ = Float32(0.2)

    iso = GPs.Metrics.Euclidean(ℓ)
    sqrexp = GPs.Kernels.SqrExponential(iso, σf)
    training = GPs.DataSets.TrainingSet(x, y)
    gp = GPs.GaussianLikelihoods.GP(sqrexp, σn, training)

    # Calculate the mean between the training points
    xx = Float32(0.5)*(x[2:end] + x[1:end-1])

    μ = mean(gp, xx)
    @test isa(μ, Matrix{Float32})
    σ² = var(gp, xx)
    @test isa(σ², Vector{Float32})
    K = cov(gp, xx)
    @test isa(K, Matrix{Float32})
    L = loglikelihood(gp)
    @test isa(L, Float32)

end
