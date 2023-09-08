module Takens

using DelayEmbeddings: selfmutualinfo, estimate_delay, delay_afnn, Dataset, mdop_embedding
using Peaks: argmaxima, argminima
using PyCall: pyimport

export aFNN, MDOP

function aFNN(X)

    epoch = size(X)[1]
    d = size(X)[2]
    max_delay = 30
    max_dim = 30


    global w = Vector{Float64}(undef, max_delay)
    global D = Vector{Float64}(undef, max_dim)

    for i = 1:epoch
        X1 = X[i, :, :]
        df_X = Dataset(X1')

        for j = 1:d
            AMI = selfmutualinfo(df_X[:, j], 1:max_delay)
            τ = estimate_delay(df_X[:, j], "mi_min")
            𝒟 = delay_afnn(df_X[:, j], τ, 1:max_dim)
            w = w + AMI
            D = D + 𝒟
        end
    end

    w = w/(d*epoch)
    D = D/(d*epoch)

    D_E = minimum(argmaxima(D; strict=false))
    delay = minimum(argminima(w; strict=false))

    return D_E, delay
end


function MDOP(X)

    epoch = size(X)[1]
    d = size(X)[2]
    delay = 0.0
    D_E = 0.0
    max_delay = 30
    max_dim = 30

    np = pyimport("numpy")

    for i = 1:epoch
        X1 = X[i, :, :]
        df_X = Dataset(X1')

        w = Vector{Int64}(undef, d)
        for j = 1:d
            τ = estimate_delay(df_X[:, j], "mi_min")
            w[j] = τ
            end
        theiler = maximum(w)

        Y_m, τ_vals_m, ts_vals_m, = mdop_embedding(df_X; r = 2, w = theiler)

        delay = delay + round(np.mean(τ_vals_m))
        D_E = D_E + size(τ_vals_m)[1]
    end

    delay = round(Int, delay/epoch)
    D_E = round(Int, D_E/epoch)

    return D_E, delay
end


end # module

