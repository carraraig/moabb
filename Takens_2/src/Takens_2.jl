module Takens_2

using DelayEmbeddings: selfmutualinfo, estimate_delay, delay_afnn, Dataset
using Peaks: argmaxima, argminima

export takens

function takens(X)

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


end # module

