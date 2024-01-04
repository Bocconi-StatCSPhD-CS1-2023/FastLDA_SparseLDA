module FAST_LDA_22

using SpecialFunctions 

mutable struct PTM
    T::Int64    
    W::Int64    
    D::Int64
    Nt::Vector{Int64}    
    Ntw::Matrix{Int64}   
    Ndt::Matrix{Int64}  
    Nd::Vector{Int64}
    Ntw_avg::Matrix{Float64}   
    Ndt_avg::Matrix{Float64}  
    indx_dt::Matrix{Int64}
    z::Vector{Vector{Vector{Int64}}}
    α::Vector{Float64}
    β::Float64
    I::Int64   
    PX::Float64  
    pdw::Vector{Vector{Float64}}  
    Trace::Array{Int64, 3}
    PTM(T, W) = new(T, W)
end

function init_vars(F::PTM, corpus_train::Vector{Any}, α::Vector{Float64}, β::Float64) 
    F.D = length(corpus_train)
    D, T, W = F.D, F.T, F.W

    F.α = α 
    F.β = β 
    F.Nt = zeros(T)
    F.Ntw = zeros(T, W)
    F.Ndt = zeros(D, T)  
    F.indx_dt = zeros(D, T)
    F.Nd = zeros(D) 
    F.z = Vector{Vector{Int64}}(undef, D)
    
    for d in 1:D
        corpus_d = corpus_train[d]
        F.z[d] = Vector{Int64}[]

        for iv in eachindex(corpus_d)
            (w, Ndw) = corpus_d[iv]
            push!(F.z[d], Int64[])

            for i in 1:Ndw
                t = rand(1:T)
                push!(F.z[d][iv], t) 
                F.Ndt[d, t] += 1    
                F.Ntw[t, w] += 1  
                F.Nt[t] += 1     
                F.Nd[d] += 1     
            end
        end
    end 
end

function sort_update(T::Int64, Nt::Vector{Int64}, ind::Vector{Int64}, t_new::Int64, t_old::Int64)
    prev = 0

    t_new = ind[t_new]

    while t_new > 1 && Nt[ind[t_new]] > Nt[ind[t_new - 1]]
        prev = ind[t_new]
        ind[t_new] = ind[t_new - 1]
        ind[t_new - 1] = prev
        t_new -= 1
    end

    t_old = ind[t_old]

    while t_old < T && Nt[ind[t_old]] <= Nt[ind[t_old + 1]]
        prev = ind[t_old]
        ind[t_old] = ind[t_old + 1]
        ind[t_old + 1] = prev
        t_old += 1
    end
end

function subtract(F::PTM, d::Int64, w::Int64, t_old::Int64)
    F.Nt[t_old] -= 1      
    F.Ndt[d, t_old] -= 1  
    F.Ntw[t_old, w] -= 1 
end

function update_norms(F::PTM, a::Vector{Float64}, b::Vector{Float64}, d::Int64, w::Int64, 
    t_new::Int64, t_old::Int64, d_old::Int64, w_old::Int64)

    T = F.T
    if d_old != t_old
        a[d] -= (F.α[d_old] + F.Ndt[d, d_old] - 1)^2
        a[d] -= (F.α[t_old] + F.Ndt[d, t_old] + 1)^2
        a[d] += (F.α[d_old] + F.Ndt[d, d_old])^2
        a[d] += (F.α[t_old] + F.Ndt[d, t_old])^2
        sort_update(T, F.Ndt[d, :], F.indx_dt[d, :], d_old, t_new)
    end
    if w_old != t_old
        b[w] -= (F.β + F.Ntw[w_old, w] - 1)^2
        b[w] -= (F.β + F.Ntw[t_old, w] + 1)^2
        b[w] += (F.β + F.Ntw[w_old, w])^2
        b[w] += (F.β + F.Ntw[t_old, w])^2
    end
end

function increase(F::PTM, d::Int64, w::Int64, t_new::Int64)   
    F.Nt[t_new] += 1
    F.Ndt[d, t_new] += 1  
    F.Ntw[t_new, w] += 1
end

function FAST_GIBBS(F::PTM, corpus_train::Vector{Any}, corpus_test::Vector{Any}, burnin::Int64, sample::Int64)
    D, T, W = F.D, F.T, F.W
    iter = burnin + sample 
    indx_t = zeros(Int,T)    
    Ad = zeros(T)   
    Bw = zeros(T)
    pbs = zeros(T) 
    b = zeros(W) 
    a = zeros(D)
    d_last = zeros(Int, D)  
    w_last = zeros(Int, W)
    t_new = 0 
    totmin = 0
    Z = 0.0 
    C = 0.0

    for g = 1:iter 

        for d in 1:D
            words = corpus_train[d] 

            for iv in eachindex(words) 
                (w, Ndw) = words[iv]
                ts = F.z[d][iv]   

                for i in 1:Ndw  
                    t_old = ts[i]  
                    
                    subtract(F, d, w, t_old)

                    if g == 1  
                        if iv == 1 && i == 1 
                            indx_t .= sortperm(F.Nt, rev = true)    
                        elseif t_old != t_new 
                            sort_update(T, F.Nt, indx_t, t_new, t_old)
                        end 
                        totmin = F.Nt[indx_t[T]] 
                        C = 1.0 / (totmin + W * F.β)

                        if iv == length(words) && i == Ndw 
                            F.indx_dt[d, :] .= sortperm(F.Ndt[d, :], rev = true) 
                        end 
                        
                        for t in 1:T
                            Ad[t] = (F.Ndt[d, t] + F.α[t])
                            Bw[t] = (F.Ntw[t, w] + F.β) 
                        end 
                        a[d] = Ad' * Ad   
                        b[w] = Bw' * Bw 
                        
                        for t in 1:T 
                            pbs[t] = (F.Ndt[d, t] + F.α[t]) * (F.Ntw[t, w] + F.β) / (F.Nt[t] + W * F.β)
                        end 
                        Z = sum(pbs)
                        U = Z * rand()  
                        currprob = pbs[1]
                        t_new = 1

                        while U > currprob 
                            t_new += 1 
                            currprob += pbs[t_new]
                        end
                    else 
                        d_old = d_last[d]  
                        w_old = w_last[w] 
    
                        if t_new != t_old 
                            sort_update(T, F.Nt, indx_t, t_new, t_old)
                            totmin = F.Nt[indx_t[T]] 
                            C = 1.0 / (totmin + W * F.β)
                        end 
                        
                        update_norms(F, a, b, d, w, t_new, t_old, d_old, w_old)

                        A = a[d]
                        B = b[w]
                        U = rand()  
                        indx = F.indx_dt[d, :]
                        
                        for t in 1:T 
                            q = indx[t] 

                            if t != 1 
                                pbs[t] = pbs[t - 1]
                            else  
                                pbs[t] = 0.0
                            end 

                            pbs[t] += (F.Ndt[d, q] + F.α[q]) * (F.Ntw[q, w] + F.β) / (F.Nt[q] + W * F.β) 
                            A -= (F.α[q] + F.Ndt[d, q])^2  
                            B -= (F.β + F.Ntw[q, w])^2 
                            A = max(0.0, A)
                            B = max(0.0, B)

                            Z_old = Z  
                            Z = pbs[t] + sqrt(A * B) * C

                            if pbs[t] < U * Z 
                                continue 
                            elseif t == 1 || U * Z > pbs[t - 1]
                                t_new = indx[t]
                                break 
                            else 
                                U = (U * Z_old - pbs[t - 1]) * Z / (Z_old - Z) 
                                for j in 1:t 
                                    if pbs[j] >= U 
                                        t_new = indx[j]
                                        break 
                                    end 
                                end 
                            end
                        end
                    end 
                    increase(F, d, w, t_new)  
                    d_last[d] = t_new    
                    w_last[w] = t_new
                    F.z[d][iv][i] = t_new  
                end
            end 
        end
        update_and_sample(F, g, burnin, corpus_test)
    end 
end

function prior_update(F::PTM)   
    D, T, W = F.D, F.T, F.W

    A = sum(F.α)
    β_num = 0.0
    β_den = 0.0

    for t in 1:T
        α_num = 0.0
        α_den = 0.0
        β_num += sum(digamma.(F.Ntw[t, :] .+ F.β))
        β_den += digamma(F.Nt[t] + F.β * W)
        
        for d in 1:D
            α_num += digamma(F.Ndt[d, t] + F.α[t])
            α_den += digamma(sum(F.Ndt[d, :]) + A)
        end
        F.α[t] = F.α[t] * (α_num - D * digamma(F.α[t])) / (α_den - D * digamma(A))
    end
    F.β = F.β * (β_num - T * W * digamma(F.β)) / (W * β_den - T * W * digamma(F.β * W))
end

function PPLEX(F::PTM, corpus_test::Vector{Any})
    W = F.W
    
    A = sum(F.α)
    N = sum(F.Nd)
    lL = 0.0 
    
    if F.I == 1  
        F.pdw = Vector{Vector{Float64}}()
        F.pdw = [rand(Float64, length(words)) for words in corpus_test]
    end

    for (d, words) in enumerate(corpus_test)
        
        for (iw, (w, Ndw)) in enumerate(words)
            F.pdw[d][iw] *= (F.I - 1.0) / F.I 

            φ = (F.Ntw[:, w] .+ F.β) / (F.Nt .+ F.β * W)
            θ = (F.Ndt[d, :] .+ F.α) / (F.Nd[d] + A)

            F.pdw[d][iw] += sum(φ .* θ) / F.I  
            lL += Ndw * log(F.pdw[d][iw])
        end
    end
    F.PX = exp(-lL / N)  
end

function sampling(F::PTM)
    T, W, D = F.T, F.W, F.D
    
    if F.I == 1
        F.Ntw_avg = zeros(T, W)
        F.Ndt_avg = zeros(D, T)  
    end
    
    F.Ntw_avg .= 1.0 / F.I .* F.Ntw .+ (F.I - 1) / F.I .* F.Ntw_avg
    F.Ndt_avg .= 1.0 / F.I .* F.Ndt .+ (F.I - 1) / F.I .* F.Ndt_avg
end

function update_and_sample(F::PTM, g::Int64, burnin::Int64, corpus_test::Vector{Any})

    prior_update(F)
    F.Trace[:, :, g] = F.Ndt

    if g <= burnin
        println("Iter = ", g)
    end

    if g == burnin + 1
        println("Sampling from the posterior:")
    end

    if g > burnin
        F.I += 1
        PPLEX(F, corpus_test)
        sampling(F)
        println("Iter = ", g, ", Perplexity = ", F.PX)
    end
end

function Run_FAST(F::PTM, corpus_train::Vector{Any}, corpus_test::Vector{Any}, burnin = 100, sample = 50)

    init_vars(F, corpus_train, rand(F.T), rand()) 
    F.PX = 1000.0
    F.I = 0
    F.Trace = zeros(Int, F.D, F.T, (burnin + sample))
    FAST_GIBBS(F, corpus_train, corpus_test, burnin, sample)
end
end
