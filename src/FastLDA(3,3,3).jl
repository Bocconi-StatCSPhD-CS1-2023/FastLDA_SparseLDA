module FAST_LDA_333

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
    a::Vector{Float64}
    b::Vector{Float64}
    c::Float64
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
    F.c = 0.0 
    F.b = zeros(W) 
    F.a = zeros(D)
    
    for d in 1:D
        corpus_d = corpus_train[d]
        F.z[d] = Vector{Int64}[]

        for ind_w in eachindex(corpus_d)
            (w, Rw) = corpus_d[ind_w]
            push!(F.z[d], Int64[])

            for i in 1:Rw
                t = rand(1:T)
                push!(F.z[d][ind_w], t) 
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

function update_norms(F::PTM, d::Int64, w::Int64, i::Int64, ind_w::Int64, t_new::Int64, t_old::Int64, d_old::Int64, w_old::Int64)
    T = F.T
    W = F.W
    
    if d==1 && ind_w == 1 && i ==1
        F.c -= (1.0/(W*F.β + F.Nt[t_old] +1))^3
        F.c += (1.0/(W*F.β + F.Nt[t_old]))^3
    elseif t_new != t_old 
        F.c -= (1.0/(W*F.β + F.Nt[t_new] -1))^3
        F.c -= (1.0/(W*F.β + F.Nt[t_old] +1))^3
        F.c += (1.0/(W*F.β + F.Nt[t_new]))^3
        F.c += (1.0/(W*F.β + F.Nt[t_old]))^3                           
    end 

    if d_old != t_old 
        F.a[d] -= (F.α[d_old] + F.Ndt[d, d_old] -1)^3; 
        F.a[d] -= (F.α[t_old] + F.Ndt[d, t_old] +1)^3 ;
        F.a[d] += (F.α[d_old] + F.Ndt[d, d_old])^3
        F.a[d] += (F.α[t_old] + F.Ndt[d, t_old])^3
        sort_update(T, F.Ndt[d, :], F.indx_dt[d, :], d_old, t_new)
    end 

    if w_old != t_old 
        F.b[w] -= (F.β + F.Ntw[w_old, w] -1)^3
        F.b[w] -= (F.β + F.Ntw[t_old, w] +1)^3
        F.b[w] += (F.β + F.Ntw[w_old, w])^3
        F.b[w] += (F.β + F.Ntw[t_old, w])^3
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
    Ad = zeros(T)   
    Bw = zeros(T)
    Ct = zeros(T)
    pbs = zeros(T) 
    Z = 0.0 
    d_last = zeros(Int, D)  
    w_last = zeros(Int, W)
    t_new = 0 

    for g = 1:iter 
        for t in 1:T
            Ct[t] = 1.0 / (F.Nt[t] + W*F.β) 
        end 
        F.c = sum(y -> y^3, Ct)

        for d in 1:D
            words = corpus_train[d] 

            for ind_w in eachindex(words) 
                (w, Rw) = words[ind_w]
                ts = F.z[d][ind_w]   

                for i in 1:Rw 
                    t_old = ts[i]  
                    subtract(F, d, w, t_old) 

                    if g == 1  
                        if ind_w == length(words) && i == Rw
                            F.indx_dt[d, :] .= sortperm(F.Ndt[d, :], rev = true) 
                            for t in 1:T
                                Ad[t] = (F.Ndt[d, t] + F.α[t])
                            end 
                            F.a[d] = sum(y -> y^3, Ad)
                        end 

                        for t in 1:T
                            Bw[t] = (F.Ntw[t, w] + F.β) 
                        end 
                        F.b[w] = sum(y -> y^3, Bw) 
                        
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

                        update_norms(F, d, w, i, ind_w, t_new, t_old, d_old, w_old)
                        A = F.a[d]
                        B = F.b[w] 
                        C = F.c 
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

                            A -= (F.α[q] + F.Ndt[d, q])^3  
                            B -= (F.β + F.Ntw[q, w])^3 
                            C -= (1.0 / (W*F.β  + F.Nt[q]))^3
                            A = max(0.0, A)
                            B = max(0.0, B)
                            C = max(0.0, C)

                            Z_old = Z  
                            Z = pbs[t] + cbrt(A * B * C)   

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
                    F.z[d][ind_w][i] = t_new  
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
        
        for (ind_w, (w, Rw)) in enumerate(words)
            F.pdw[d][ind_w] *= (F.I - 1.0) / F.I 

            φ = (F.Ntw[:, w] .+ F.β) / (F.Nt .+ F.β * W)
            θ = (F.Ndt[d, :] .+ F.α) / (F.Nd[d] + A)

            F.pdw[d][ind_w] += sum(φ .* θ) / F.I  
            lL += Rw * log(F.pdw[d][ind_w])
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
