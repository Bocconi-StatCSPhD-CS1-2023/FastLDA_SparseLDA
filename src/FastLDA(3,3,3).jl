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

function init_vars(H::PTM, corpus_train::Vector{Any}, α::Vector{Float64}, β::Float64) 
    H.D = length(corpus_train)
    D, T, W = H.D, H.T, H.W
    H.α = α 
    H.β = β 
    H.Nt = zeros(T)
    H.Ntw = zeros(T, W)
    H.Ndt = zeros(D, T)  
    H.indx_dt = zeros(D, T)
    H.Nd = zeros(D) 
    H.z = Vector{Vector{Int64}}(undef, D)
    H.b = zeros(W) 
    H.a = zeros(D)
    H.c = 0.0 
    
    for d in 1:D
        corpus_d = corpus_train[d]
        H.z[d] = Vector{Int64}[]

        for ind_w in eachindex(corpus_d)
            (w, Rw) = corpus_d[ind_w]
            push!(H.z[d], Int64[])

            for i in 1:Rw
                t = rand(1:T)
                push!(H.z[d][ind_w], t) 
                H.Ndt[d, t] += 1    
                H.Ntw[t, w] += 1  
                H.Nt[t] += 1     
                H.Nd[d] += 1     
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

function update_norms(H::PTM, d::Int64, w::Int64, i::Int64, ind_w::Int64, t_new::Int64, t_old::Int64, d_last::Int64, w_last::Int64)
    T = H.T
    W = H.W
    if d==1 && ind_w == 1 && i ==1
        H.c -= (1.0/(W*H.β + H.Nt[t_old] +1))^3
        H.c += (1.0/(W*H.β + H.Nt[t_old]))^3
    elseif t_new != t_old 
        H.c -= (1.0/(W*H.β + H.Nt[t_new] -1))^3
        H.c -= (1.0/(W*H.β + H.Nt[t_old] +1))^3
        H.c += (1.0/(W*H.β + H.Nt[t_new]))^3
        H.c += (1.0/(W*H.β + H.Nt[t_old]))^3                           
    end 

    if d_last != t_old 
        H.a[d] -= (H.α[d_last] + H.Ndt[d, d_last] -1)^3; 
        H.a[d] -= (H.α[t_old] + H.Ndt[d, t_old] +1)^3 ;
        H.a[d] += (H.α[d_last] + H.Ndt[d, d_last])^3
        H.a[d] += (H.α[t_old] + H.Ndt[d, t_old])^3
        sort_update(T, H.Ndt[d, :], H.indx_dt[d, :], d_last, t_old)
    end 

    if w_last != t_old 
        H.b[w] -= (H.β + H.Ntw[w_last, w] -1)^3
        H.b[w] -= (H.β + H.Ntw[t_old, w] +1)^3
        H.b[w] += (H.β + H.Ntw[w_last, w])^3
        H.b[w] += (H.β + H.Ntw[t_old, w])^3
    end 
end

function subtract(H::PTM, d::Int64, w::Int64, t_old::Int64)
    H.Nt[t_old] -= 1      
    H.Ndt[d, t_old] -= 1  
    H.Ntw[t_old, w] -= 1 
end

function increase(H::PTM, d::Int64, w::Int64, t_new::Int64)   
    H.Nt[t_new] += 1
    H.Ndt[d, t_new] += 1  
    H.Ntw[t_new, w] += 1
end

function FAST_GIBBS(H::PTM, corpus_train::Vector{Any}, corpus_test::Vector{Any}, burnin::Int64, sample::Int64)
    D, T, W = H.D, H.T, H.W
    iter = burnin + sample  
    Ad = zeros(T)   
    Bw = zeros(T)
    Ct = zeros(T)
    pbs = zeros(T) 
    Z = 0.0 
    d_new = zeros(Int, D)  
    w_new = zeros(Int, W)
    t_new = 0 

    for g = 1:iter 
        for t in 1:T
            Ct[t] = 1.0 / (H.Nt[t] + W*H.β) 
        end 
        H.c = sum(y -> y^3, Ct)

        for d in 1:D
            words = corpus_train[d] 

            for ind_w in eachindex(words) 
                (w, Rw) = words[ind_w]
                ts = H.z[d][ind_w]   

                for i in 1:Rw 
                    t_old = ts[i]  
                    subtract(H, d, w, t_old) 

                    if g == 1  
                        if ind_w == length(words) && i == Rw
                            H.indx_dt[d, :] .= sortperm(H.Ndt[d, :], rev = true) 
                            for t in 1:T
                                Ad[t] = (H.Ndt[d, t] + H.α[t])
                            end 
                            H.a[d] = sum(y -> y^3, Ad)
                        end 

                        for t in 1:T
                            Bw[t] = (H.Ntw[t, w] + H.β) 
                        end 
                        H.b[w] = sum(y -> y^3, Bw) 
                        
                        for t in 1:T 
                            pbs[t] = (H.Ndt[d, t] + H.α[t]) * (H.Ntw[t, w] + H.β) / (H.Nt[t] + W * H.β)
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
                        d_last = d_new[d]  
                        w_last = w_new[w] 
                        update_norms(H, d, w, i, ind_w, t_new, t_old, d_last, w_last)

                        A = H.a[d]
                        B = H.b[w] 
                        C = H.c 
                        U = rand()  
                
                        for t in 1:T 
                            q = H.indx_dt[d, t]
                            if t != 1 
                                pbs[t] = pbs[t - 1]
                            else  
                                pbs[t] = 0.0
                            end 
                            pbs[t] += (H.Ndt[d, q] + H.α[q]) * (H.Ntw[q, w] + H.β) / (H.Nt[q] + W * H.β) 

                            A -= (H.α[q] + H.Ndt[d, q])^3  
                            B -= (H.β + H.Ntw[q, w])^3 
                            C -= (1.0 / (W*H.β  + H.Nt[q]))^3
                            A = max(0.0, A)
                            B = max(0.0, B)
                            C = max(0.0, C)
                            Z_old = Z  
                            Z = pbs[t] + cbrt(A * B * C)   

                            if pbs[t] < U * Z 
                                continue 
                            elseif t == 1 || U * Z > pbs[t - 1]
                                t_new = q
                                break 
                            else 
                                U = (U * Z_old - pbs[t - 1]) * Z / (Z_old - Z) 
                                for j in 1:t 
                                    if pbs[j] >= U 
                                        t_new = H.indx_dt[d, j]
                                        break 
                                    end 
                                end 
                            end
                        end
                    end 
                    increase(H, d, w, t_new)  
                    d_new[d] = t_new    
                    w_new[w] = t_new   
                    H.z[d][ind_w][i] = t_new  
                end
            end 
        end
        update_and_sample(H, g, burnin, corpus_test)
    end 
end

function prior_update(H::PTM)   
    D, T, W = H.D, H.T, H.W
    A = sum(H.α)
    β_num = 0.0
    β_den = 0.0

    for t in 1:T
        α_num = 0.0
        α_den = 0.0
        β_num += sum(digamma.(H.Ntw[t, :] .+ H.β))
        β_den += digamma(H.Nt[t] + H.β * W)
        for d in 1:D
            α_num += digamma(H.Ndt[d, t] + H.α[t])
            α_den += digamma(sum(H.Ndt[d, :]) + A)
        end
        H.α[t] = H.α[t] * (α_num - D * digamma(H.α[t])) / (α_den - D * digamma(A))
    end
    H.β = H.β * (β_num - T * W * digamma(H.β)) / (W * β_den - T * W * digamma(H.β * W))
end

function PPLEX(H::PTM, corpus_test::Vector{Any})
    W = H.W
    A = sum(H.α)
    N = sum(H.Nd)
    lL = 0.0 

    if H.I == 1  
        H.pdw = Vector{Vector{Float64}}()
        H.pdw = [rand(Float64, length(words)) for words in corpus_test]
    end

    for (d, words) in enumerate(corpus_test) 
        for (ind_w, (w, Rw)) in enumerate(words)
            H.pdw[d][ind_w] *= (H.I - 1.0) / H.I 

            φ_tw = (H.Ntw[:, w] .+ H.β) / (H.Nt .+ H.β * W)
            θ_dt = (H.Ndt[d, :] .+ H.α) / (H.Nd[d] + A)

            H.pdw[d][ind_w] += sum(φ_tw .* θ_dt) / H.I  
            lL += Rw * log(H.pdw[d][ind_w])
        end
    end
    H.PX = exp(-lL / N)  
end

function sampling(H::PTM)
    T, W, D = H.T, H.W, H.D
    if H.I == 1
        H.Ntw_avg = zeros(T, W)
        H.Ndt_avg = zeros(D, T)  
    end
    H.Ntw_avg .= 1.0 / H.I .* H.Ntw .+ (H.I - 1) / H.I .* H.Ntw_avg
    H.Ndt_avg .= 1.0 / H.I .* H.Ndt .+ (H.I - 1) / H.I .* H.Ndt_avg
end

function update_and_sample(H::PTM, g::Int64, burnin::Int64, corpus_test::Vector{Any})
    H.Trace[:, :, g] = H.Ndt
    prior_update(H)
    if g == 1 
        println("Starting FASTLDA(3,3,3):")
    end 
    if g <= burnin
        println("Iter = ", g)
    end
    if g == burnin + 1
        println("Sampling from the posterior:")
    end
    if g > burnin
        H.I += 1  
        PPLEX(H, corpus_test)
        sampling(H)
        println("Iter = ", g, ", Perplexity = ", H.PX)  
    end
end

function Run_FAST(H::PTM, corpus_train::Vector{Any}, corpus_test::Vector{Any}, burnin = 100, sample = 50)
    init_vars(H, corpus_train, rand(H.T), rand()) 
    H.PX = 1000.0
    H.I = 0
    H.Trace = zeros(Int, H.D, H.T, (burnin + sample))
    FAST_GIBBS(H, corpus_train, corpus_test, burnin, sample)
end
end
