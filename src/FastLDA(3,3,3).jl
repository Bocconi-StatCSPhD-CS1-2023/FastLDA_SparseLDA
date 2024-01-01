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
    z::Vector{Vector{Vector{Int64}}}
    alpha::Vector{Float64}
    beta::Float64
    I::Int64    
    c::Float64
    PX::Float64  
    pdw::Vector{Vector{Float64}}
    Ntw_avg::Matrix{Float64}   
    Ndt_avg::Matrix{Float64}  
    Trace::Array{Int64, 3}
    PTM(T, W) = new(T, W)
end

function init_alpha(F::PTM, alpha)
    F.alpha = alpha 
end

function init_beta(F::PTM, beta)
    F.beta = beta
end

function init_vars(F::PTM, corpus_train) 
    F.D = length(corpus_train)
    D, T, W = F.D, F.T, F.W
    F.Nt = zeros(T)
    F.Ntw = zeros(T, W)
    F.Ndt = zeros(D, T)  
    F.Nd = zeros(D) 
    F.z = Vector{Vector{Int64}}(undef, D)
    F.c = 0.0 
    
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
    t_new::Int64, t_old::Int64, d_old::Int64, w_old::Int64, indx_dt::Matrix{Int64})

    T = F.T
    W = F.W
    
    if t_new != t_old 
        F.c -= (1.0/(W*F.beta + F.Nt[t_old]-1))^3
        F.c -= (1.0/(W*F.beta + F.Nt[t_new]+1))^3
        F.c += (1.0/(W*F.beta + F.Nt[t_old]))^3
        F.c += (1.0/(W*F.beta + F.Nt[t_new]))^3                           
    end 

    if d_old != t_old 
        a[d] -= (F.alpha[d_old] + F.Ndt[d, d_old] - 1)^3; 
        a[d] -= (F.alpha[t_old] + F.Ndt[d, t_old] + 1)^3 ;
        a[d] += (F.alpha[d_old] + F.Ndt[d, d_old])^3
        a[d] += (F.alpha[t_old] + F.Ndt[d, t_old])^3

        sort_update(T, F.Ndt[d, :], indx_dt[d, :], d_old, t_new)
    end 

    if w_old != t_old 
        b[w] -= (F.beta + F.Ntw[w_old, w] - 1)^3
        b[w] -= (F.beta + F.Ntw[t_old, w] + 1)^3
        b[w] += (F.beta + F.Ntw[w_old, w])^3
        b[w] += (F.beta + F.Ntw[t_old, w])^3
    end 
end

function increase(F::PTM, d::Int64, w::Int64, t_new::Int64)   
    F.Nt[t_new] += 1
    F.Ndt[d, t_new] += 1  
    F.Ntw[t_new, w] += 1
end

function FAST_GIBBS(F::PTM, corpus_train, corpus_test, burnin, sample )
    D, T, W = F.D, F.T, F.W
    iter = burnin + sample  
    indx_dt = zeros(Int, D, T)   
    Ad = zeros(T)   
    Bw = zeros(T)
    Ct = zeros(T)
    pbs = zeros(T) 
    b = zeros(W) 
    a = zeros(D)
    Z = 0.0 
    d_last = zeros(Int, D)  
    w_last = zeros(Int, W)
    t_new = 0 

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

                        if iv == length(words) && i == Ndw 
                            indx_dt[d, :] .= sortperm(F.Ndt[d, :], rev = true) 
                        end 
                        
                        for t in 1:T
                            Ad[t] = (F.Ndt[d, t] + F.alpha[t])
                            Bw[t] = (F.Ntw[t, w] + F.beta) 
                            Ct[t] = 1.0 / (F.Nt[t] + W*F.beta) 
                        end 
                        a[d] = sum(y -> y^3, Ad)    
                        b[w] = sum(y -> y^3, Bw) 
                        F.c = sum(y -> y^3, Ct)
                        
                        for t in 1:T 
                            pbs[t] = (F.Ndt[d, t] + F.alpha[t]) * (F.Ntw[t, w] + F.beta) / (F.Nt[t] + W * F.beta)
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
                        
                        update_norms(F, a, b, d, w, t_new, t_old, d_old, w_old, indx_dt)
                        
                        aa = a[d]
                        bb = b[w] 
                        cc = F.c 
                        
                        U = rand()  
                        indx = indx_dt[d, :]
                        
                        for t in 1:T 
                            q = indx[t] 

                            if t != 1 
                                pbs[t] = pbs[t - 1]
                            else  
                                pbs[t] = 0.0
                            end 

                            pbs[t] += (F.Ndt[d, q] + F.alpha[q]) * (F.Ntw[q, w] + F.beta) / (F.Nt[q] + W * F.beta) 

                            aa -= (F.alpha[q] + F.Ndt[d, q])^3  
                            bb -= (F.beta + F.Ntw[q, w])^3 
                            cc -= (1.0/(W*F.beta  + F.Nt[q]))^3

                            aa = max(0.0, aa)
                            bb = max(0.0, bb)
                            cc = max(0.0, cc)

                            Zp_old = Z  
                            Z = pbs[t] + cbrt(aa * bb * cc)   

                            if pbs[t] < U * Z 
                                continue 
                            elseif t == 1 || U * Z > pbs[t - 1]
                                t_new = indx[t]
                                break 
                            else 
                                U = (U * Zp_old - pbs[t - 1]) * Z / (Zp_old - Z) 
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

    alpha_sum = sum(F.alpha)
    beta_num = 0.0
    beta_den = 0.0

    for t in 1:T

        alpha_num = 0.0
        alpha_den = 0.0
        beta_num += sum(digamma.(F.Ntw[t, :] .+ F.beta))
        beta_den += digamma(F.Nt[t] + F.beta * W)

        for d in 1:D
            alpha_num += digamma(F.Ndt[d, t] + F.alpha[t])
            alpha_den += digamma(sum(F.Ndt[d, :]) + alpha_sum)
        end

        F.alpha[t] = F.alpha[t] * (alpha_num - D * digamma(F.alpha[t])) / (alpha_den - D * digamma(alpha_sum))
    end

    F.beta = F.beta * (beta_num - T * W * digamma(F.beta)) / (W * beta_den - T * W * digamma(F.beta * W))
end

function PPLEX(F::PTM, corpus_test)
    W = F.W
   
    alpha_sum = sum(F.alpha)
    N = sum(F.Nd)
    LL = 0.0 
    
    if F.I == 1   
        F.pdw = Vector{Vector{Float64}}()
        F.pdw = [rand(Float64, length(words)) for words in corpus_test]
    end
    
    for (d, words) in enumerate(corpus_test)
        
        for (iw, (w, Ndw)) in enumerate(words)
            F.pdw[d][iw] *= (F.I - 1.0) / F.I  
            
            phi_tw = (F.Ntw[:, w] .+ F.beta) / (F.Nt .+ F.beta * W)
            theta_dw = (F.Ndt[d, :] .+ F.alpha) / (F.Nd[d] + alpha_sum)
    
            F.pdw[d][iw] += sum(phi_tw .* theta_dw) / F.I  
            LL += Ndw * log(F.pdw[d][iw])
        end
    end

    F.PX = exp(-LL / N)  
end

function sample_Ntw(F::PTM)
    T, W = F.T, F.W

    if F.I == 1  
        F.Ntw_avg = zeros(T, W)
    end
    
    F.Ntw_avg .= 1.0 / F.I .* F.Ntw .+ (F.I - 1) / F.I .* F.Ntw_avg  
end

function sample_Ndt(F::PTM)
    D, T = F.D, F.T

    if F.I == 1  
        F.Ndt_avg = zeros(D, T)
    end
    
    F.Ndt_avg .= 1.0 / F.I .* F.Ndt .+ (F.I - 1) / F.I .* F.Ndt_avg  
end

function update_and_sample(F, g, burnin, corpus_test)

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
        sample_Ndt(F)
        sample_Ntw(F)
        println("Iter = ", g, ", Perplexity = ", F.PX)  
    end
end

function Run_FAST(F::PTM, corpus_train, corpus_test, burnin = 100, sample = 50)
    
    init_alpha(F, rand(F.T)) 
    init_beta(F, rand())  
    init_vars(F, corpus_train) 
    F.PX = 1000.0
    F.I = 0
    F.Trace = zeros(Int, F.D, F.T, (burnin + sample))
    FAST_GIBBS(F, corpus_train, corpus_test, burnin, sample)
end
end

