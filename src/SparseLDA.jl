module SPARSE_LDA

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
    z::Vector{Vector{Vector{Int64}}}
    alpha::Vector{Float64}
    beta::Float64
    s::Float64
    r::Float64
    I::Int64  
    PX::Float64  
    pdw::Vector{Vector{Float64}}
    Trace::Array{Int64, 3}
    
    PTM(T, W) = new(T, W)
end

function init_alpha(S::PTM, alpha)
    S.alpha = alpha
    return
end

function init_beta(S::PTM, beta)
    S.beta = beta
    return
end

function init_vars(S::PTM, corpus_train)
    S.D, = size(corpus_train)
    T = S.T
    D = S.D
    W = S.W
    S.r = 0.0
    S.Nt = zeros(T)
    S.Ntw = zeros(T, W)
    S.Ndt = zeros(D, T)
    S.Nd = zeros(D)   
    S.z = Vector{Vector{Int64}}(undef, D)

    for d in 1:D
        corpus_d = corpus_train[d]
        S.z[d] = Vector{Int64}[]

        for iw in eachindex(corpus_d)
            (w, Ndw) = corpus_d[iw]
            push!(S.z[d], Int64[])

            for i in 1:Ndw
                t = rand(1:T)
                push!(S.z[d][iw], t)  
                S.Ndt[d,t] += 1   
                S.Ntw[t,w] += 1   
                S.Nt[t] += 1     
                S.Nd[d] += 1     
            end
        end
    end
end

function subtract(S::PTM, d::Int64, w::Int64, t_old::Int64)
    S.Nt[t_old] -= 1      
    S.Ndt[d, t_old] -= 1  
    S.Ntw[t_old, w] -= 1 
end

function increase(S::PTM, d::Int64, w::Int64, t_new::Int64)   
    S.Nt[t_new] += 1
    S.Ndt[d, t_new] += 1  
    S.Ntw[t_new, w] += 1
end

function first_update(S::PTM, d::Int64, w::Int64, t_old::Int64, c::Vector{Float64})
    W = S.W

    S.s -= S.alpha[t_old]*S.beta/(S.beta*W + S.Nt[t_old]) 
    S.r -= S.beta*S.Ndt[d, t_old]/(S.beta*W + S.Nt[t_old])

    subtract(S, d, w, t_old)

    c[t_old] = (S.alpha[t_old] + S.Ndt[d, t_old])/(S.beta*W + S.Nt[t_old])

    S.s += S.alpha[t_old]*S.beta/(S.beta*W + S.Nt[t_old])
    S.r += S.beta*S.Ndt[d, t_old]/(S.beta*W + S.Nt[t_old])
end

function second_update(S::PTM, d::Int64, w::Int64, t_new::Int64, c::Vector{Float64})
    W = S.W
    
    S.s -= S.alpha[t_new]*S.beta/(S.beta*W + S.Nt[t_new])
    S.r -= S.beta*S.Ndt[d, t_new]/(S.beta*W + S.Nt[t_new])

    increase(S, d, w, t_new)

    c[t_new] = (S.alpha[t_new] + S.Ndt[d, t_new])/(S.beta*W + S.Nt[t_new])
    
    S.s += S.alpha[t_new]*S.beta/(S.beta*W + S.Nt[t_new])
    S.r += S.beta*S.Ndt[d, t_new]/(S.beta*W + S.Nt[t_new])
end

function SPARSE_GIBBS(S::PTM, corpus_train, corpus_test, burnin, sample)
    T = S.T
    D = S.D
    W = S.W
    iter = burnin + sample 
    c = zeros(T)
    
    posNdt = [findall(x -> x != 0, S.Ndt[d, :]) for d in 1:D]
    posNtw = [findall(x -> x != 0, S.Ntw[:, w]) for w in 1:W]

    for g in 1:iter 

        S.s = 0.0
        for t in 1:T
            S.s += S.alpha[t]*S.beta / (S.beta*W + S.Nt[t])  
            c[t] = S.alpha[t] / (S.beta*W + S.Nt[t])  
        end

        for d in 1:D

            S.r = 0.0
            for t in posNdt[d]    
                S.r += S.beta*S.Ndt[d,t] / (S.beta*W + S.Nt[t])  
                c[t] = (S.alpha[t] + S.Ndt[d,t]) / (S.beta*W + S.Nt[t]) 
            end

            words = corpus_train[d] 
            for iv in eachindex(words) 

                (w, Rep) = words[iv]
                ts = S.z[d][iv]  
                 
                for i in 1:Rep 
                    t_old = ts[i] 
                    first_update(S, d, w, t_old, c) 

                    r = S.r 
                    s = S.s
                    
                    if S.Ndt[d, t_old] == 0    
                        filter!(e->e!=t_old, posNdt[d])  
                    end
                    if S.Ntw[t_old, w] == 0
                        filter!(e->e!=t_old, posNtw[w])   
                    end
                    
                    q = 0.0 
                    for t in posNtw[w]   
                        q += S.Ntw[t, w] * c[t]   
                    end
                    
                    zsum = rand() * (s + r + q)  
                    t_new = 0 
                    
                    if zsum < q        
                        l = 1
                        while zsum > 0
                            t_new = posNtw[w][l]
                            zsum -= S.Ntw[t_new, w] * c[t_new] 
                            l += 1
                        end
                        
                    elseif zsum < r + q
                        l = 1
                        zsum = zsum - q            
                        while zsum > 0 
                            t_new = posNdt[d][l]      
                            zsum -= S.beta*S.Ndt[d, t_new] / (S.beta*W + S.Nt[t_new])   
                            l += 1
                        end
                        
                    else zsum <= r + q + s   
                        zsum = zsum - (q + r)      
                        t_new = 0
                        for t in 1:T
                            zsum -= S.alpha[t]*S.beta / (S.beta*W + S.Nt[t])   
                            t_new = t
                            if zsum < 0
                                break
                            end
                        end
                    end

                    second_update(S, d, w, t_new, c)   
                    
                    if S.Ndt[d, t_new] == 1
                        push!(posNdt[d], t_new)
                    end
                    if S.Ntw[t_new, w] == 1
                        push!(posNtw[w], t_new)
                    end

                    S.z[d][iv][i] = t_new  
                end
            end 
            for t in posNdt[d]
                c[t] = S.alpha[t] / (S.beta*W + S.Nt[t])
            end
        end
       
        update_and_sample(S, g, burnin, corpus_test)
    end
end


function prior_update(S::PTM)   
    T, W, D = S.T, S.W, S.D
    
    alpha_sum = sum(S.alpha)
    beta_num=0.0
    beta_den=0.0

    for t in 1:T

        alpha_num = 0.0
        alpha_den = 0.0
        beta_num += sum(digamma.(S.Ntw[t,:] .+ S.beta))
        beta_den += digamma(S.Nt[t]+S.beta*W)

        for d in 1:D
            alpha_num += digamma(S.Ndt[d,t]+S.alpha[t])
            alpha_den += digamma(sum(S.Ndt[d,:])+alpha_sum)
        end
        S.alpha[t] = S.alpha[t]*(alpha_num - D*digamma(S.alpha[t]))/(alpha_den - D*digamma(alpha_sum))
    end
    S.beta = S.beta*(beta_num - T*W*digamma(S.beta))/(W*beta_den - T*W*digamma(S.beta*W))
end

function PPLEX(S::PTM, corpus_test)
    W = S.W
    
    alpha_sum = sum(S.alpha)
    N = sum(S.Nd)
    LL = 0.0 
    
    if S.I == 1
        S.pdw = Vector{Vector{Float64}}()
        S.pdw = [rand(Float64, length(words)) for words in corpus_test]
    end
    
    for (d, words) in enumerate(corpus_test)

        for (iw, (w, Ndw)) in enumerate(words)

            S.pdw[d][iw] *= (S.I - 1.0)/S.I 

            phi_tw = (S.Ntw[:, w] .+ S.beta) / (S.Nt .+ S.beta * W)
            theta_dt = (S.Ndt[d,:] .+ S.alpha) / (S.Nd[d] + alpha_sum)
            
            S.pdw[d][iw] += sum(phi_tw .* theta_dt) /S.I
            LL += Ndw * log(S.pdw[d][iw])
        end
    end
    S.PX = exp(-LL / N)
end

function sample_Ntw(S::PTM)
    T, W = S.T, S.W
    
    if S.I == 1
        S.Ntw_avg = zeros(T, W)  
    end
    
    S.Ntw_avg .= 1.0 / S.I .* S.Ntw .+ (S.I - 1) / S.I .* S.Ntw_avg
end

function sample_Ndt(S::PTM)
    D, T = S.D, S.T
    
    if S.I == 1
        S.Ndt_avg = zeros(D, T)  
    end
    
    S.Ndt_avg .= 1.0 / S.I .* S.Ndt .+ (S.I - 1) / S.I .* S.Ndt_avg
end

function update_and_sample(S::PTM, g, burnin, corpus_test)

    prior_update(S)
    S.Trace[:, :, g] = S.Ndt 

    if g <= burnin
        println("Iter = ", g)
    end
    if g == burnin + 1
        println("Sampling from the posterior:")
    end
    if g > burnin
        S.I += 1  
        PPLEX(S, corpus_test)
        sample_Ndt(S)
        sample_Ntw(S)
        println("Iter = ", g, ", Perplexity = ", S.PX)
    end
end


function Run_SPARSE(S::PTM, corpus_train, corpus_test, burnin=100, sample=50)

    init_alpha(S, rand(S.T)) 
    init_beta(S, rand()) 
    init_vars(S, corpus_train) 
    S.PX = 1000.0
    S.I = 0
    S.Trace = zeros(Int, S.D, S.T, (burnin + sample))
    SPARSE_GIBBS(S, corpus_train, corpus_test, burnin, sample)
end 
end
