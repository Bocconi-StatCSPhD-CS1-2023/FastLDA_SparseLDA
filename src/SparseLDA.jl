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
    f::Vector{Float64}
    I::Int64  
    PX::Float64  
    pdw::Vector{Vector{Float64}}
    Trace::Array{Int64, 3}
    
    PTM(T, W) = new(T, W)
end

function init_vars(S::PTM, corpus_train::Vector{Any}, alpha::Vector{Float64}, beta::Float64)
    S.D, = size(corpus_train)
    T, D, W = S.T, S.D, S.W

    S.alpha = alpha 
    S.beta = beta 
    S.r = 0.0
    S.f = zeros(T)
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

function update_buckets(S::PTM, d::Int64, last_d::Int64, iv::Int64, i::Int64, t_old::Int64, t_new::Int64)
    W = S.W

    if d==1 && iv == 1 && i ==1
        S.s -= S.alpha[t_old]* S.beta / (S.beta*W + S.Nt[t_old] +1)
        S.s += S.alpha[t_old]* S.beta / (S.beta*W + S.Nt[t_old])

    elseif t_new != t_old 
        S.s -= S.alpha[t_new] * S.beta / (S.beta*W + S.Nt[t_new]-1) 
        S.s -= S.alpha[t_old] * S.beta / (S.beta*W + S.Nt[t_old]+1)
        S.s += S.alpha[t_new] * S.beta / (S.beta*W + S.Nt[t_new])
        S.s += S.alpha[t_old] * S.beta / (S.beta*W + S.Nt[t_old]) 
    end 

    if last_d != d 
        S.r -= S.beta*(S.Ndt[d, t_old]+1) / (S.beta*W + S.Nt[t_old]+1)
        S.r += S.beta*S.Ndt[d, t_old] / (S.beta*W + S.Nt[t_old])

    elseif t_new != t_old 
        S.r -= S.beta*(S.Ndt[d, t_new]-1) / (S.beta*W + S.Nt[t_new]-1)
        S.r -= S.beta*(S.Ndt[d, t_old]+1) / (S.beta*W + S.Nt[t_old]+1)
        S.r += S.beta*S.Ndt[d, t_new] / (S.beta*W + S.Nt[t_new])
        S.r += S.beta*S.Ndt[d, t_old] / (S.beta*W + S.Nt[t_old])
    end 
end 

function update_pos_f(S::PTM, posNdt::Vector{Vector{Int64}}, posNtw::Vector{Vector{Int64}}, 
    d::Int64, w::Int64, t_old::Int64, t_new::Int64, step::Int64)
    W = S.W

    if step == 0 
        S.Ndt[d, t_old] == 0 && filter!(e->e!=t_old, posNdt[d]) 
        S.Ntw[t_old, w] == 0 && filter!(e->e!=t_old, posNtw[w]) 
        
        S.f[t_old] = (S.alpha[t_old] + S.Ndt[d, t_old])/(S.beta*W + S.Nt[t_old])
    else
        S.Ndt[d, t_new] == 1 && push!(posNdt[d], t_new)
        S.Ntw[t_new, w] == 1 && push!(posNtw[w], t_new)

        S.f[t_new] = (S.alpha[t_new] + S.Ndt[d, t_new])/(S.beta*W + S.Nt[t_new]) 
    end
end

function increase(S::PTM, d::Int64, w::Int64, t_new::Int64)  
    S.Nt[t_new] += 1
    S.Ndt[d, t_new] += 1  
    S.Ntw[t_new, w] += 1
end

function SPARSE_GIBBS(S::PTM, corpus_train::Vector{Any}, corpus_test::Vector{Any}, burnin::Int64, sample::Int64)
    T = S.T
    D = S.D
    W = S.W
    iter = burnin + sample 
    t_new = 0 
    last_d = 0

    posNdt = [findall(e -> e != 0, S.Ndt[d, :]) for d in 1:D]
    posNtw = [findall(e -> e != 0, S.Ntw[:, w]) for w in 1:W]

    for g in 1:iter 

        S.s = 0.0
        for t in 1:T
            S.s += S.alpha[t] * S.beta / (S.beta*W + S.Nt[t])  
            S.f[t] = S.alpha[t] / (S.beta*W + S.Nt[t])
        end

        for d in 1:D
            S.r = 0.0
            for t in posNdt[d]    
                S.r += S.beta*S.Ndt[d,t] / (S.beta*W + S.Nt[t]) 
                S.f[t] = (S.alpha[t] + S.Ndt[d,t]) / (S.beta*W + S.Nt[t]) 
            end

            words = corpus_train[d] 
            for iv in eachindex(words) 

                (w, Rep) = words[iv]
                ts = S.z[d][iv]  
                 
                for i in 1:Rep 
                    t_old = ts[i] 
                    step = 0 

                    subtract(S, d, w, t_old) 
                    
                    update_pos_f(S, posNdt, posNtw, d, w, t_old, t_new, step)
                    update_buckets(S, d, last_d, iv, i, t_old, t_new)

                    s = S.s
                    r = S.r 
                    q = 0.0 
                    
                    for t in posNtw[w]   
                        q += S.Ntw[t, w] * S.f[t]   
                    end
                    
                    Z = (s + r + q)
                    U = rand() * Z 
                    t_new = 0 
                    
                    if U < s  
                        U = U + r + q 
                        for t in 1:T
                            U += S.alpha[t]*S.beta / (S.beta*W + S.Nt[t])   
                            t_new = t
                            if U >= Z
                                break
                            end
                        end
                    elseif U < s + r 
                        l = 1
                        U = U + q          
                        while U <= Z
                            t_new = posNdt[d][l]      
                            U += S.beta*S.Ndt[d, t_new] / (S.beta*W + S.Nt[t_new])   
                            l += 1
                        end
                    else  
                        l = 1
                        while U <= Z
                            t_new = posNtw[w][l]
                            U += S.Ntw[t_new, w] * S.f[t_new] 
                            l += 1
                        end
                    end 
                    last_d = d 
                    step = 1 

                    increase(S, d, w, t_new)
                    update_pos_f(S, posNdt, posNtw, d, w, t_old, t_new, step)

                    S.z[d][iv][i] = t_new
                end
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

function PPLEX(S::PTM, corpus_test::Vector{Any})
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

function sampling(S::PTM)
    T, W, D = S.T, S.W, S.D
    
    if S.I == 1
        S.Ntw_avg = zeros(T, W)
        S.Ndt_avg = zeros(D, T)  
    end
    
    S.Ntw_avg .= 1.0 / S.I .* S.Ntw .+ (S.I - 1) / S.I .* S.Ntw_avg
    S.Ndt_avg .= 1.0 / S.I .* S.Ndt .+ (S.I - 1) / S.I .* S.Ndt_avg
end

function update_and_sample(S::PTM, g::Int64, burnin::Int64, corpus_test::Vector{Any})

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
        sampling(S)
        println("Iter = ", g, ", Perplexity = ", S.PX)
    end
end


function Run_SPARSE(S::PTM, corpus_train::Vector{Any}, corpus_test::Vector{Any}, burnin=100, sample=50)

    init_vars(S, corpus_train, rand(S.T),rand())
    S.PX = 1000.0
    S.I = 0
    S.Trace = zeros(Int, S.D, S.T, (burnin + sample))
    SPARSE_GIBBS(S, corpus_train, corpus_test, burnin, sample)
end 
end

