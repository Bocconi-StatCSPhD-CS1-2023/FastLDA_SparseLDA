module MCMC_Plots

using Plots 
using Measures

function Runplots(Trace::Array{Int64, 3}, abs = 0) 
    D, T, iter = size(Trace)[1], size(Trace)[2], size(Trace)[3]
    time = 1:(iter) 
    if T > 6 
        error("Excessive Number of Topics")
    end 
    if abs == 0
        d = rand(1:D)
    else 
        d = abs
    end 
    tit = "Abstract:"*string(d)
    series = zeros(iter, T)
    for i = 1:T
        series[:,i] = Trace[d,i,:]
    end
    r = 0
    if T % 2 == 0
        r = div(T,2)
    else 
        r = convert(Int, ceil(T/2))
    end 
    l = @layout [a{0.01h}; grid(r,2)]
    p = fill(plot(),T+1,1)
    p[1] = plot(title=tit,framestyle=nothing,showaxis=false,xticks=false,yticks=false,margin=0mm)
    [p[i+1] = plot(time, series[:,i], color=:blue, linewidth=2, legend=false, titlefont=13) for i in 1:T]
    plot(p..., layout=l)
end 
end 
