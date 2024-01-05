module MCMC_Plots

using Plots 
using Measures

function Runplots(Trace::Array{Int64, 3}, D::Int64, T::Int64, iter::Int64) 
    time = 1:(iter) 
    d = rand(1:D)
    tit = "Abstract:"*string(d)
    series = zeros(iter, T)
    for i = 1:T
        series[:,i] = Trace[d,i,:]
    end
    l = @layout [a{0.01h}; grid(2,2)]
    p = fill(plot(),5,1)
    p[1] = plot(title=tit,framestyle=nothing,showaxis=false,xticks=false,yticks=false,margin=0mm)
    [p[i+1] = plot(time, series[:,i], linewidth=2, legend=false, title="topic:"*string(i),titlefont=13) for i in 1:4]
    plot(p..., layout=l)
end 
end 