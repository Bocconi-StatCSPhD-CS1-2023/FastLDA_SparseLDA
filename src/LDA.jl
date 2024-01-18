module LDA

using CSV
using DataFrames
using Measures 
using SpecialFunctions 
using Plots 

export RunLDA 
export Runplots
export F 
export H
export S

include("main/FastLDA(2,2,infty).jl")
include("main/FastLDA(3,3,3).jl")
include("main/SparseLDA.jl")
include("RunLDA.jl")
include("plots/MCMCPlots.jl")
end 
