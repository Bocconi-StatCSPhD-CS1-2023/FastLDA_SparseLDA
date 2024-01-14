using CSV
using DataFrames

include("FastLDA(2,2,infty).jl")
include("FastLDA(3,3,3).jl")
include("SparseLDA.jl")
include("MCMCPlots.jl")

file_path = "AbstractsMPCSST.csv"
frame = CSV.File(file_path) |> DataFrame
abstracts = frame[:, "ABSTRACTS"]

chars_cut = r"[\\,\.!;:'\"()\[\]{}<>?/\|_+*=^%&$#~`]"
cleaned_texts = [replace.(split(abstract), chars_cut => "") for abstract in abstracts]

unique_words = Set(vcat(cleaned_texts...))
vocabulary = collect(unique_words)

check_id = Dict(word => id for (id, word) in enumerate(vocabulary))
check_word = Dict(id => word for (id, word) in enumerate(vocabulary))

corpus = []
for text in cleaned_texts
    w_ids = map(w -> check_id[w], text)
    w_counts = Dict{Int, Int}()

    for w_id in w_ids
        w_counts[w_id] = get(w_counts, w_id, 0) + 1
    end

    counts_tuple = [(w_id, counts) for (w_id, counts) in w_counts]
    push!(corpus, counts_tuple)
end

ratio = 0.85
corpus_train = []
corpus_test = []
for counts_tuple in corpus
    Nwd, = size(counts_tuple)
    length = Int(round(Nwd * ratio))
    push!(corpus_train, counts_tuple[1:length])
    push!(corpus_test, counts_tuple[length + 1:end])
end

W = length(vocabulary)
T = 4
D = length(corpus)
burnin = 80
sample = 80

S = SPARSE_LDA.PTM(T, W)
SPARSE_LDA.Run_SPARSE(S, corpus_train, corpus_test, burnin, sample)
Trace = S.Trace
MCMC_Plots.Runplots(Trace, D, T, burnin + sample)

H = FAST_LDA_333.PTM(T, W)
FAST_LDA_333.Run_FAST(H, corpus_train, corpus_test, burnin, sample)
Trace = H.Trace
MCMC_Plots.Runplots(Trace, D, T, burnin + sample)

F = FAST_LDA_22.PTM(T, W)
FAST_LDA_22.Run_FAST(F, corpus_train, corpus_test, burnin, sample)
Trace = F.Trace
MCMC_Plots.Runplots(Trace, D, T, burnin + sample)
