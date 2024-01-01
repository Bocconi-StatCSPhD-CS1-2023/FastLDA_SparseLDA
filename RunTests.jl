using CSV
using DataFrames

include("FastLDA(2,2,infty).jl")
include("FastLDA(3,3,3).jl")
include("SparseLDA.jl")

file_path = "AbstractsMPCSST.csv"

cd("C:\\Users\\User\\Desktop\\CS I (Progr)")
df = CSV.File(file_path) |> DataFrame
abstracts = df[:, "ABSTRACTS"]

char_removed = r"[\\,\.!;:'\"()\[\]{}<>?/\|_+*=^%&$#~`]"
cleaned_texts = [replace.(split(abstract), char_removed => "") for abstract in abstracts]

unique_words = Set(vcat(cleaned_texts...))
vocabulary = collect(unique_words)

word_to_id = Dict(word => id for (id, word) in enumerate(vocabulary))
id_to_word = Dict(id => word for (id, word) in enumerate(vocabulary))

corpus = []
for text in cleaned_texts
    words_id = map(w -> word_to_id[w], text)
    word_counts = Dict{Int, Int}()
    for word_id in words_id
        word_counts[word_id] = get(word_counts, word_id, 0) + 1
    end

    counts_tuple = [(word_id, counts) for (word_id, counts) in word_counts]
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

T = 4
voc_size = length(vocabulary)
W = voc_size

burnin = 10 
sample = 40

S = SPARSE_LDA.PTM(T, W)
SPARSE_LDA.Run_SPARSE(S, corpus_train, corpus_test, burnin, sample)

F = FAST_LDA_22.PTM(T, W)
FAST_LDA_22.Run_FAST(F, corpus_train, corpus_test, burnin, sample)

H = FAST_LDA_333.PTM(T, W)
FAST_LDA_333.Run_FAST(H, corpus_train, corpus_test, burnin, sample)
