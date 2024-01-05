# FastLDA_SparseLDA
The goal of this project is to implement SparseLDA and FastLDA, two CPU-enhancing algorithms for Latent Dirichlet Allocation, originally proposed by [[Yao et. al., 2009]](https://www.researchgate.net/publication/221653450_Efficient_methods_for_topic_model_inference_on_streaming_document_collections) and [[Newman et. al., 2008]](https://www.researchgate.net/publication/221653277_Fast_collapsed_Gibbs_sampling_for_latent_Dirichlet_allocation). 
Details concerning the algorithms and my coding choices are discussed in the attached pdf file. 
This README is a demo to show how to run the code.
# Initialize Corpus, Data Extraction 
The main corpus of data is generated in the file "RunLDA.jl", splitting textual documents word-to-word and removing unnecessary characters. 
```julia
...

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

...
```

All words IDs as well as the size of the vocabulary and the number of running documents are defined. 




A training corpus, to perform LDA, and a testing corpus, to compute Perplexities, are then obtained. 
ratio = 0.85
corpus_train = []
corpus_test = []
for counts_tuple in corpus
    Nwd, = size(counts_tuple)
    length = Int(round(Nwd * ratio))
    push!(corpus_train, counts_tuple[1:length])
    push!(corpus_test, counts_tuple[length + 1:end])
end
