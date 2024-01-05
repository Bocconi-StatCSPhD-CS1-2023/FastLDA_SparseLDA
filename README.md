# FastLDA_SparseLDA
The goal of this project is to implement SparseLDA and FastLDA, two CPU-enhancing algorithms for Latent Dirichlet Allocation, originally proposed by [[Yao et. al., 2009]](https://www.researchgate.net/publication/221653450_Efficient_methods_for_topic_model_inference_on_streaming_document_collections) and [[Newman et. al., 2008]](https://www.researchgate.net/publication/221653277_Fast_collapsed_Gibbs_sampling_for_latent_Dirichlet_allocation). 
Details concerning the algorithms and my coding choices are discussed in the attached pdf file. 
This README is a demo to show how to run the code. All code mentioned here comes from the file "RunLDA.jl". 
# Initialize Corpus, Data Extraction 
First step is the generation of the corpus of data. 
```julia
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
```
The main corpus is a vector of D elements, whose entries are defined as follows: 
```julia
corpus[d][g] = (w, Ndw)

#d: Document number
#w: Word id corresponding to word at position "g"
#Ndw: Number of repetitions of word "w" in document "d"
```
 Each element (corpus[d]) is a vector corresponding to one of the documents, containing the allocated indexes and number of repetitions for each word in the document. 
