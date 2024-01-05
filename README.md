# FastLDA_SparseLDA
The goal of this project is to implement SparseLDA and FastLDA, two CPU-enhancing algorithms for Latent Dirichlet Allocation, originally proposed by [[Yao et. al., 2009]](https://www.researchgate.net/publication/221653450_Efficient_methods_for_topic_model_inference_on_streaming_document_collections) and [[Newman et. al., 2008]](https://www.researchgate.net/publication/221653277_Fast_collapsed_Gibbs_sampling_for_latent_Dirichlet_allocation). 
Details concerning the algorithms and my coding choices are discussed in the attached pdf file. 
This README is a demo to show how to run the code. All operations mentioned here are in the file "RunLDA.jl". 
# Initialize Corpus
First step is the generation of the corpus of data. A training corpus and a testing corpus, to compute Perplexities, are obtained from the corpus.
All words IDs, as well as size of the vocabulary and the number of running documents are defined. 
The generated corpus is a vector of D elements, whose entries are defined as follows: 
```julia
corpus[d][g] = (w, Rw)

#d: Document number
#w: Word ID corresponding to word at position "g"
#Rw: Number of repetitions of word "w" in document "d"
```
 Each of the d-elements is a vector containing the allocated indexes and number of repetitions for each word in a single document. 
 # Definition of Variables
 Before running LDA, define: 
 ```julia
W = length(vocabulary)    #Size of the vocabulary
T = 4                     #Number of topics to perform LDA(choice of the implementer) 
burnin = 100              #Number of MCMC samples to discard 
sample = 50               #Number of MCMC samples on which to perform averages
```
 # Running LDA
 To Run Latent Dirichlet allocation, after having included the relevant .jl files, first define mutable structs: 
```julia
S = SPARSE_LDA.PTM(T, W)
F = FAST_LDA_22.PTM(T, W)
H = FAST_LDA_333.PTM(T, W)
```
Then run: 
```julia
SPARSE_LDA.Run_SPARSE(S, corpus_train, corpus_test, burnin, sample)
FAST_LDA_22.Run_FAST(F, corpus_train, corpus_test, burnin, sample)
FAST_LDA_333.Run_FAST(H, corpus_train, corpus_test, burnin, sample)
```
