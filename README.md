# FastLDA_SparseLDA
The goal of this project is to implement SparseLDA and FastLDA, two CPU-enhancing algorithms for Latent Dirichlet Allocation, originally proposed by [[Yao et. al., 2009]](https://www.researchgate.net/publication/221653450_Efficient_methods_for_topic_model_inference_on_streaming_document_collections) and [[Newman et. al., 2008]](https://www.researchgate.net/publication/221653277_Fast_collapsed_Gibbs_sampling_for_latent_Dirichlet_allocation). 
Details concerning the algorithms and my coding choices are discussed in the attached pdf file. 
This README is a demo to show how to run the code. All code mentioned here is in the file "RunLDA.jl". 
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
 To Run Latent Dirichlet allocation, after having included the relevant .jl files, first define mutable structs, then run: 
```julia
S = SPARSE_LDA.PTM(T, W)
F = FAST_LDA_22.PTM(T, W)
H = FAST_LDA_333.PTM(T, W)

SPARSE_LDA.Run_SPARSE(S, corpus_train, corpus_test, burnin, sample)
FAST_LDA_22.Run_FAST(F, corpus_train, corpus_test, burnin, sample)
FAST_LDA_333.Run_FAST(H, corpus_train, corpus_test, burnin, sample)
```
An example of output (burnin = 5): 
```julia
julia> FAST_LDA_22.Run_FAST(F, corpus_train, corpus_test, burnin, sample)
Iter = 1
Iter = 2
Iter = 3
Iter = 4
Iter = 5
Sampling from the posterior:
Iter = 6, Perplexity = 7.734364357679075
Iter = 7, Perplexity = 7.7456977244161695
Iter = 8, Perplexity = 7.746247730839275
...
...
```
 # Results
 To assess the mean of topic allocations for each document: 
 ```julia
julia> F.Ndt_avg
1000×4 Matrix{Float64}:    
  14.0    1.7    8.0   66.3
  10.8   25.6    7.0   35.6
  67.0   50.8   93.2   11.0
  20.0   11.7   83.1    2.2
 129.6   24.1    9.4    2.9
   2.0   52.3   36.1   40.6
   4.5    5.8   28.2  120.5
   4.8   36.7   13.4   74.1
   6.0   14.2   38.1   27.7
  71.2   20.4    9.5   15.9
  11.4   38.3    5.3  111.0
  46.2   37.7   54.7   14.4
  11.8   95.0   64.7   70.5
  53.9   31.8   16.7    5.6
   ⋮
   1.9   14.3  147.0    4.8
  19.2    9.0  117.8   35.0
   1.9   37.0   29.5   43.6
  19.7   27.8   60.3    2.2
  18.3    1.1   60.8    3.8
  17.3   31.1   85.3   17.3
  13.0    2.7   35.2  110.1
 103.6   59.3   14.5   13.6
```
To assess the mean of topic allocations for each word: 
 ```julia
julia> F.Ntw_avg
4×14477 Matrix{Float64}:
 0.0  0.0   2.8  0.0  0.1  1.0  0.0  1.0  1.0  0.0  1.0  1.0  0.0  0.4  0.0  …  0.2  2.0  1.0  0.0  0.1  0.0  0.5  1.0  1.8  0.0  0.0  0.0  0.2  0.3  0.0       
 0.4  0.0  52.4  0.0  0.2  0.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0  3.0  0.0     0.0  0.0  0.0  0.0  1.0  0.0  1.0  0.0  0.0  2.0  2.0  0.0  1.0  1.2  1.0
 2.6  0.0  24.4  0.0  0.9  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.6  0.0     0.3  0.0  0.0  1.0  0.5  0.0  0.5  0.0  0.1  0.0  0.0  3.0  0.0  0.0  0.0       
 0.0  2.0   0.4  2.0  0.8  0.0  1.0  0.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0     1.5  0.0  0.0  0.0  0.4  2.0  0.0  0.0  0.1  0.0  0.0  0.0  0.8  0.5  0.0
 ```
