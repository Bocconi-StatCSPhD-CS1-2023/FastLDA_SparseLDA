# FastLDA_SparseLDA
The goal of this project is to implement SparseLDA and FastLDA, two CPU-enhancing algorithms for Latent Dirichlet Allocation, originally proposed by [[Yao et. al., 2009]](https://www.researchgate.net/publication/221653450_Efficient_methods_for_topic_model_inference_on_streaming_document_collections) and [[Newman et. al., 2008]](https://www.researchgate.net/publication/221653277_Fast_collapsed_Gibbs_sampling_for_latent_Dirichlet_allocation). 
Details concerning the algorithms and efficiency considerations, as well as more detailed simulations, are discussed in the attached pdf file.
This README is a demo to show how to run the code. All code mentioned here is in the file "RunLDA.jl". 
 # Definition of Variables
 When running LDA, define the variables:  
 ```julia
function RunLDA(algorithm = "All", T = 4, burnin = 100, sample = 100)

T = 4                    #Number of topics to perform LDA(choice of the implementer) 
burnin = 100             #Number of MCMC samples to discard 
sample = 100             #Number of MCMC samples to use
```
 # Running LDA
 To Run Latent Dirichlet allocation, after having included all the relevant .jl files: 
```julia
S = SPARSE_LDA.PTM(T, W)         #Mutable Struct for SparseLDA
F = FAST_LDA_22.PTM(T, W)        #Mutable Struct for FastLDA(2,2,+inf)
H = FAST_LDA_333.PTM(T, W)       #Mutable Struct for FastLDA(3,3,3) 

SPARSE_LDA.Run_SPARSE(S, corpus_train, corpus_test, burnin, sample)      
FAST_LDA_22.Run_FAST(F, corpus_train, corpus_test, burnin, sample)
FAST_LDA_333.Run_FAST(H, corpus_train, corpus_test, burnin, sample)
```
Mutable structs must be defined separately for each of the chosen algorithms, as they assume heterogeneous entries. 

An example of output (burnin = 5): 
```julia
julia> RunLDA("Sparse", burnin = 5, sample = 200) 
#Starting FASTLDA(2,2,infty):
Iter = 1
Iter = 2
Iter = 3
Iter = 4
Iter = 5
#Sampling from the posterior:
Iter = 6, Perplexity = 4.531108993623731
Iter = 7, Perplexity = 4.534064445002236
Iter = 8, Perplexity = 4.53951981935439
Iter = 9, Perplexity = 4.544411772519728
Iter = 10, Perplexity = 4.5467327859613675
...
...
```
# Format of the Corpus
First step is the generation of the corpus of data. The generated corpus is a vector of D elements, whose entries are defined as follows: 
```julia
corpus[d][g] = (w, Rw)

#d: Document number
#w: Unique word ID corresponding to the word at position "g"
#Rw: Number of repetitions of word "w" in document "d"
```
 Each of the d-elements is a vector containing the allocated indexes and number of repetitions for each word in a single document. The unique IDs for each word are defined automatically after having cleaned and made texts homogeneous. Training and testing corpuses are also defined; the training ratio is a choice of the implementer. 
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
```
To assess the mean of topic allocations for each unique word in the vocabulary: 
 ```julia
julia> F.Ntw_avg
4×14477 Matrix{Float64}:
 0.0  0.0   2.8  0.0  0.1  1.0  0.0  1.0  1.0  0.0  1.0  1.0  0.0  0.4  0.0  …  0.2  2.0  1.0  0.0  0.1  0.0  0.5  1.0  1.8  0.0  0.0  0.0  0.2  0.3  0.0       
 0.4  0.0  52.4  0.0  0.2  0.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0  3.0  0.0     0.0  0.0  0.0  0.0  1.0  0.0  1.0  0.0  0.0  2.0  2.0  0.0  1.0  1.2  1.0
 2.6  0.0  24.4  0.0  0.9  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.6  0.0     0.3  0.0  0.0  1.0  0.5  0.0  0.5  0.0  0.1  0.0  0.0  3.0  0.0  0.0  0.0       
 0.0  2.0   0.4  2.0  0.8  0.0  1.0  0.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0     1.5  0.0  0.0  0.0  0.4  2.0  0.0  0.0  0.1  0.0  0.0  0.0  0.8  0.5  0.0
 ```
# Plots
After having run LDA with one of the algorithms, it is possible to visually check MCMC convergence for document-topic allocations: 
 ```julia
Trace = F.Trace
MCMC_Plots.Runplots(Trace, D, T, burnin + sample)
 ```
This simple visualization method only works for a number of chosen topics inferior to 6. It creates plots of sampled topic allocations for each topic, from all iterations, for a document chosen at random in the corpus. In case the number of topics is higher than 6, it is possible to check convergence of each of the document-topic chains with: 
 ```julia
plot(F.Trace[1,1,:])
 ```
However, this method only works for 1 document and 1 topic at the time. 
