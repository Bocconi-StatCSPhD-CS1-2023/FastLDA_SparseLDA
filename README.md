# FastLDA_SparseLDA
The goal of this project is to implement SparseLDA and FastLDA, two CPU-enhancing algorithms for Latent Dirichlet Allocation, originally proposed by [[Yao et. al., 2009]](https://www.researchgate.net/publication/221653450_Efficient_methods_for_topic_model_inference_on_streaming_document_collections) and [[Newman et. al., 2008]](https://www.researchgate.net/publication/221653277_Fast_collapsed_Gibbs_sampling_for_latent_Dirichlet_allocation). 
Details concerning the algorithms and efficiency considerations, as well as more detailed simulations, are discussed in the attached pdf file.
This README is a demo to show how to run the code. All code mentioned here is in the file "RunLDA.jl". 

 # Running LDA
 Once the project environment has been activated and instantiated, the user can run each of the algorithms separately or jointly with: 
```julia
julia> using LDA        

julia> RunLDA("Sparse") 
julia> RunLDA("Fast2")
julia> RunLDA("Fast3")
julia> RunLDA("All")       #Runs all the three algorithms in sequence 
```
The default settings for RunLDA are: 
```julia
function RunLDA(algorithm = "All", T = 4, burnin = 40, sample = 40, plots = "N")

T = 4                   #Number of topics to perform LDA(choice of the implementer) 
burnin = 40             #Number of MCMC samples to discard 
sample = 40             #Number of MCMC samples to use
plots = "N"             #Including plots or not ("Y" = include plots) 
```

An example of output (burnin = 5): 
```julia
julia> RunLDA("Fast2", 4, 5, 200, "N")
Starting FASTLDA(2,2,infty):
Iter = 1
Iter = 2
Iter = 3
Iter = 4
Iter = 5
Sampling from the posterior:
Iter = 6, Perplexity = 4.505722039714804
Iter = 7, Perplexity = 4.505224901932152
Iter = 8, Perplexity = 4.505240983564042
Iter = 9, Perplexity = 4.5060945464958655
Iter = 10, Perplexity = 4.504944294219416
...
...
``` 
# Format of the Corpus
The corpus automatically generated by RunLDA (details in the pdf) is a vector of D elements, whose entries are defined as follows: 
```julia
corpus[d][g] = (w, Rw)

#d: Document number
#w: Unique word ID corresponding to the word at position "g"
#Rw: Number of repetitions of word "w" in document "d"
```
 Each of the d-elements is a vector containing the allocated indexes and number of repetitions for each word in a single document. The unique IDs for each word are defined automatically after having cleaned and made texts homogeneous. Training and testing corpuses are also defined; the training ratio is a choice of the user and is set to 0.85 by default. 
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
Mutable structs on which all operations are performed are: 
```julia
S = SPARSE_LDA.PTM(T, W)         #Mutable Struct for SparseLDA
F = FAST_LDA_22.PTM(T, W)        #Mutable Struct for FastLDA(2,2,+inf)
H = FAST_LDA_333.PTM(T, W)       #Mutable Struct for FastLDA(3,3,3)
 ```
Each is separately defined as it assumes algorithm-specific entries. 
# Plots
After having run LDA with one of the algorithms, it is possible to visually check MCMC convergence for document-topic allocations: 
 ```julia
RunPlots(F.Trace)

#Specifying the document of interest:
RunPlots(F.Trace, 12) 
 ```
This simple visualization method only works for a number of chosen topics inferior to 6. It creates plots of sampled topic allocations for each topic, from all iterations, for a document chosen at random in the corpus. This is done automatically by RunLDA when specifying "Y" as the last entry. In case the number of topics is higher than 6, it is possible to check convergence of each of the document-topic chains with: 
 ```julia
plot(F.Trace[d,t,:])
 ```
However, this method only works for 1 document and 1 topic at the time. 
