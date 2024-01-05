# FastLDA_SparseLDA
The goal of this project is to implement SparseLDA and FastLDA, two CPU-enhancing algorithms for Latent Dirichlet Allocation, originally proposed by [[Yao et. al., 2009]](https://www.researchgate.net/publication/221653450_Efficient_methods_for_topic_model_inference_on_streaming_document_collections) and [[Newman et. al., 2008]](https://www.researchgate.net/publication/221653277_Fast_collapsed_Gibbs_sampling_for_latent_Dirichlet_allocation). 
Details concerning the algorithms and my coding choices are discussed in the attached pdf file. 
This README is a demo to show how to run the code.
# Initialize Corpus, Data Exctraction from texts 
The main corpus of data is generated in the file "RunLDA.jl". A training corpus and a testing corpus are obtained from the whole corpus to enable meaningful computation of Perplexity. All words ids, as well as the size of the vocabulary and the number of running documents are defined at this stage. 
