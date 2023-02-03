# Enhancing-Spectral-based-GNNs-with-Structure-aware-Sparsifiers


This is a sample code for running the SPARSY on Citeseer dataset

**1.Requirements**

networkx==2.8,.8
scipy==1.9.3,
setuptools==40.6.3,
numpy==1.23.4,
torch==1.12.1
torch_geometric = 2.1.0
metis = 0.2a5 (in)
scikit-learn=1.1.3

**2.Installing metis on Ubuntu**

      sudo apt-get install libmetis-deve

**3.Datasets**

**4.Options**

--top_k_degree:      choose top-k nodes based no the degree

--size_clique:       decide which size of clique will be processed

--percent_edges:     decide how many percentage of edges will be removed

--percent_cycle:    decide how many percentage of cycles will be processed

--pencent_cliques:   decide how many percentage of cliques will be processed

--method:    deicide which edge removal methods (include metis_sp, km_sp, bm_sp, cycle_sp, node_sp and clique_sp)
 
**5. About graph partition methods**

1.You can use the METIS, Bisecting K-Means and K-Means by yourself to generate partitions.

2.We provide you sample partitions in this folder, which are named as Citeseer_bm.npy, Citeseer_km.npy and Citeseer_metis.py

**5.Example**
  	
      python main.py. #or you need to set the above parameters to decide how to remove edges. 
