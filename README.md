
# In short:
Skymap is a database that offers: 1.) a single data matrix for each omic layer for each specie that spans >200k sequencing runs from all the public studies, 2.) a metadata file that extracted all the controlled keywords for the associated data matrix. The data is hosted on sage bio synapse https://www.synapse.org/skymap, take < 3 minutes to set up the account). Code is hosted in github, the jupyer-notebooks aim to show analysis examples of going from data slicing to hypothesis testing ([CLIK ME: KEY WALK THROUGH](https://github.com/brianyiktaktsui/Skymap/blob/master/DataSlicingExample.ipynb) ). 

# In long: 
##Motivation- Pooling processed data from multiple studies is time-consuming: 
When I first started in bioinformatic couple years ago, I spent much of my time doing two things: 1.) cleaning omic data matrices, e.g. mapping between gene IDs (hgnc, enseml, ucsc, etc.) for processed data matrices, trying all sort of different bioinformatics pipelines that yield basically the same results, investigating what is the exact unit being counted over when pulling data from public database, etc.  2.) cleaning metadata annotation, which usually involves extracting and mapping the labels to alias to the same categories. 

This question came to my mind: Can we merge and reduce the peta-bytes worth of public omic data in a table while capturing the commonly used information that can fit into your hard drive (<500 GB), like firehose for TCGA data? 

## Solution: An automated pipeline to generate a single data matrix that does simple counting for each specie and omic layer 
What I am offering in here is a metadata table and a single data matrix for each omic layer that encapsulate majority of the public data out there. I do believe that “Science started with counting” (a quote I loved from “Cancer: Emperor of all malady” by Siddhartha Mukherjee), and thus I offer raw counts for all the features: 1. ) the  base resolution ACGT read counts for over 200k experiments among NCBI curated SNPs, where read depth and allelic fraction are usually the main drivers for SNP calling. We also offer an expression matrix, where most counts at both transcript and gene resolution, where most normalization can be done post-hoc. 
The metadata table consists of controlled vocabulary (NCI Terminology) from free text annotations of each experiment. I used the NLM metamap engine for this purpose. The nice thing is that the UMLS ecosystem from NLM allow the IDs (Concept Unique Identifiers) to be mapped onto different ontology hierarchy to relate the terms. 
The pipeline in here is trying to suit the needs of the common use cases. In another word, most pipelines out there are more like sport cars, having custom flavors for a specific group of drivers. What I am trying to create is more like a train system, aiming to suit most needs. Unfortunately, if you have more specific requirements, what I am offering is probably not going to work. 

## Why Skymap while there are so many groups out there also trying to unify the public data
To the best of my knowledge, Skymap is the first that offer both the unified omic data and cleaned metadata. The other important aspect is that the process of data extraction is fully automated, so it is supposed to be scalable.  
Data format and coding style:
I tried to keep the code and parameters to be lean and self-explanatory for your reference, but most of the scripts I wrote are not remotely close to the industrial standard. 
The storage is in python pandas pickle format. The ecosystem in python appears to be much better at handling large dataset while offering intuitive coding interfaces. For now, Skymap is geared towards ML/data science folks who are hungry for the vast amount of data and ain’t afraid of coding.

Skymap is still in Beta V0.0. [Please feel free to leave comments](https://www.synapse.org/#!Synapse:syn11415602/discussion/default) and suggestions!!! We would love to hear feedbacks from you.
## Acknowledgement

Please considering citing if you are using Skymap. (doi:10.7303/syn11415602)
Acknowledgement: We want to thank for the advice and resources from Dr. Hannah Carter, Dr. Jill Mesirove,Dr. Trey Ideker and Shamin Mollah. We also want to thank Dr. Ruben Arbagayen, Dr. Nate Lewis for their suggestion.
The method will soon be posted in bioarchive. Also, we want to thank the Sage Bio Network for hosting the data.
Term of use: Use Skymap however you want. Just don't sue me, I have no money.

