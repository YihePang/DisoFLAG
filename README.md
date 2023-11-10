# DisoFLAG: Accurate prediction of protein intrinsic disorder and its functions using graph-based interaction protein language

This repository contains the source code used in our paper titled _Yihe Pang, Bin Liu_. DisoFLAG: Accurate prediction of protein intrinsic disorder and its functions using graph-based interaction protein language. The code is implemented to realize the predictors proposed in the paper and includes datasets and tutorials to assist users in utilizing the code. <br>

More materials and dataset used in this study can be obtained from (http://bliulab.net/IDP_LM/download/).


## Citation
Upon the usage the users are requested to use the following citation:<br>
_Yihe Pang, Bin Liu_. IDP-LM: prediction of protein intrinsic disorder and disorder functions based on language models. (Submitted)

## Introduction
We proposed a disorder specific protein language model, IDP-BERT. The IDP-BERT was trained as a Restrictive Masked Language Model (ReMLM) to focus on the disordered regions mainly located in the N’ and C’ terminals of sequences. Furthermore, we proposed a computational predictor called IDP-LM for predicting intrinsic disorder and disorder functions by leveraging the pre-trained protein language models. IDP-LM takes the embeddings extracted from three pre-trained protein language models as the exclusive inputs, including ProtBERT, ProtT5, and IDP-BERT. The evaluation results on independent test datasets demonstrated that the IDP-LM provided high-quality prediction results for intrinsic disorder and four common disordered functions including disorder protein binding, DNA binding, RNA binding, and disorder flexible linkers.

![image](https://github.com/YihePang/IDP-LM/assets/38775429/c90910d8-4b5a-4bf4-860a-73b830cac49a)

Figure.1 Overview of IDP-LM predictor. The input sequences were processed by three language models to generate the embedding vector for each residue of proteins. The IDP-BERT disordered language model adopts the BERT architecture of stacking multiple Transformer encoders, and it was self-supervised pre-trained with the sequence data collected from the MobiDB and PDB database. Three prediction layers in IDP-LM were used to calculate per-residue disordered propensity scores based on embeddings extracted from three language models, respectively. Then the model outputs the final propensity scores and binary results by fusing the calculations from three prediction layers.


## Usage Guide
First, ensure that your environment meets the necessary requirements (Linux, Python 3.5+). Then, follow these steps to use the source code:<br> 
* Download model file [here](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/pytorch_model.bin) and copy it to "/protTrans/prot_t5_xl_uniref50/".<br>
* Download model file [here](https://huggingface.co/Rostlab/prot_bert/resolve/main/pytorch_model.bin) and copy it to "/protTrans/prot_bert/".<br>
* Create and activate the required environment of IDP-LM using the following commands:<br>
```Bash
conda env create -f IDP_LM/torch.yml 
conda activate torch
```
* Upload the sequences to be predicted in fasta format in "examples.txt" file, and using the following commands to generate the intrinsic disorder prediction results:<br>
```Bash
sh run.sh examples.txt disorder
```
Wait until the program completed, and you can find the final prediction results:"/IDP_LM/temp/results/IDR_results.txt"
* Using the following commands to generate four common disorder function prediction results (disorder protein binding, disorder DNA binding, disorder RNA binding, disorder flexible linker):<br>
```Bash
sh run.sh examples.txt function pb
sh run.sh examples.txt function db
sh run.sh examples.txt function rb
sh run.sh examples.txt function linker
```
The corresponding result files are avaliable at "/IDP_LM/temp/results/".
  
## Acknowledgments
  We acknowledge with thanks the following databases and softwares used in this study:<br> 
    		[DisProt](https://www.disprot.org/): database of intrinsically disordered proteins.<br> 
    		[MobiDB](https://mobidb.bio.unipd.it/): database of protein disorder and mobility annotations.<br> 
    		[PDB](https://www.rcsb.org/): RCSB Protein Data Bank.<br> 
    		[ProtTrans](https://github.com/agemagician/ProtTrans): Protein pre-trained language models.<br> 
