# DisoFLAG: Accurate prediction of protein intrinsic disorder and its functions using graph-based interaction protein language

This repository contains the source code used in our paper titled _Yihe Pang, Bin Liu_. DisoFLAG: Accurate prediction of protein intrinsic disorder and its functions using graph-based interaction protein language. The code is implemented to realize the predictors proposed in the paper and includes datasets and tutorials to assist users in utilizing the code. <br>

A convenient web server for DisoFLAG were provided, which can be accessed from (http://bliulab.net/DisoFLAG/).


## Citation
Upon the usage the users are requested to use the following citation:<br>
_Yihe Pang, Bin Liu_. DisoFLAG: Accurate prediction of protein intrinsic disorder and its functions using graph-based interaction protein language. (BMC biology Submitted)

## Introduction
DisoFLAG, a computational method that leverages a graph-based interaction protein language model (GiPLM) for jointly predicting disorder and its multiple potential functions. GiPLM integrates protein semantic information based on pre-trained protein language models into graph-based interaction units to enhance the correlation of the semantic representation of multiple disordered functions. The DisoFLAG predictor takes amino acid sequences as the only inputs and provides predictions of intrinsic disorder and six disordered functions for proteins, including protein-binding, DNA-binding, RNA-binding, ion-binding, lipid-binding, and flexible linker.

![image](https://github.com/YihePang/DisoFLAG/blob/main/img/fig_1.png)

**Figure.1** Schematic overview of DisoFLAG. **(a)** DisoFLAG provides predictions of six functions for intrinsically disordered regions in proteins. Joint prediction of the six functional regions results in a lower information entropy compared to individual prediction. The reduction in information entropy is known as information gain (IG), which reflects the correlation between different functions. High IG, strong correlation. **(b)** The graph-based interaction protein language model (GiPLM) architecture employed in DisoFLAG. The Bi-directional gated recurrent unit (Bi-GRU) layer is used to capture the protein contextual semantic information based on the residue embeddings extracted from the pre-trained protein language model. The subsequent attention-based gated recurrent unit (GRU) layer is used to model the global correlation among sequences and produces a hidden representation for each residue. The feature mapping layers are used to generate six different function embedding vectors for each residue. Subsequently, for each residue, the graph-based interaction unit models six functions and their correlations as a functional graph, utilizing function embedding vectors as node representations and pre-calculated IG matrix as the weighted adjacency matrix for graph edges. Finally, the propensity scores for disorder and six disordered functions were calculated based on the semantic correlation features aggregated on the functional graph by the graph convolutional network (GCN) layer.


## Usage Guide
**Step1:** Ensure that your environment meets the necessary requirements (Linux, Python 3.5+). <br>
* Download python3.5 version and install.<br>
* Download Anaconda and intall.<br>
* Using the following command to check the environment:<br>
```Bash 
python -V
conda --version
```
<br>

**Step2:** Download the model source file.
* Download the model “pytorch_model.bin” file [here](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/pytorch_model.bin) and copy it to "/DisoFLAG/protTrans/prot_t5_xl_uniref50/".<br>

**Step3:** Create and activate the required environments using the following commands:<br>
```Bash
cd DisoFLAG
conda env create -f torch.yml
conda activate torch
```
<br>

**Step4:** Put the sequence file to be predicted in FASTA format into the "DisoFLAG/temp/" folder.<br>
<br>

**Step5:** Run the predictor using the following command :<br>
```Bash
sh run.sh -i [./temp/seqfile] -o [output_type]
```
positional arguments:<br>
[./temp/seqfile]: seqfile is the input FASTA formatted sequence filename<br>
[output_type]: select the type of result file('p' or 'b'), p represents "propensity score" and b represents "binary result"<br>
<br>

**Step6:** After the prediction is completed, find the result file in the "DisoFLAG/temp/result/" folder.<br>
Explanation of result file:<br>
		Line 1: >sequence ID<br>
		Line 2: protein sequence (1-letter amino acid encoding)<br>
		Line 3: Predicted results of intrinsic disorder regions (IDR)<br>
		Line 4: Predicted results of disordered Protein-binding regions (PB)<br>
		Line 5: Predicted results of disordered DNA-binding regions (DB)<br>
		Line 6: Predicted results of disordered RNA-binding regions (RB)<br>
		Line 7: Predicted results of disordered Ion-binding regions (IB)<br>
		Line 8: Predicted results of disordered Lipid-binding regions (LB)<br>
		Line 9: Predicted results of disordered flexible linkers (DFL)<br>
  
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
