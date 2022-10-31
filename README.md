

# VPSNN
[[Paper]](https://cinc.org/2022/Program/accepted/49_Preprint.pdf) [[Colab notebook]](https://colab.research.google.com/github/KavehSam/VPSNN/blob/main/notebooks/colab_vpsnn_cpu.ipynb)

Spiking/Artificial Variable Projection Neural Networks

A PyTorch package for architecting spiking and artifical variable projection neural network. This work was presented at Computing in Cardiology 2022. 

> Under Construction :construction:
> 
> Currently only CPU backend is supported for VP models!


# Running Online

Please take a look at [Google Colab notebook example](https://colab.research.google.com/github/KavehSam/VPSNN/blob/main/notebooks/colab_vpsnn_cpu.ipynb)
The example trains a spiking VPNN, with temporal deattenuation spike encoder in latent space, on ECG heartbeats of [MITBIH Arrhythmia] dataset (https://physionet.org/content/mitdb/1.0.0/) in order to perform binary discrimination between ectopic and normal beats. Training set was selected from 23 records (numbered from 100 to 124) while the test set includes the rest of available records in the dataset. More information about the splitting strategy of records in training and test sets can be found [here](
https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#selection).


# Running and testing on local machines

Before running [the example notebook](notebooks/vpsnn_cpu.ipynb), you need to install the python environment using:

>	conda env create -f environment.yml 


## Licensing
This project is licensed under The Apache 2.0 license.
