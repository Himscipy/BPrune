Bayesian Neural Network Pruning
===============================

![](Logo_Bprune.png)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Description:
------------

BPrune is developed to perform inference and pruning of Bayesian Neural Networks(BNN) models developed with [Tensorflow](https://www.tensorflow.org/) and [Tensorflow Probability](https://www.tensorflow.org/probability). The BNN's supported by the package are one which uses mean field approximation principle of VI i.e uses gaussian to define the priors on the weights. Currently, the pruning threshold is based on the signal to noise ratio threshold.

Features:
---------

1.  Library for performing inference for trained Bayesian Neural Network (BNN).
2.  Library for performing pruning trained Bayesian Neural Network(BNN).
3.  Supports Tensorflow and Tensorflow\_probability based Bayesian Neural Network model architecture.
4.  Independent to the BNN's learning task, support BNN models for classification & regression.
5.  Capabilities of handling BNN's which are trained with distributed training libraries such as Horovod.

Installation Instructions:
--------------------------

-   Before installation ensure that you have a working Tensorflow and Tensorflow probability environment.

``` python3
python3 install -r requirements.txt
python3 install setup.py 
```

If you are using a pip installation, simply do

``` {.sourceCode .python3}
python3 -m pip install BPrune
```

-   For development of the package following command can be used after git clone.

``` {.sourceCode .}
python3 setup.py develop
```

Quick Start Guide
-----------------

-   Before running the model for inference or for pruning ensure that at the end of the training script details about the layer names and operations in the graph are written as text files.
-   To achieve this user can use the utility provided with BPrune named as Graph\_Info\_Writer.
-   The usage of the utility is described as follows:

    ``` python3  
    import numpy as np
    import tensorflow as tf
    :
    import bprune.src.utils as UT

    #
    # All the code for training the BNN
    ...
    ..
    ..

    # This path will be used as model_dir path in the argument when running BNN for inference
    case_dir = path/to/the/casefolder

    UT.Graph_Info_Writer(case_dir)
    ```

-   For successful run of BPrune following files ('LayerNames.txt', 'Ops\_name\_BNN.txt') must be present in the case directory. The above described procedure will ensure these files are written at the end of the BNN training procedure.
-   Once the required text files are written at the end of training, BPrune can be used. The example use case can be found in example folder with the package.
-   The runtime arguments to a BPrune code can be provide using command-line or can be specified using a text file each line stating the argument. example:

    > ``` shell
    > python Prune_BNN_MNIST_Model.py @ArgFilePrune.txt
    > ```

Limitations/TODO's:
-------------------

> -   Only support models trained using tensorflow placeholders for feeding data to the graph.
> -   Pruning Algorithm for models using other than Mean Field approximation functions for Variational Inference.
> -   Unit-Test for the functionalities.


Cite:
-----

- Bibtex Format(Arxiv):
   ```
    @article{sharma2020bayesian,
    title={Bayesian Neural Networks at Scale: A Performance Analysis and Pruning Study},
    author={Sharma, Himanshu and Jennings, Elise},
    journal={arXiv preprint arXiv:2005.11619},
    year={2020}}
   ```
- MLA Format (Arxiv):
  ```
    Sharma, Himanshu, and Elise Jennings. "Bayesian Neural Networks at Scale: A Performance Analysis and Pruning Study." arXiv preprint arXiv:2005.11619 (2020).
  ```

Contact:
--------

-   [Himanshu Sharma](https://himscipy.github.io/), himanshu90sharma@gmail.com
-   [Elise Jennings](https://www.ichec.ie/staff/elise-jennings-phd), elise.jennings@ichec.ie



Acknowledgement:
---------------

This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. This research was funded in part and used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. DOE or the United States Government. Declaration of Interests - None.
