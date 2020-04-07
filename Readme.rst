################################
Bayesian Neural Network Pruning
################################
.. image:: Logo_Bprune.jpg
    :width: 200px
    :align: center
    :height: 100px
    :alt: alternate text

Description:  
------------
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Bprune is written to perform inference and pruning of Bayesian Neural Networks(BNN) the 
models developed with `tensorflow <https://www.tensorflow.org/>`_ and `tensorflow probability <https://www.tensorflow.org/probability>`_.
The BNN's supported by the package are one which uses mean field approximation principle of VI i.e uses 
gaussian to define the priors on the weights. Currently, the pruning threshold is based on 
the signal to noise ratio thresholding.  



Installation Instruction:
--------------------------

- Before installtion ensure that you have a working Tensorflow and Tensorflow probability working environment.  
- The installation can be done either after git cloning as follows:

```
python install -r requirement.txt
python install setup.py
```  
- Or can be done using pip as 
```
pip install Bprune
```

How to use Bprune?
------------------
- The example can be found in. 

Limitations/TODO:
-----------------
 - Only support models trained using placeholders.
 - Prunning only for models using Mean Field appoximation for Vatiational Inference. 