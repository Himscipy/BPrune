# import numpy as np
# from bprune.src.utils import *
# from bprune.src.Prune_BNN import *
# import ..src.utils as UT
# import ..src.Prune_BNN as BP
import bprune.src.utils as UT
import bprune.src.Prune_BNN as BP

import pytest






def test_parser():
    Init = UT.Initialize()    
    arg = Init.ArgParser()
    # print (arg)
    # Arrange 
    # Act 
    # Assert
    assert (type (arg.b)
    
# def test_ReadLayer(self):
#     assert ()


# def test_ReadOps(self):
#     assert ()


# def test_LoadModel(self):
#     assert ()


# def test_LoadInferVar(self):
#     assert()


# def test_CalcRatioMuSigma(self):
#     assert ()


# def test_GlobalSNR(self):
#     assert ()


# def test_SoftPlus(self):
#     assert()

# def test_CalcHeld_Acc(self):
#     assert()
