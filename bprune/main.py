# import numpy as np
import src.utils as utils

init = utils.Initialize()

flags = init.FlagParser()

print(flags.flag_values_dict())

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
 
