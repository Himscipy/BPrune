import numpy as np
import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import figure 
from matplotlib.backends import backend_agg
import matplotlib.pyplot as plt


try:
  import seaborn as sns 
  sns.set_context('paper')
  sns.set(color_codes=True)
  sns.set_style("whitegrid")
  paper_rc = {'lines.linewidth': 2.0, 'lines.markersize': 10} 
  sns.set_context("paper", font_scale=2.5 ,rc=paper_rc)
  HAS_SEABORN = True
  # print(HAS_SEABORN)
except ImportError:
  HAS_SEABORN = False



class Plot_Viz:
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS


    def Plot_dist(self,Ratio_mean_STD,layer):
        fig = figure.Figure(figsize=(20 , 20))
        canvas = backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        sns.distplot(Ratio_mean_STD,label='Actual',hist=False, color="g", kde_kws={"shade": True},ax=ax)
        ax.set_xlabel('SNR = |mu|/s')
        ax.set_ylabel('Density')
        ax.legend()
        fig.tight_layout()
        fname = os.path.join(self.FLAGS.data_dir, (layer[0].split('/')[-2]+"_SNR_ratioPlot.png") )
        canvas.print_figure(fname, format="png")
        return

    def Plot_mean_std(mean,std,layer):
        fig = figure.Figure(figsize=(20 , 20))
        canvas = backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(121)
        sns.distplot(mean,label='mean',hist=False, color="g", kde_kws={"shade": True},ax=ax)
        ax1 = fig.add_subplot(122)
        sns.distplot(std,label='std',hist=False, color="g", kde_kws={"shade": True},ax=ax1)
        
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Density')
        ax.legend()
        fig.tight_layout()
        fname = os.path.join(self.FLAGS.data_dir, (layer[0].split('/')[-2]+"_Mean_STD_Plot.png") )
        canvas.print_figure(fname, format="png")
        return
