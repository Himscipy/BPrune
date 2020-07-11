import io 
import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


DESCRIPTION = "BPrune is developed to perform inference and pruning of Bayesian Neural Networks(BNN) models developed with tensorflow and tensorflow probability."

here = os.path.abspath(os.path.dirname(__file__))
# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
  name = 'BPrune',         
  packages = find_packages(),   
  version = '0.1.0',      
  license='MIT',
  long_description=long_description,
  long_description_content_type="text/markdown",
  package_data={'bprune': ['src/*.py', 'test/*.py'] },
  include_package_data=True ,       
  description = 'Bayesiean Neural Network Pruning Library',   # Give a short description about your library
  author = 'Himanshu Sharma',                   # Type in your name
  author_email = 'himanshu90sharma@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/Himscipy/BPrune/',   # Provide either the link to your github or to your website
  download_url = '',    # explain this later on
  keywords = ['Neural Network', 'TensorFlow Probability', 'Bayesian Neural Network', 'Deep Learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'markdown',
          'numpy',
          'scipy',
          'pandas',
          'seaborn',
          'tensorflow',
          'tensorflow-probability'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    
  ],
)