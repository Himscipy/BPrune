try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'Readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'bprune',         
  packages = find_packages(),   
  version = '0.3',      
  license='MIT',
  package_data={'bprune': ['src/*.py', 'test/*.py'] },
  include_package_data=True ,       
  description = 'Bayesiean Neural Network Pruning Library',   # Give a short description about your library
  long_description=long_description,
  long_description_content_typ='text/markdown'
  author = 'Himanshu Sharma',                   # Type in your name
  author_email = 'himanshu90sharma@gmail.com',      # Type in your E-Mail
  url = '',   # Provide either the link to your github or to your website
  download_url = '',    # I explain this later on
  keywords = ['Neural Network', 'TensorFlow Probability', 'Bayesian'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
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