import os
import sys
sys.path.insert(0, os.path.abspath('..'))
project = 'PHM Agent'
author = 'PHM Team'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
html_theme = 'furo'
html_static_path = ['_static']
