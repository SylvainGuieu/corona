#!/usr/bin/env python

import os
import sys

#from distutils.core import setup # dependency does not works 
from setuptools import setup

name = "coronapy"
version='0.1.6' # https://www.python.org/dev/peps/pep-0440/
url = "https://github.com/SylvainGuieu/corona"
author='Sylvain Guieu'
author_email='sylvain.guieu@univ-grenoble-alpes.fr'
install_requires = ['pandas', 'matplotlib']
# in setuptools the egg fragment must have the version in it 
# unlike with the pip install git+https://gricad-gitlab.univ-grenoble-alpes.fr/guieus/instru#egg=instru
dependency_links=[]

# any file inside the data relative directory name/data_dir will be include as data file
data_directories = []
script_directories = []
license='Creative Commons Attribution-Noncommercial-Share Alike license'

long_description_content_type="text/markdown"


# Python 3.6 or later needed
if sys.version_info < (3, 0, 0, 'final', 0):
    raise SystemExit('Python 3.0 or later is required!')


## ######################################################
##
##  Try to make this part bellow stand alone so I can copy/past 
##   to other projects 
## 
## ######################################################


rootdir = os.path.abspath(os.path.dirname(__file__))



# Build a list of all project modules
packages = []
py_modules =["coronapy"]

#if os.path.exists(name+".py"):
#    packages.append(name+".py")

# for dirname, dirnames, filenames in os.walk(name):
#         if '__init__.py' in filenames:
#             packages.append(dirname.replace('/', '.'))

#package_dir = {name: name}

# Data files used e.g. in tests
#package_data = {name: [os.path.join(name, 'tests', 'prt.txt')]}

# The current version number - MSI accepts only version X.X.X
#exec(open(os.path.join(name, 'version.py')).read())

# Scripts
scripts = []
for script_dir in script_directories:
    for dirname, dirnames, filenames in os.walk(script_dir):
        for filename in filenames:
            if not filename.endswith('.bat'):
                scripts.append(os.path.join(dirname, filename))

# Provide bat executables in the tarball (always for Win)
# if 'sdist' in sys.argv or os.name in ['ce', 'nt']:
#     for s in scripts[:]:
#         scripts.append(s + '.bat')

# Data_files (e.g. doc) needs (directory, files-in-this-directory) tuples

data_files = []

for data_dir, file_ext in data_directories:
    for dirname, dirnames, filenames in os.walk( os.path.join(data_dir)):
        fileslist = []
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if (not file_ext) or (ext == file_ext):
                fullname = os.path.join(dirname, filename)
                fileslist.append(fullname)
                #print(('share/' + dirname, fileslist))        
        data_files.append(( os.path.join('share' , dirname), fileslist ))

#####

for path in ["README.md", "readme.md", "readme.txt"]:
    try:
        readme = open('README.md').read()
    except Exception as er:
        print("No readme file : ", er)
    else:
        break
else:
    readme = ""

long_description = """
Tools to load and annylise the Corona virus daily report data from the GitHub of Johns Hopkins University Center for Systems Science and Engineering

See home page [home](https://github.com/SylvainGuieu/corona)
"""

setup(
    name=name,
    url=url, 
    version=version,
    author=author,
    author_email=author_email,
    #packages=packages,
    py_modules=py_modules,
    scripts=scripts,
    data_files=data_files,
    license=license,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    install_requires=install_requires, 
    dependency_links=dependency_links,   
)