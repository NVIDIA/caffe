# Installation

Find out which python version was used to build the boost python library.
For that you can use the dependency walker (http://www.dependencywalker.com/)
and load the boost python dll.

Download and install the corresponding python version from http://continuum.io/downloads .

Locate and copy the relevant files into the following folder structure:

- include: header files from $(PYTHONDIR)\include and header files from numpy package
           $(PYTHONDIR)\Lib\site-packages\numpy\core\include
- lib: library files (python27.lib)
