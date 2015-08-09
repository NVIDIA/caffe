# Installation

Caffe depends on several third-party libraries.
You need to download and build them.
Each external library is located in a separate
folder in the `extern` directory. Follow
the instructions in the individual `INSTALL.md`
files. 

All build settings are managed using property
sheets. The main settings are contained in
the `caffe.props` file.

This file should be adjusted if you want to
add custom caffe build flags, e.g. USE_CUDNN.

The checked in version uses cuda 7.5. If you
use a different release please do a search
and replace in the project files (*.vcxproj)
of `CUDA 7.5.props` to your desired version.