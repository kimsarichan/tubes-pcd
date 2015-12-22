-------------------------------------------------------------------------------
                                Image Analyzer
-------------------------------------------------------------------------------
        Copyright (C) 2000-2004  |  Michael Vinther  |  mv@logicnet·dk
                        MeeSoft  |  http://meesoft.logicnet.dk
-------------------------------------------------------------------------------

Description:
------------
  Advanced image editing, enhancement and analysis software.


Some of the features:
---------------------
  · Read/write BMP, ICO, CUR, WMF, EMF, JPEG, PNG, GIF, PCX and JPEG 2000 files
  · Read RAS, PNM, PGM, PPM, HIPS and Matlab files
  · Scanner and printer support
  · Build-in conventional and adaptive filters
  · User specified filters in spatial and frequency domain
  · Distance, Fourier and discrete cosine transformation
  · Restoration by (blind) deconvolution
  · Color model conversion: RGB, CMY, HSI, Lab, PCA, YCbCr and YIQ
  · Easy red-eye removal
  · Morphological operations
  · Resize, rotate, crop and warping
  · Math expression module for creating and transforming images
  · Advanced "pocket" calculator with equation solver
  · Plugin system for image processing and extended file format support


Short description of MAP file format:
-------------------------------------
  File header:
    MAP           : 3 bytes     The characters 'MAP'
    Version       : DWord       1
    PixelFormat   : DWord       $01  : Byte data
                                $02  : Word data
                                $40  : Single floating point data (32 bit)
                                $80  : Double floating point data (64 bit)
    Width, Height : DWord       Image size
    Planes        : Byte        1    : Real values
                                2    : Complex values (real,imaginary)
    Compression   : Byte        0    : None, raw data
                                1    : Deflate compressed (default)
                                2    : Another variant of LZ77

  Following is the data compressed as specified by the Compression parameter.
  For complex data the real plane is stored first and then the complex plane.
  The data is stored line by line, left to right from top to bottom.
  The present version of Image Analyzer always uses the deflate compression
  for writing files. Description of the Deflate standard can be found at
  http://www.gzip.org/zlib/zlib_docs.html


License information:
--------------------
Conditions for using and copying the software:
  · I (Michael Vinther) or MeeSoft can not in any way be held responsible for
    any damage or loss caused by using this software.
  · The software can only be copied/distributed in its original form (installer
    or zip archive) unless otherwise agreed with the author.
  · Distribution of this software for commercial purposes is not allowed
    without consent from the author.

Thanks to Markus Oberhumer and Laszlo Molnar for their brilliant EXE compressor
UPX. (See http://upx.sourceforge.net/)
This software includes a library created by Anders Melander.
The JasPerLib DLL which contains the JPEG 2000 stuff is (c) 1999-2000, Image
Power, Inc. and the University of British Columbia, Canada, see libjasper.txt.
Most of the filter and palette files was provided by Marian Stach.


Version history:
----------------
Note that this version history is incomplete. It does not include all changes
in all versions.

Changes in version 1.01  (2001-03-15):
  PNG transparency supported
Changes in version 1.02  (2001-03-18):
  PNG transparency bug corrected
  More keyboard shortcuts
  Support for loading multiple files
Changes in version 1.03  (2001-04-23):
  Auto window resizing
  Statistics map
  JPEG filter
Changes in version 1.04  (2001-05-15):
  Adaptive noise filter
  Auto color correction
  More keyboard shortcuts
Changes in version 1.06  (2001-08-16):
  Adaptive noise filter improved
  JPEG 2000 support
Changes in version 1.07  (2001-09-03):
  Improved JPEG 2000 support
  Expression evaluator in all edit boxes
  Improved color mapper
  Improved combine images
Changes in version 1.07  (2001-09-23):
  Median filter optimized
Changes in version 1.08  (2002-01-16):
  Lapp filter now preserve colors
  Windows ICO and CUR file support
  Frequency domain filter
  Selective smooth filter
  Rotate grayscale bug fixed
Changes in version 1.09  (2002-07-01):
  Auto adjust black/white point
  Fixed bug in combine grayscale images
  Fixed bug in resize
  Improved printing
Changes in version 1.10  (2003-01-01):
  Image warping
Changes in version 1.11  (2003-02-13):
  JPEG 2000 support improved
  Reading of Sun Rasterfiles (*.ras)
  Image resizing using spline interpolation for better quality
  PNG gamma, transparency, and alpha channel handling improved
  Expression evaluator improved
  Lab color space support
  Morphological operations on color images
Changes in version 1.12  (2003-03-08):
  Restoration by deconvolution (see example page)
  JPEG desired file size specification
  Improved help
  Filtering of float maps
  Color mapper can now work on luminance plane (Y in YCbCr)
Changes in version 1.13  (2003-03-31):
  Open, save and color mapper dialogs improved
  The amount of smoothing can be specified in "Smooth image"
  Highpass filter in "Edge detection" menu
Changes in version 1.14  (2003-04-17):
  Three different methods for automatic color correction
  Feature list in help menu
  Some bugfixes
  Texture synthesis (see example page)
  Mouse selection editing
  Added an installer wizard version
  Added help/tutorial file, in CHM format.
Changes in version 1.15  (2003-05-24):
  Improved PCA and reconstruction from components
  Paste to selection in insert menu
  Lossless PNG compression ratio improved by about 20%
  Lossy PNG compression introduced
  Writing of large JPEG files (>1 MB) fixed
  Test button in file format options for lossy JPEG and PNG
  Deconvolution filter can be specified in a Matlab, text or MAP file
  Open and save dialogs updated
  EMF/WMF reading fixed
  Texture synthesis improved
Changes in version 1.17  (2003-06-22):
  Image processing plugin system
  Texture synthesis by averaging
  Copy/paste of floating point maps (can be used as simple undo feature)
  Pixel drawing feature
Changes in version 1.17.3  (2003-07-27):
  Frequency domain filter improved
  Better image resize quality
  Paste to selection improved
  24 to 8 bit conversion improved
  Local histogram equalization quality improved
  Option to register image file formats
  More advanced sharpening method
Changes in version 1.18  (2003-09-13):
  Blind deconvolution
  16 bit grayscale PNGs and TIFFs now read to floating point map
  Split planes color correction improved
  Filtering of floating point map with filter from file
  Frequency domain filter improved + bugfix
  Expression evaluator improved
  More than 60 new filters (click Load in the User defined filter dialog)
  More than 20 new palettes
Changes in version 1.19  (2003-12-29):
  Problem with large JPEG 2000 images and temporary files fixed
  Expression evaluator precision and plot axis autoscaling improved
  Improved de-interlacing method
  MNG image file support
  Fixed mouse selection bug
  Language module
Changes in version 1.20  (2004-02-16):
  Minor bugfixes
  F8 now changes between mouse zoom and mouse select
  Improved deconvolution kernel estimation
  Color mapper dialog updated
Changes in version 1.21  (2004-05-31):
  New high quality image enlargement method (super-resolution by Wiener filter)
  Better handling of large fonts (120 DPI) screen setting
  Rectangle selection can be locked to specific aspect ratio
