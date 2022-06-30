from MIAS_2_Folders import ALLpgm
from MIAS_2_Folders import ALLpng
from MIAS_2_Folders import ALLtiff

from MIAS_4_MIAS_Functions import changeExtension

pgm = '.pgm'
png = '.png'
tiff = '.tiff'

PGMtoPNG = changeExtension(folder = ALLpgm, newfolder = ALLpng, extension = pgm, newextension = png)
PGMtoTIFF = changeExtension(folder = ALLpgm, newfolder = ALLtiff, extension = pgm, newextension = tiff)

PGMtoPNG.ChangeExtension()
PGMtoTIFF.ChangeExtension()