These are versions of the Standalone Channel Shifting Toolbox (SCS Toolbox) modules, designed to work as Python Processing Scripts in QGIS (tested using version 3.40.11-Bratislava).
The modules have been ported to QGIS and further modified by Martin Dawson using ChatGPT v.5.1

Martin Dawson
Department of Geography and Environmental Science,
Queen Mary University, 
Mile End Road
London E1 4NS
martin.dawson@qmul.ac.uk

Please reference Rusnak et al (2025) when using this code.

# Author: Milos Rusnak
#   CNRS - UMR5600 Environnement Ville Societe
#   15 Parvis Rene Descartes, BP 7000, 69342 Lyon Cedex 07, France 
               
# geogmilo@savba.sk
#   Institute of geography SAS
#   Stefanikova 49, 814 73 Bratislava, Slovakia 
               
# Standalone channel shifting toolbox (SCS Toolbox) was developed as extension of the FluvialCorridor toolbox with implemented the centerline 
# extraction approach and segmentation of DGO from FluvialCorridor toolbox.
# For each use of the Channel toolbox leading to a publication, report, presentation or any other
# document, please refer also to the following articles:
#       Rusnák, M., Opravil, Š., Dunesme, S., Afzali, H., Rey, L., Parmentier, H., Piégay, H., 2025 A channel shifting GIS toolbox for exploring
#       floodplain dynamics through channel erosion and deposition. Geomorphology, 477, 109688. 
#       https://doi.org/10.1016/j.geomorph.2025.109688 

#       Roux, C., Alber, A., Bertrand, M., Vaudor, L., Piegay, H., 2015. "FluvialCorridor": A new ArcGIS 
#       package for multiscale riverscape exploration. Geomorphology, 242, 29-37.
#       https://doi.org/10.1016/j.geomorph.2014.04.018
#
Please refer to the SCS Differences Documentation for details of differences between the ARCPro version of the code and this QGIS version, particularly module 4a that includes a different treatment of the channel 
and flood plain assemblage.
