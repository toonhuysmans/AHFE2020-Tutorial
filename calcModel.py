#! /usr/bin/env python3

__author__    = "Toon Huysmans"
__copyright__ = "Copyright (C) 2018 TU Delft"
__license__   = "proprietary"
__version__   = "1.0" 

import argparse

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

import scipy
import scipy.linalg
import scipy.linalg.blas as blas

from progressbar import ProgressBar, Percentage, Bar, Counter

# command line input
parser = argparse.ArgumentParser()
parser.add_argument('-s',dest='surfaceFilenames', type=str, nargs='+',
                    help='the vtk surface files')
parser.add_argument('-o',dest='outputFilename', type=str, default="model.vtk",
                    help='the output vtk surface filename')
parser.add_argument('-t',dest='transform', action="store_true",
                    help='flag to enable procrustus aligment before modeling')
args = parser.parse_args()

print("Reading files...")

pbar = ProgressBar(widgets=[Percentage(), Bar()])

surfaces = []
for sFilename in pbar(args.surfaceFilenames):    

    # reading surface
    sReader = vtk.vtkPolyDataReader()
    sReader.SetFileName( sFilename )
    sReader.Update()
    
    surface = sReader.GetOutput()
    pd = vtk.vtkPolyData()
    pd.DeepCopy(surface)
    surfaces.append( pd )

# if requested do alignment
nrOfSurfaces = len(surfaces)
if args.transform == True:

    print("Procrustes Alignment...")

    group = vtk.vtkMultiBlockDataGroupFilter()
    for surface in surfaces:  
        group.AddInputData(surface)  
    
    align = vtk.vtkProcrustesAlignmentFilter()
    align.SetInputConnection(group.GetOutputPort())
    align.GetLandmarkTransform().SetModeToRigidBody()
    align.Update()

    surfaces = []
    for i in range(nrOfSurfaces):  
        pd = vtk.vtkPolyData()
        pd.DeepCopy(align.GetOutput().GetBlock(i))
        surfaces.append( pd )

print("Creating landmark matrix...")

# gather all points for all surfaces
allPoints = []
for surface in surfaces:        
    surfaceDSA = dsa.WrapDataObject( surface )    
    allPoints.append( surfaceDSA.Points )

# get connectivity for model
reader = vtk.vtkPolyDataReader()
reader.SetFileName( args.surfaceFilenames[0] )
reader.Update()

modelSurface = reader.GetOutput()
modelSurfaceDSA = dsa.WrapDataObject( modelSurface )
modelSurfaceDSA.Points = scipy.mean( allPoints , axis=0)

# calculate and add Mean_To_Object
meanPoints = modelSurfaceDSA.Points
#for i,points in enumerate(allPoints):
#    modelSurfaceDSA.PointData.append( points - meanPoints , "Mean_To_Object_" + str(i+1).rjust(int(scipy.log10(nrOfSurfaces)+1),"0") )

# calculate Model_Eigenmode
L = scipy.array([ points.flatten() for points in allPoints ])
LN = L - L.mean(0)
#C = blas.ssyrk( alpha = 1.0 / float(nrOfSurfaces-1) , a = LN )

print("Calculating covariance matrix...")

C = LN.dot(LN.T) * ( 1.0 / float(nrOfSurfaces-1))

print("SVD...")

(U,S,V) = scipy.linalg.svd(C)
E =  LN.T.dot(U).T
for i in range(len(E)):
    E[i,:] /= scipy.linalg.norm(E[i,:]) 

B = LN.dot(E.T)
Btruncated = B[:,0:6]

print("Calculating shape modes...")

Es = scipy.diag(scipy.sqrt(S) * 3.0).dot(E) 

# add Model_Eigenmode
for i,PC in enumerate(Es):
    modelSurfaceDSA.PointData.append( PC.reshape((-1,3)) , "Model_Eigenmode_" + str(i+1).rjust(int(scipy.log10(nrOfSurfaces)+1),"0") )
    if i>25: break

# add mean distance
meanDistance = scipy.sqrt(scipy.square(scipy.array(allPoints) - meanPoints).sum(axis=2)).mean(axis=0)
modelSurfaceDSA.PointData.append( meanDistance , "Mean_DistanceToMean")

# add tensor
apa = scipy.array(allPoints) - meanPoints
covarianceTensors = scipy.array([ blas.ssyrk( alpha = 1.0 / float(nrOfSurfaces-1) , a = apa[:,i,:].T ) for i in range( apa.shape[1] ) ])
modelSurfaceDSA.PointData.append( covarianceTensors , "CovarianceTensor")
#modelSurfaceDSA.PointData.append( covarianceTensors / scipy.array([scipy.linalg.norm(c) for c in covarianceTensors ]) , "CovarianceTensor_normalized")

print("Writing model...")

# save first and second PC
#scipy.savetxt(args.outputFilename.replace(".vtk","_PC1and2.txt") , Btruncated, delimiter="\t")

# save model
writer = vtk.vtkPolyDataWriter()
writer.SetInputData( modelSurface )
writer.SetFileName( args.outputFilename )
# writer.SetFileTypeToBinary()
writer.Update()

print("Done.")

