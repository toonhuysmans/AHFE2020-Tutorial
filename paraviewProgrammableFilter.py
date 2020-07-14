from vtk.util import numpy_support 
pdi = self.GetPolyDataInput()
pdo = self.GetPolyDataOutput() 
pdo.DeepCopy(pdi)
meanPts = mean( [numpy_support.vtk_to_numpy(pd.Points) for pd in inputs], 0)
pdo.GetPoints().SetData(dsa.numpyTovtkDataArray(meanPts , "Points"))
