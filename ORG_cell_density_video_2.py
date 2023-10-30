# trace generated using paraview version 5.10.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
from pathlib import Path
import os
import re

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

directory_folder = "Folder_Output"
parent_dir_folder = os.getcwd()
path_folder = os.path.join(parent_dir_folder, directory_folder)
Files = Path(path_folder)
# Iterate directory
res=[]
a=0.0
for f in Files.iterdir():
    for file in Files.iterdir():
        if re.match("Output_3_\d+\.vtk", file.name):
            Output, degree, number=file.stem.split('_')
            if(a == float(number)):
                res.append(str(file))
                a+=1
        
        
# create a new 'Legacy VTK Reader'
output_3_ = LegacyVTKReader(registrationName='Output_3_*', FileNames= res)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
output_3_Display = Show(output_3_, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'RG_cell_desnity'
rG_cell_desnityLUT = GetColorTransferFunction('RG_cell_desnity')
rG_cell_desnityLUT.RGBPoints = [-0.0842647, 0.231373, 0.298039, 0.752941, 53.136367650000004, 0.865003, 0.865003, 0.865003, 106.357, 0.705882, 0.0156863, 0.14902]
rG_cell_desnityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'RG_cell_desnity'
rG_cell_desnityPWF = GetOpacityTransferFunction('RG_cell_desnity')
rG_cell_desnityPWF.Points = [-0.0842647, 0.0, 0.5, 0.0, 106.357, 1.0, 0.5, 0.0]
rG_cell_desnityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
output_3_Display.Representation = 'Surface'
output_3_Display.ColorArrayName = ['POINTS', 'RG_cell_desnity']
output_3_Display.LookupTable = rG_cell_desnityLUT
output_3_Display.SelectTCoordArray = 'None'
output_3_Display.SelectNormalArray = 'None'
output_3_Display.SelectTangentArray = 'None'
output_3_Display.OSPRayScaleArray = 'RG_cell_desnity'
output_3_Display.OSPRayScaleFunction = 'PiecewiseFunction'
output_3_Display.SelectOrientationVectors = 'displacement'
output_3_Display.ScaleFactor = 0.20035666100000002
output_3_Display.SelectScaleArray = 'cell_desnity'
output_3_Display.GlyphType = 'Arrow'
output_3_Display.GlyphTableIndexArray = 'cell_desnity'
output_3_Display.GaussianRadius = 0.010017833050000001
output_3_Display.SetScaleArray = ['POINTS', 'RG_cell_desnity']
output_3_Display.ScaleTransferFunction = 'PiecewiseFunction'
output_3_Display.OpacityArray = ['POINTS', 'RG_cell_desnity']
output_3_Display.OpacityTransferFunction = 'PiecewiseFunction'
output_3_Display.DataAxesGrid = 'GridAxesRepresentation'
output_3_Display.PolarAxes = 'PolarAxesRepresentation'
output_3_Display.ScalarOpacityFunction = rG_cell_desnityPWF
output_3_Display.ScalarOpacityUnitDistance = 0.3538057902458452
output_3_Display.OpacityArrayName = ['POINTS', 'RG_cell_desnity']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
output_3_Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
output_3_Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

#changing interaction mode based on data extents
renderView1.CameraPosition = [-1.000356661, 0.998216695, 10000.0]

# show color bar/color legend
output_3_Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(output_3_Display, ('POINTS', 'ORG_cell_desnity', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(rG_cell_desnityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
output_3_Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
output_3_Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'ORG_cell_desnity'
oRG_cell_desnityLUT = GetColorTransferFunction('ORG_cell_desnity')
oRG_cell_desnityLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 5.878906683738906e-39, 0.865003, 0.865003, 0.865003, 1.1757813367477812e-38, 0.705882, 0.0156863, 0.14902]
oRG_cell_desnityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'ORG_cell_desnity'
oRG_cell_desnityPWF = GetOpacityTransferFunction('ORG_cell_desnity')
oRG_cell_desnityPWF.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
oRG_cell_desnityPWF.ScalarRangeInitialized = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
oRG_cell_desnityLUT.ApplyPreset('Greens', True)

# invert the transfer function
oRG_cell_desnityLUT.InvertTransferFunction()

# Rescale transfer function
oRG_cell_desnityLUT.RescaleTransferFunction(30, 150)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(2000, 1354)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-1.000356661, 0.998216695, 10000.0]
renderView1.CameraFocalPoint = [-1.000356661, 0.998216695, 0.0]
renderView1.CameraParallelScale = 1.4152231609833807

# save animation
SaveAnimation(str(path_folder)+'/ORG_Cell_desnity.avi', renderView1, ImageResolution=[2000, 1352],
    OverrideColorPalette='WhiteBackground',
    FrameRate=10,
    FrameWindow=[0, len(res)])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(2000, 1354)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-1.000356661, 0.998216695, 10000.0]
renderView1.CameraFocalPoint = [-1.000356661, 0.998216695, 0.0]
renderView1.CameraParallelScale = 1.4152231609833807

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
