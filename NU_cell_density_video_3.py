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
output_2_Display = Show(output_3_, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'RG_cell_desnity'
rG_cell_desnityLUT = GetColorTransferFunction('RG_cell_desnity')
rG_cell_desnityLUT.RGBPoints = [-0.0842647, 0.231373, 0.298039, 0.752941, 53.136367650000004, 0.865003, 0.865003, 0.865003, 106.357, 0.705882, 0.0156863, 0.14902]
rG_cell_desnityLUT.ScalarRangeInitialized = 1.0


# get opacity transfer function/opacity map for 'RG_cell_desnity'
rG_cell_desnityPWF = GetOpacityTransferFunction('RG_cell_desnity')
rG_cell_desnityPWF.Points = [-0.0842647, 0.0, 0.5, 0.0, 106.357, 1.0, 0.5, 0.0]
rG_cell_desnityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
output_2_Display.Representation = 'Surface'
output_2_Display.ColorArrayName = ['POINTS', 'RG_cell_desnity']
output_2_Display.LookupTable = rG_cell_desnityLUT
output_2_Display.SelectTCoordArray = 'None'
output_2_Display.SelectNormalArray = 'None'
output_2_Display.SelectTangentArray = 'None'
output_2_Display.OSPRayScaleArray = 'RG_cell_desnity'
output_2_Display.OSPRayScaleFunction = 'PiecewiseFunction'
output_2_Display.SelectOrientationVectors = 'displacement'
output_2_Display.ScaleFactor = 0.4
output_2_Display.SelectScaleArray = 'cell_desnity'
output_2_Display.GlyphType = 'Arrow'
output_2_Display.GlyphTableIndexArray = 'cell_desnity'
output_2_Display.GaussianRadius = 0.02
output_2_Display.SetScaleArray = ['POINTS', 'RG_cell_desnity']
output_2_Display.ScaleTransferFunction = 'PiecewiseFunction'
output_2_Display.OpacityArray = ['POINTS', 'RG_cell_desnity']
output_2_Display.OpacityTransferFunction = 'PiecewiseFunction'
output_2_Display.DataAxesGrid = 'GridAxesRepresentation'
output_2_Display.PolarAxes = 'PolarAxesRepresentation'
output_2_Display.ScalarOpacityFunction = rG_cell_desnityPWF
output_2_Display.ScalarOpacityUnitDistance = 0.43868963652317955
output_2_Display.OpacityArrayName = ['POINTS', 'RG_cell_desnity']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
output_2_Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
output_2_Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# show color bar/color legend
output_2_Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(output_2_Display, ('POINTS', 'N_cell_desnity', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(rG_cell_desnityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
output_2_Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
output_2_Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'N_cell_desnity'
n_cell_desnityLUT = GetColorTransferFunction('N_cell_desnity')
n_cell_desnityLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 5.878906683738906e-39, 0.865003, 0.865003, 0.865003, 1.1757813367477812e-38, 0.705882, 0.0156863, 0.14902]
n_cell_desnityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'N_cell_desnity'
n_cell_desnityPWF = GetOpacityTransferFunction('N_cell_desnity')
n_cell_desnityPWF.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
n_cell_desnityPWF.ScalarRangeInitialized = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
n_cell_desnityLUT.ApplyPreset('Reds', True)

# invert the transfer function
n_cell_desnityLUT.InvertTransferFunction()

# Rescale transfer function
n_cell_desnityLUT.RescaleTransferFunction(0.0, 10000)

animationScene1.GoToLast()

# rescale color and/or opacity maps used to exactly fit the current data range
output_2_Display.RescaleTransferFunctionToDataRange(False, True)

# Rescale transfer function
rG_cell_desnityLUT.RescaleTransferFunction(0.0, 5555.0)

# Rescale transfer function
rG_cell_desnityPWF.RescaleTransferFunction(0.0, 5555.0)

# get color legend/bar for cell_desnityLUT in view renderView1
rG_cell_desnityLUTColorBar = GetScalarBar(rG_cell_desnityLUT, renderView1)

# Properties modified on cell_desnityLUTColorBar
rG_cell_desnityLUTColorBar.TitleColor = [0.06465247577630275, 0.11065842679484245, 0.04505989166094453]
rG_cell_desnityLUTColorBar.LabelColor = [0.06465247577630275, 0.11065842679484245, 0.04505989166094453]

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(2000, 1354)

# current camera placement for renderView1
renderView1.CameraPosition = [0.885942603219279, 8.973266794586076, 8.367843284778974]
renderView1.CameraFocalPoint = [-1.6028965695410085e-17, 0.9982166950000003, -1.6028965695410085e-17]
renderView1.CameraViewUp = [-0.08796048047942562, 0.725721354515419, -0.6823426334871171]
renderView1.CameraParallelScale = 3.0005949060439203

# save animation
SaveAnimation(str(path_folder)+'/NU_Cell_desnity.avi', renderView1, ImageResolution=[2000, 1352],
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
renderView1.CameraPosition = [0.885942603219279, 8.973266794586076, 8.367843284778974]
renderView1.CameraFocalPoint = [-1.6028965695410085e-17, 0.9982166950000003, -1.6028965695410085e-17]
renderView1.CameraViewUp = [-0.08796048047942562, 0.725721354515419, -0.6823426334871171]
renderView1.CameraParallelScale = 3.0005949060439203

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
