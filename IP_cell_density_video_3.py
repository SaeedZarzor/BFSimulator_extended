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
rG_cell_desnityLUT.RGBPoints = [-4.3442, 0.231373, 0.298039, 0.752941, 473.0234, 0.865003, 0.865003, 0.865003, 950.391, 0.705882, 0.0156863, 0.14902]
rG_cell_desnityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'RG_cell_desnity'
rG_cell_desnityPWF = GetOpacityTransferFunction('RG_cell_desnity')
rG_cell_desnityPWF.Points = [-4.3442, 0.0, 0.5, 0.0, 950.391, 1.0, 0.5, 0.0]
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
output_2_Display.SelectScaleArray = 'RG_cell_desnity'
output_2_Display.GlyphType = 'Arrow'
output_2_Display.GlyphTableIndexArray = 'RG_cell_desnity'
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
output_2_Display.ScaleTransferFunction.Points = [-4.3442, 0.0, 0.5, 0.0, 950.391, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
output_2_Display.OpacityTransferFunction.Points = [-4.3442, 0.0, 0.5, 0.0, 950.391, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# show color bar/color legend
output_2_Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()


# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=output_3_)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'RG_cell_desnity']
clip1.Value = 473.0234

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.0, 0.998216695, 0.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [0.0, 0.998216695, 0.0]

# show data in view
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'RG_cell_desnity']
clip1Display.LookupTable = rG_cell_desnityLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleArray = 'RG_cell_desnity'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'displacement'
clip1Display.ScaleFactor = 0.4
clip1Display.SelectScaleArray = 'RG_cell_desnity'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'RG_cell_desnity'
clip1Display.GaussianRadius = 0.02
clip1Display.SetScaleArray = ['POINTS', 'RG_cell_desnity']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'RG_cell_desnity']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = rG_cell_desnityPWF
clip1Display.ScalarOpacityUnitDistance = 0.42752329812227563
clip1Display.OpacityArrayName = ['POINTS', 'RG_cell_desnity']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [-4.3442, 0.0, 0.5, 0.0, 950.391, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [-4.3442, 0.0, 0.5, 0.0, 950.391, 1.0, 0.5, 0.0]

# hide data in view
Hide(output_3_, renderView1)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1.ClipType)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
rG_cell_desnityLUT.ApplyPreset('Blues', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
rG_cell_desnityLUT.ApplyPreset('Blues', True)

# invert the transfer function
rG_cell_desnityLUT.InvertTransferFunction()

# set scalar coloring
ColorBy(clip1Display, ('POINTS', 'IP_cell_desnity'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(rG_cell_desnityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
clip1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'IP_cell_desnity'
iP_cell_desnityLUT = GetColorTransferFunction('IP_cell_desnity')
iP_cell_desnityLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 5.878906683738906e-39, 0.865003, 0.865003, 0.865003, 1.1757813367477812e-38, 0.705882, 0.0156863, 0.14902]
iP_cell_desnityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'IP_cell_desnity'
iP_cell_desnityPWF = GetOpacityTransferFunction('IP_cell_desnity')
iP_cell_desnityPWF.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
iP_cell_desnityPWF.ScalarRangeInitialized = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
iP_cell_desnityLUT.ApplyPreset('Oranges', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
iP_cell_desnityLUT.ApplyPreset('Oranges', True)

# invert the transfer function
iP_cell_desnityLUT.InvertTransferFunction()

# get color legend/bar for iP_cell_desnityLUT in view renderView1
iP_cell_desnityLUTColorBar = GetScalarBar(iP_cell_desnityLUT, renderView1)
iP_cell_desnityLUTColorBar.Title = 'IP_cell_desnity'
iP_cell_desnityLUTColorBar.ComponentTitle = ''

# change scalar bar placement
iP_cell_desnityLUTColorBar.WindowLocation = 'Any Location'
iP_cell_desnityLUTColorBar.Position = [0.8036997126436782, 0.3197932053175776]
iP_cell_desnityLUTColorBar.ScalarBarLength = 0.3300000000000002

# change scalar bar placement
iP_cell_desnityLUTColorBar.Position = [0.7931633141762453, 0.31536189069423937]

animationScene1.GoToLast()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# Rescale transfer function
iP_cell_desnityLUT.RescaleTransferFunction(20.0, 300.0)

# Rescale transfer function
iP_cell_desnityPWF.RescaleTransferFunction(20.0, 300.0)

animationScene1.Play()

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(2088, 1354)

# current camera placement for renderView1
renderView1.CameraPosition = [10.433614262909549, 2.720288801910666, 4.751976504269732]
renderView1.CameraFocalPoint = [-4.721227602241471e-16, 0.9982166950000002, 4.3116233810424395e-16]
renderView1.CameraViewUp = [-0.1296365940161464, 0.9888185409343645, -0.0737037759977551]
renderView1.CameraParallelScale = 3.0005949060439203

animationScene1.GoToFirst()

# layout/tab size in pixels
layout1.SetSize(2088, 1354)

# current camera placement for renderView1
renderView1.CameraPosition = [10.433614262909549, 2.720288801910666, 4.751976504269732]
renderView1.CameraFocalPoint = [-4.721227602241471e-16, 0.9982166950000002, 4.3116233810424395e-16]
renderView1.CameraViewUp = [-0.1296365940161464, 0.9888185409343645, -0.0737037759977551]
renderView1.CameraParallelScale = 3.0005949060439203

# save animation
SaveAnimation(str(path_folder)+'/IP_Cell_desnity.avi', renderView1, ImageResolution=[2088, 1352],
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
layout1.SetSize(2088, 1354)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [10.433614262909549, 2.720288801910666, 4.751976504269732]
renderView1.CameraFocalPoint = [-4.721227602241471e-16, 0.9982166950000002, 4.3116233810424395e-16]
renderView1.CameraViewUp = [-0.1296365940161464, 0.9888185409343645, -0.0737037759977551]
renderView1.CameraParallelScale = 3.0005949060439203

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
