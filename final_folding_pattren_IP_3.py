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
output_2_ = LegacyVTKReader(registrationName='Output_3_*', FileNames= res)

animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
output_2_Display = Show(output_2_, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'RG_cell_desnity'
rG_cell_desnityLUT = GetColorTransferFunction('RG_cell_desnity')
rG_cell_desnityLUT.AutomaticRescaleRangeMode = 'Never'
rG_cell_desnityLUT.RGBPoints = [0.0, 0.968627, 0.984314, 1.0, 52.94114999999992, 0.922491, 0.954787, 0.985236, 109.41164999999997, 0.87328, 0.923291, 0.969489, 165.88215000000002, 0.825928, 0.891795, 0.953741, 222.35309999999993, 0.778685, 0.8603, 0.937993, 278.8236, 0.701423, 0.826928, 0.910988, 335.2941000000001, 0.622684, 0.793464, 0.883429, 391.7646, 0.523137, 0.739193, 0.861546, 448.2352935, 0.422745, 0.684075, 0.839892, 504.70605, 0.341423, 0.628958, 0.808704, 561.17655, 0.260715, 0.573841, 0.777209, 617.6470500000001, 0.195386, 0.509112, 0.743791, 674.1175499999999, 0.130427, 0.444152, 0.710327, 730.58805, 0.080969, 0.38113, 0.661361, 787.059, 0.031757, 0.318139, 0.612149, 843.5295, 0.031373, 0.253195, 0.516063, 900.0, 0.031373, 0.188235, 0.419608]
rG_cell_desnityLUT.ColorSpace = 'Lab'
rG_cell_desnityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'RG_cell_desnity'
rG_cell_desnityPWF = GetOpacityTransferFunction('RG_cell_desnity')
rG_cell_desnityPWF.Points = [0.0, 0.0, 0.5, 0.0, 900.0, 1.0, 0.5, 0.0]
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

animationScene1.GoToLast()

# reset view to fit data
renderView1.ResetCamera(False)

# get color legend/bar for rG_cell_desnityLUT in view renderView1
rG_cell_desnityLUTColorBar = GetScalarBar(rG_cell_desnityLUT, renderView1)
rG_cell_desnityLUTColorBar.WindowLocation = 'Any Location'
rG_cell_desnityLUTColorBar.Position = [0.8187199252153126, 0.2976366322008862]
rG_cell_desnityLUTColorBar.Title = 'RG_cell_desnity'
rG_cell_desnityLUTColorBar.ComponentTitle = ''
rG_cell_desnityLUTColorBar.ScalarBarLength = 0.32999999999999996

# change scalar bar placement
rG_cell_desnityLUTColorBar.Position = [0.834609755723787, 0.2991137370753323]

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=output_2_)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'RG_cell_desnity']
clip1.Value = 531.4736

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.0009799999999999809, 1.1789216949999999, 0.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [0.0009799999999999809, 1.1789216949999999, 0.0]

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
clip1Display.ScaleFactor = 0.4698744969529077
clip1Display.SelectScaleArray = 'RG_cell_desnity'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'RG_cell_desnity'
clip1Display.GaussianRadius = 0.023493724847645385
clip1Display.SetScaleArray = ['POINTS', 'RG_cell_desnity']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'RG_cell_desnity']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = rG_cell_desnityPWF
clip1Display.ScalarOpacityUnitDistance = 0.4956932954107853
clip1Display.OpacityArrayName = ['POINTS', 'RG_cell_desnity']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [-61.5828, 0.0, 0.5, 0.0, 1124.51, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [-61.5828, 0.0, 0.5, 0.0, 1124.51, 1.0, 0.5, 0.0]

# hide data in view
Hide(output_2_, renderView1)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1.ClipType)

# set active source
SetActiveSource(output_2_)

# create a new 'Clip'
clip2 = Clip(registrationName='Clip2', Input=output_2_)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['POINTS', 'RG_cell_desnity']
clip2.Value = 531.4736

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.0009799999999999809, 1.1789216949999999, 0.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip2.HyperTreeGridClipper.Origin = [0.0009799999999999809, 1.1789216949999999, 0.0]

# Properties modified on clip2.ClipType
clip2.ClipType.Normal = [0.0, 0.0, 1.0]

# show data in view
clip2Display = Show(clip2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['POINTS', 'RG_cell_desnity']
clip2Display.LookupTable = rG_cell_desnityLUT
clip2Display.SelectTCoordArray = 'None'
clip2Display.SelectNormalArray = 'None'
clip2Display.SelectTangentArray = 'None'
clip2Display.OSPRayScaleArray = 'RG_cell_desnity'
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'displacement'
clip2Display.ScaleFactor = 0.4698979991616128
clip2Display.SelectScaleArray = 'RG_cell_desnity'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'RG_cell_desnity'
clip2Display.GaussianRadius = 0.02349489995808064
clip2Display.SetScaleArray = ['POINTS', 'RG_cell_desnity']
clip2Display.ScaleTransferFunction = 'PiecewiseFunction'
clip2Display.OpacityArray = ['POINTS', 'RG_cell_desnity']
clip2Display.OpacityTransferFunction = 'PiecewiseFunction'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityFunction = rG_cell_desnityPWF
clip2Display.ScalarOpacityUnitDistance = 0.4910723117431861
clip2Display.OpacityArrayName = ['POINTS', 'RG_cell_desnity']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip2Display.ScaleTransferFunction.Points = [-61.58279711130222, 0.0, 0.5, 0.0, 1124.53, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip2Display.OpacityTransferFunction.Points = [-61.58279711130222, 0.0, 0.5, 0.0, 1124.53, 1.0, 0.5, 0.0]

# hide data in view
Hide(output_2_, renderView1)

# show color bar/color legend
clip2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip2.ClipType)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
rG_cell_desnityLUT.ApplyPreset('Blues', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
rG_cell_desnityLUT.ApplyPreset('Blues', True)

# invert the transfer function
rG_cell_desnityLUT.InvertTransferFunction()

# set scalar coloring
ColorBy(clip1Display, ('POINTS', 'IP_cell_desnity'))
ColorBy(clip2Display, ('POINTS', 'IP_cell_desnity'))


# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(rG_cell_desnityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
clip1Display.RescaleTransferFunctionToDataRange(True, False)
clip2Display.RescaleTransferFunctionToDataRange(True, False)


# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)
clip2Display.SetScalarBarVisibility(renderView1, True)


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

# Rescale transfer function
iP_cell_desnityLUT.RescaleTransferFunction(0.0, 10)

# Rescale transfer function
iP_cell_desnityPWF.RescaleTransferFunction(0, 10)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1888, 1354)

# current camera placement for renderView1
renderView1.CameraPosition = [8.378611772997756, 5.717005683906464, 9.74297989272462]
renderView1.CameraFocalPoint = [0.0009799999999999824, 1.1789216949999999, 0.0]
renderView1.CameraViewUp = [-0.2555928359189305, 0.9416895054388286, -0.21884509949572428]
renderView1.CameraParallelScale = 3.527016855639022

# save screenshot
SaveScreenshot(str(path_folder)+'/IP_folds_pattern.png', renderView1, ImageResolution=[2000, 1354],
    OverrideColorPalette='WhiteBackground')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1888, 1354)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [8.378611772997756, 5.717005683906464, 9.74297989272462]
renderView1.CameraFocalPoint = [0.0009799999999999824, 1.1789216949999999, 0.0]
renderView1.CameraViewUp = [-0.2555928359189305, 0.9416895054388286, -0.21884509949572428]
renderView1.CameraParallelScale = 3.527016855639022

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
