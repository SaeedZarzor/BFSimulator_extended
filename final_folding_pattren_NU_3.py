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

# reset view to fit data
renderView1.ResetCamera(False)

# reset view to fit data
renderView1.ResetCamera(False)

animationScene1.GoToLast()

# set scalar coloring
ColorBy(output_2_Display, ('POINTS', 'N_cell_desnity'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(rG_cell_desnityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
output_2_Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
output_2_Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'N_cell_desnity'
n_cell_desnityLUT = GetColorTransferFunction('N_cell_desnity')
n_cell_desnityLUT.RGBPoints = [-132.773, 0.231373, 0.298039, 0.752941, 4958.7135, 0.865003, 0.865003, 0.865003, 10050.2, 0.705882, 0.0156863, 0.14902]
n_cell_desnityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'N_cell_desnity'
n_cell_desnityPWF = GetOpacityTransferFunction('N_cell_desnity')
n_cell_desnityPWF.Points = [-132.773, 0.0, 0.5, 0.0, 10050.2, 1.0, 0.5, 0.0]
n_cell_desnityPWF.ScalarRangeInitialized = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
n_cell_desnityLUT.ApplyPreset('PuBu', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
n_cell_desnityLUT.ApplyPreset('Reds', True)

# invert the transfer function
n_cell_desnityLUT.InvertTransferFunction()

# get color legend/bar for n_cell_desnityLUT in view renderView1
n_cell_desnityLUTColorBar = GetScalarBar(n_cell_desnityLUT, renderView1)
n_cell_desnityLUTColorBar.Title = 'N_cell_desnity'
n_cell_desnityLUTColorBar.ComponentTitle = ''

# change scalar bar placement
n_cell_desnityLUTColorBar.WindowLocation = 'Any Location'
n_cell_desnityLUTColorBar.Position = [0.8247484110169492, 0.2902511078286559]
n_cell_desnityLUTColorBar.ScalarBarLength = 0.33

# reset view to fit data
renderView1.ResetCamera(False)

# change scalar bar placement
n_cell_desnityLUTColorBar.Position = [0.8183924788135593, 0.26070901033973415]
n_cell_desnityLUTColorBar.ScalarBarLength = 0.33000000000000007

# Rescale transfer function
n_cell_desnityLUT.RescaleTransferFunction(0.0, 10050.2)

# Rescale transfer function
n_cell_desnityPWF.RescaleTransferFunction(0.0, 10050.2)

# Rescale transfer function
n_cell_desnityLUT.RescaleTransferFunction(0.0, 11000.0)

# Rescale transfer function
n_cell_desnityPWF.RescaleTransferFunction(0.0, 11000.0)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1888, 1354)

# current camera placement for renderView1
renderView1.CameraPosition = [8.755819272205242, 6.687727985942068, 8.871889323648897]
renderView1.CameraFocalPoint = [0.0009799999999999809, 1.1789216949999999, 0.0]
renderView1.CameraViewUp = [-0.29969534558592636, 0.9144240393984363, -0.2720503188830686]
renderView1.CameraParallelScale = 3.527016855639022

# save screenshot
SaveScreenshot(str(path_folder)+'/NU_folds_pattern.png', renderView1, ImageResolution=[2000, 1354],
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
renderView1.CameraPosition = [8.755819272205242, 6.687727985942068, 8.871889323648897]
renderView1.CameraFocalPoint = [0.0009799999999999809, 1.1789216949999999, 0.0]
renderView1.CameraViewUp = [-0.29969534558592636, 0.9144240393984363, -0.2720503188830686]
renderView1.CameraParallelScale = 3.527016855639022

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
