# CMAKE generated file: DO NOT EDIT!
# Generated by "MSYS Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /C/msys64/clang64/bin/cmake.exe

# The command to remove a file.
RM = /C/msys64/clang64/bin/cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /C/Users/olive/Documents/MCHA4400/lab6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /C/Users/olive/Documents/MCHA4400/lab6/build

# Include any dependencies generated for this target.
include CMakeFiles/lab6.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/lab6.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/lab6.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab6.dir/flags.make

CMakeFiles/lab6.dir/src/main.cpp.obj: CMakeFiles/lab6.dir/flags.make
CMakeFiles/lab6.dir/src/main.cpp.obj: C:/Users/olive/Documents/MCHA4400/lab6/src/main.cpp
CMakeFiles/lab6.dir/src/main.cpp.obj: CMakeFiles/lab6.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/C/Users/olive/Documents/MCHA4400/lab6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lab6.dir/src/main.cpp.obj"
	/C/msys64/clang64/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/lab6.dir/src/main.cpp.obj -MF CMakeFiles/lab6.dir/src/main.cpp.obj.d -o CMakeFiles/lab6.dir/src/main.cpp.obj -c /C/Users/olive/Documents/MCHA4400/lab6/src/main.cpp

CMakeFiles/lab6.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lab6.dir/src/main.cpp.i"
	/C/msys64/clang64/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /C/Users/olive/Documents/MCHA4400/lab6/src/main.cpp > CMakeFiles/lab6.dir/src/main.cpp.i

CMakeFiles/lab6.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lab6.dir/src/main.cpp.s"
	/C/msys64/clang64/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /C/Users/olive/Documents/MCHA4400/lab6/src/main.cpp -o CMakeFiles/lab6.dir/src/main.cpp.s

# Object files for target lab6
lab6_OBJECTS = \
"CMakeFiles/lab6.dir/src/main.cpp.obj"

# External object files for target lab6
lab6_EXTERNAL_OBJECTS =

lab6.exe: CMakeFiles/lab6.dir/src/main.cpp.obj
lab6.exe: CMakeFiles/lab6.dir/build.make
lab6.exe: libcommon.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_gapi.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_stitching.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_alphamat.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_aruco.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_bgsegm.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_ccalib.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_cvv.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_dnn_objdetect.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_dnn_superres.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_dpm.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_face.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_freetype.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_fuzzy.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_hdf.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_hfs.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_img_hash.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_intensity_transform.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_line_descriptor.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_mcc.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_ovis.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_quality.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_rapid.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_reg.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_rgbd.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_saliency.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_sfm.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_stereo.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_structured_light.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_superres.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_surface_matching.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_tracking.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_videostab.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_viz.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_wechat_qrcode.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_xfeatures2d.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_xobjdetect.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_xphoto.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkWrappingTools.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkViewsQt.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkViewsInfovis.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkUtilitiesBenchmarks.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkViewsContext2D.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkzfp.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkTestingRendering.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkTestingIOSQL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkTestingGenericBridge.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkTestingDataModel.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingRayTracing.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingVolumeAMR.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingQt.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkPythonContext2D.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingParallel.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingOpenXR.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingOpenVR.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingVR.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingVolumeOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingMatplotlib.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkPythonInterpreter.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingLabel.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingLOD.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingLICOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingImage.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingFreeTypeFontConfig.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingFFMPEGOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingExternal.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingContextOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOXdmf2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkxdmf2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOVeraOut.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOVPIC.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkvpic.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOTecplotTable.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOTRUCHAS.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOSegY.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOPostgreSQL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOXdmf3.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkxdmf3.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOParallelXML.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOParallelLSDyna.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOParallelExodus.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOParallel.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOPLY.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOPIO.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOPDAL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOOpenVDB.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOOggTheora.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOOMF.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOODBC.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIONetCDF.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOMySQL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOMotionFX.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOMINC.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOLSDyna.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOLAS.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOInfovis.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOImport.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOIOSS.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkioss.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOH5part.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkh5part.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOH5Rage.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOGeoJSON.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOFFMPEG.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOVideo.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOMovie.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOExportPDF.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOExportGL2PS.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingGL2PSOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOExport.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingVtkJS.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingSceneGraph.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOExodus.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkexodusII.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOEnSight.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOCityGML.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOChemistry.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOCesium3DTiles.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOGeometry.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOCONVERGECFD.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOHDF.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOCGNSReader.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOAsynchronous.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOAMR.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOADIOS2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkInteractionImage.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingStencil.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingStatistics.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingMorphological.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingMath.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingFourier.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkGUISupportQtSQL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOSQL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkGUISupportQtQuick.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkGUISupportQt.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkGeovisGDAL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOGDAL.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkGeovisCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkInfovisLayout.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkViewsCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkInteractionWidgets.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingVolume.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingAnnotation.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingHybrid.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingColor.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkInteractionStyle.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersTopology.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersSelection.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersSMP.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersReebGraph.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersPython.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersProgrammable.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersPoints.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersVerdict.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkverdict.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersParallelStatistics.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersParallelImaging.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersParallelDIY2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersOpenTURNS.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersImaging.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingGeneral.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersGeneric.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersFlowPaths.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersAMR.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersParallel.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersTexture.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersModeling.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkDomainsMicroscopy.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkDomainsChemistryOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingOpenGL2.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingUI.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingHyperTreeGrid.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersHyperTree.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersHybrid.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkDomainsChemistry.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonPython.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkWrappingPythonCore3.10.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonArchive.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkChartsCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingContext2D.dll.a
lab6.exe: _deps/nanobench-build/libnanobench.a
lab6.exe: C:/msys64/clang64/lib/libopencv_phase_unwrapping.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_optflow.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_highgui.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_datasets.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_plot.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_text.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_videoio.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_ml.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_shape.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_ximgproc.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_video.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_imgcodecs.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_objdetect.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_calib3d.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_dnn.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_features2d.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_flann.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_photo.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_imgproc.dll.a
lab6.exe: C:/msys64/clang64/lib/libopencv_core.dll.a
lab6.exe: C:/msys64/clang64/lib/libopenxr_loader.dll.a
lab6.exe: C:/msys64/clang64/lib/libopenvr_api.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkxdmfcore.dll.a
lab6.exe: C:/msys64/clang64/lib/libtheora.dll.a
lab6.exe: C:/msys64/clang64/lib/libtheoradec.dll.a
lab6.exe: C:/msys64/clang64/lib/libtheoraenc.dll.a
lab6.exe: C:/msys64/clang64/lib/libogg.dll.a
lab6.exe: C:/msys64/clang64/lib/libxml2.dll.a
lab6.exe: C:/msys64/clang64/lib/libicuuc.dll.a
lab6.exe: C:/msys64/clang64/lib/libgl2ps.dll.a
lab6.exe: C:/msys64/clang64/lib/libhpdf.dll.a
lab6.exe: C:/msys64/clang64/lib/libjsoncpp.dll.a
lab6.exe: C:/msys64/clang64/lib/libnetcdf.dll.a
lab6.exe: C:/msys64/clang64/lib/libhdf5_hl.dll.a
lab6.exe: C:/msys64/clang64/lib/libhdf5.dll.a
lab6.exe: C:/msys64/clang64/lib/libz.dll.a
lab6.exe: C:/msys64/clang64/lib/libblosc.dll.a
lab6.exe: C:/msys64/clang64/lib/libzstd.dll.a
lab6.exe: C:/msys64/clang64/lib/libbz2.dll.a
lab6.exe: C:/msys64/clang64/lib/libcurl.dll.a
lab6.exe: C:/msys64/clang64/lib/libxml2.dll.a
lab6.exe: C:/msys64/clang64/lib/libcgns.dll.a
lab6.exe: C:/msys64/clang64/lib/libhdf5.dll.a
lab6.exe: C:/msys64/clang64/lib/libhdf5_hl.dll.a
lab6.exe: C:/msys64/clang64/lib/libboost_serialization-mt.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6Sql.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6Quick.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6QmlModels.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6Qml.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6Network.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6OpenGLWidgets.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6Widgets.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6OpenGL.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6Gui.dll.a
lab6.exe: C:/msys64/clang64/lib/libQt6Core.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkInfovisBoostGraphAlgorithms.dll.a
lab6.exe: C:/msys64/clang64/lib/libproj.dll.a
lab6.exe: C:/msys64/clang64/lib/libsqlite3.dll.a
lab6.exe: C:/msys64/clang64/lib/libopenslide.dll.a
lab6.exe: C:/msys64/clang64/lib/libglew32.dll.a
lab6.exe: C:/msys64/clang64/lib/libpython3.10.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkInfovisCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersExtraction.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkParallelDIY.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOXML.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOXMLParser.dll.a
lab6.exe: C:/msys64/clang64/lib/libexpat.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkParallelCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOLegacy.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOCore.dll.a
lab6.exe: C:/msys64/clang64/lib/liblzma.dll.a
lab6.exe: C:/msys64/clang64/lib/liblz4.dll.a
lab6.exe: C:/msys64/clang64/lib/libdouble-conversion.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersStatistics.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingSources.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkIOImage.dll.a
lab6.exe: C:/msys64/clang64/lib/libtiff.dll.a
lab6.exe: C:/msys64/clang64/lib/libpng.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkmetaio.dll.a
lab6.exe: C:/msys64/clang64/lib/libjpeg.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkDICOMParser.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingFreeType.dll.a
lab6.exe: C:/msys64/clang64/lib/libfreetype.dll.a
lab6.exe: C:/msys64/clang64/lib/libz.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkRenderingCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersSources.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonColor.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkImagingCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersGeometry.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersGeneral.dll.a
lab6.exe: C:/msys64/clang64/lib/libfmt.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonComputationalGeometry.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkFiltersCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonExecutionModel.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonDataModel.dll.a
lab6.exe: C:/msys64/clang64/lib/libpugixml.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonSystem.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonMisc.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonTransforms.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonMath.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkkissfft.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkCommonCore.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtkloguru.dll.a
lab6.exe: C:/msys64/clang64/lib/libtbb12.dll.a
lab6.exe: C:/msys64/clang64/lib/libomp.dll.a
lab6.exe: C:/msys64/clang64/lib/libvtksys.dll.a
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/C/Users/olive/Documents/MCHA4400/lab6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lab6.exe"
	/C/msys64/clang64/bin/cmake.exe -E rm -f CMakeFiles/lab6.dir/objects.a
	/C/msys64/clang64/bin/llvm-ar.exe qc CMakeFiles/lab6.dir/objects.a $(lab6_OBJECTS) $(lab6_EXTERNAL_OBJECTS)
	/C/msys64/clang64/bin/c++.exe -Wall -O3 -Wl,--whole-archive CMakeFiles/lab6.dir/objects.a -Wl,--no-whole-archive -o lab6.exe -Wl,--out-implib,liblab6.dll.a -Wl,--major-image-version,0,--minor-image-version,0  libcommon.dll.a /C/msys64/clang64/lib/libopencv_gapi.dll.a /C/msys64/clang64/lib/libopencv_stitching.dll.a /C/msys64/clang64/lib/libopencv_alphamat.dll.a /C/msys64/clang64/lib/libopencv_aruco.dll.a /C/msys64/clang64/lib/libopencv_bgsegm.dll.a /C/msys64/clang64/lib/libopencv_ccalib.dll.a /C/msys64/clang64/lib/libopencv_cvv.dll.a /C/msys64/clang64/lib/libopencv_dnn_objdetect.dll.a /C/msys64/clang64/lib/libopencv_dnn_superres.dll.a /C/msys64/clang64/lib/libopencv_dpm.dll.a /C/msys64/clang64/lib/libopencv_face.dll.a /C/msys64/clang64/lib/libopencv_freetype.dll.a /C/msys64/clang64/lib/libopencv_fuzzy.dll.a /C/msys64/clang64/lib/libopencv_hdf.dll.a /C/msys64/clang64/lib/libopencv_hfs.dll.a /C/msys64/clang64/lib/libopencv_img_hash.dll.a /C/msys64/clang64/lib/libopencv_intensity_transform.dll.a /C/msys64/clang64/lib/libopencv_line_descriptor.dll.a /C/msys64/clang64/lib/libopencv_mcc.dll.a /C/msys64/clang64/lib/libopencv_ovis.dll.a /C/msys64/clang64/lib/libopencv_quality.dll.a /C/msys64/clang64/lib/libopencv_rapid.dll.a /C/msys64/clang64/lib/libopencv_reg.dll.a /C/msys64/clang64/lib/libopencv_rgbd.dll.a /C/msys64/clang64/lib/libopencv_saliency.dll.a /C/msys64/clang64/lib/libopencv_sfm.dll.a /C/msys64/clang64/lib/libopencv_stereo.dll.a /C/msys64/clang64/lib/libopencv_structured_light.dll.a /C/msys64/clang64/lib/libopencv_superres.dll.a /C/msys64/clang64/lib/libopencv_surface_matching.dll.a /C/msys64/clang64/lib/libopencv_tracking.dll.a /C/msys64/clang64/lib/libopencv_videostab.dll.a /C/msys64/clang64/lib/libopencv_viz.dll.a /C/msys64/clang64/lib/libopencv_wechat_qrcode.dll.a /C/msys64/clang64/lib/libopencv_xfeatures2d.dll.a /C/msys64/clang64/lib/libopencv_xobjdetect.dll.a /C/msys64/clang64/lib/libopencv_xphoto.dll.a /C/msys64/clang64/lib/libvtkWrappingTools.dll.a /C/msys64/clang64/lib/libvtkViewsQt.dll.a /C/msys64/clang64/lib/libvtkViewsInfovis.dll.a /C/msys64/clang64/lib/libvtkUtilitiesBenchmarks.dll.a /C/msys64/clang64/lib/libvtkViewsContext2D.dll.a /C/msys64/clang64/lib/libvtkzfp.dll.a /C/msys64/clang64/lib/libvtkTestingRendering.dll.a /C/msys64/clang64/lib/libvtkTestingIOSQL.dll.a /C/msys64/clang64/lib/libvtkTestingGenericBridge.dll.a /C/msys64/clang64/lib/libvtkTestingDataModel.dll.a /C/msys64/clang64/lib/libvtkRenderingRayTracing.dll.a /C/msys64/clang64/lib/libvtkRenderingVolumeAMR.dll.a /C/msys64/clang64/lib/libvtkRenderingQt.dll.a /C/msys64/clang64/lib/libvtkPythonContext2D.dll.a /C/msys64/clang64/lib/libvtkRenderingParallel.dll.a /C/msys64/clang64/lib/libvtkRenderingOpenXR.dll.a /C/msys64/clang64/lib/libvtkRenderingOpenVR.dll.a /C/msys64/clang64/lib/libvtkRenderingVR.dll.a /C/msys64/clang64/lib/libvtkRenderingVolumeOpenGL2.dll.a /C/msys64/clang64/lib/libvtkRenderingMatplotlib.dll.a /C/msys64/clang64/lib/libvtkPythonInterpreter.dll.a /C/msys64/clang64/lib/libvtkRenderingLabel.dll.a /C/msys64/clang64/lib/libvtkRenderingLOD.dll.a /C/msys64/clang64/lib/libvtkRenderingLICOpenGL2.dll.a /C/msys64/clang64/lib/libvtkRenderingImage.dll.a /C/msys64/clang64/lib/libvtkRenderingFreeTypeFontConfig.dll.a /C/msys64/clang64/lib/libvtkRenderingFFMPEGOpenGL2.dll.a /C/msys64/clang64/lib/libvtkRenderingExternal.dll.a /C/msys64/clang64/lib/libvtkRenderingContextOpenGL2.dll.a /C/msys64/clang64/lib/libvtkIOXdmf2.dll.a /C/msys64/clang64/lib/libvtkxdmf2.dll.a /C/msys64/clang64/lib/libvtkIOVeraOut.dll.a /C/msys64/clang64/lib/libvtkIOVPIC.dll.a /C/msys64/clang64/lib/libvtkvpic.dll.a /C/msys64/clang64/lib/libvtkIOTecplotTable.dll.a /C/msys64/clang64/lib/libvtkIOTRUCHAS.dll.a /C/msys64/clang64/lib/libvtkIOSegY.dll.a /C/msys64/clang64/lib/libvtkIOPostgreSQL.dll.a /C/msys64/clang64/lib/libvtkIOXdmf3.dll.a /C/msys64/clang64/lib/libvtkxdmf3.dll.a /C/msys64/clang64/lib/libvtkIOParallelXML.dll.a /C/msys64/clang64/lib/libvtkIOParallelLSDyna.dll.a /C/msys64/clang64/lib/libvtkIOParallelExodus.dll.a /C/msys64/clang64/lib/libvtkIOParallel.dll.a /C/msys64/clang64/lib/libvtkIOPLY.dll.a /C/msys64/clang64/lib/libvtkIOPIO.dll.a /C/msys64/clang64/lib/libvtkIOPDAL.dll.a /C/msys64/clang64/lib/libvtkIOOpenVDB.dll.a /C/msys64/clang64/lib/libvtkIOOggTheora.dll.a /C/msys64/clang64/lib/libvtkIOOMF.dll.a /C/msys64/clang64/lib/libvtkIOODBC.dll.a /C/msys64/clang64/lib/libvtkIONetCDF.dll.a /C/msys64/clang64/lib/libvtkIOMySQL.dll.a /C/msys64/clang64/lib/libvtkIOMotionFX.dll.a /C/msys64/clang64/lib/libvtkIOMINC.dll.a /C/msys64/clang64/lib/libvtkIOLSDyna.dll.a /C/msys64/clang64/lib/libvtkIOLAS.dll.a /C/msys64/clang64/lib/libvtkIOInfovis.dll.a /C/msys64/clang64/lib/libvtkIOImport.dll.a /C/msys64/clang64/lib/libvtkIOIOSS.dll.a /C/msys64/clang64/lib/libvtkioss.dll.a /C/msys64/clang64/lib/libvtkIOH5part.dll.a /C/msys64/clang64/lib/libvtkh5part.dll.a /C/msys64/clang64/lib/libvtkIOH5Rage.dll.a /C/msys64/clang64/lib/libvtkIOGeoJSON.dll.a /C/msys64/clang64/lib/libvtkIOFFMPEG.dll.a /C/msys64/clang64/lib/libvtkIOVideo.dll.a /C/msys64/clang64/lib/libvtkIOMovie.dll.a /C/msys64/clang64/lib/libvtkIOExportPDF.dll.a /C/msys64/clang64/lib/libvtkIOExportGL2PS.dll.a /C/msys64/clang64/lib/libvtkRenderingGL2PSOpenGL2.dll.a /C/msys64/clang64/lib/libvtkIOExport.dll.a /C/msys64/clang64/lib/libvtkRenderingVtkJS.dll.a /C/msys64/clang64/lib/libvtkRenderingSceneGraph.dll.a /C/msys64/clang64/lib/libvtkIOExodus.dll.a /C/msys64/clang64/lib/libvtkexodusII.dll.a /C/msys64/clang64/lib/libvtkIOEnSight.dll.a /C/msys64/clang64/lib/libvtkIOCityGML.dll.a /C/msys64/clang64/lib/libvtkIOChemistry.dll.a /C/msys64/clang64/lib/libvtkIOCesium3DTiles.dll.a /C/msys64/clang64/lib/libvtkIOGeometry.dll.a /C/msys64/clang64/lib/libvtkIOCONVERGECFD.dll.a /C/msys64/clang64/lib/libvtkIOHDF.dll.a /C/msys64/clang64/lib/libvtkIOCGNSReader.dll.a /C/msys64/clang64/lib/libvtkIOAsynchronous.dll.a /C/msys64/clang64/lib/libvtkIOAMR.dll.a /C/msys64/clang64/lib/libvtkIOADIOS2.dll.a /C/msys64/clang64/lib/libvtkInteractionImage.dll.a /C/msys64/clang64/lib/libvtkImagingStencil.dll.a /C/msys64/clang64/lib/libvtkImagingStatistics.dll.a /C/msys64/clang64/lib/libvtkImagingOpenGL2.dll.a /C/msys64/clang64/lib/libvtkImagingMorphological.dll.a /C/msys64/clang64/lib/libvtkImagingMath.dll.a /C/msys64/clang64/lib/libvtkImagingFourier.dll.a /C/msys64/clang64/lib/libvtkGUISupportQtSQL.dll.a /C/msys64/clang64/lib/libvtkIOSQL.dll.a /C/msys64/clang64/lib/libvtkGUISupportQtQuick.dll.a /C/msys64/clang64/lib/libvtkGUISupportQt.dll.a /C/msys64/clang64/lib/libvtkGeovisGDAL.dll.a /C/msys64/clang64/lib/libvtkIOGDAL.dll.a /C/msys64/clang64/lib/libvtkGeovisCore.dll.a /C/msys64/clang64/lib/libvtkInfovisLayout.dll.a /C/msys64/clang64/lib/libvtkViewsCore.dll.a /C/msys64/clang64/lib/libvtkInteractionWidgets.dll.a /C/msys64/clang64/lib/libvtkRenderingVolume.dll.a /C/msys64/clang64/lib/libvtkRenderingAnnotation.dll.a /C/msys64/clang64/lib/libvtkImagingHybrid.dll.a /C/msys64/clang64/lib/libvtkImagingColor.dll.a /C/msys64/clang64/lib/libvtkInteractionStyle.dll.a /C/msys64/clang64/lib/libvtkFiltersTopology.dll.a /C/msys64/clang64/lib/libvtkFiltersSelection.dll.a /C/msys64/clang64/lib/libvtkFiltersSMP.dll.a /C/msys64/clang64/lib/libvtkFiltersReebGraph.dll.a /C/msys64/clang64/lib/libvtkFiltersPython.dll.a /C/msys64/clang64/lib/libvtkFiltersProgrammable.dll.a /C/msys64/clang64/lib/libvtkFiltersPoints.dll.a /C/msys64/clang64/lib/libvtkFiltersVerdict.dll.a /C/msys64/clang64/lib/libvtkverdict.dll.a /C/msys64/clang64/lib/libvtkFiltersParallelStatistics.dll.a /C/msys64/clang64/lib/libvtkFiltersParallelImaging.dll.a /C/msys64/clang64/lib/libvtkFiltersParallelDIY2.dll.a /C/msys64/clang64/lib/libvtkFiltersOpenTURNS.dll.a /C/msys64/clang64/lib/libvtkFiltersImaging.dll.a /C/msys64/clang64/lib/libvtkImagingGeneral.dll.a /C/msys64/clang64/lib/libvtkFiltersGeneric.dll.a /C/msys64/clang64/lib/libvtkFiltersFlowPaths.dll.a /C/msys64/clang64/lib/libvtkFiltersAMR.dll.a /C/msys64/clang64/lib/libvtkFiltersParallel.dll.a /C/msys64/clang64/lib/libvtkFiltersTexture.dll.a /C/msys64/clang64/lib/libvtkFiltersModeling.dll.a /C/msys64/clang64/lib/libvtkDomainsMicroscopy.dll.a /C/msys64/clang64/lib/libvtkDomainsChemistryOpenGL2.dll.a /C/msys64/clang64/lib/libvtkRenderingOpenGL2.dll.a /C/msys64/clang64/lib/libvtkRenderingUI.dll.a /C/msys64/clang64/lib/libvtkRenderingHyperTreeGrid.dll.a /C/msys64/clang64/lib/libvtkFiltersHyperTree.dll.a /C/msys64/clang64/lib/libvtkFiltersHybrid.dll.a /C/msys64/clang64/lib/libvtkDomainsChemistry.dll.a /C/msys64/clang64/lib/libvtkCommonPython.dll.a /C/msys64/clang64/lib/libvtkWrappingPythonCore3.10.dll.a /C/msys64/clang64/lib/libvtkCommonArchive.dll.a /C/msys64/clang64/lib/libvtkChartsCore.dll.a /C/msys64/clang64/lib/libvtkRenderingContext2D.dll.a _deps/nanobench-build/libnanobench.a /C/msys64/clang64/lib/libopencv_phase_unwrapping.dll.a /C/msys64/clang64/lib/libopencv_optflow.dll.a /C/msys64/clang64/lib/libopencv_highgui.dll.a /C/msys64/clang64/lib/libopencv_datasets.dll.a /C/msys64/clang64/lib/libopencv_plot.dll.a /C/msys64/clang64/lib/libopencv_text.dll.a /C/msys64/clang64/lib/libopencv_videoio.dll.a /C/msys64/clang64/lib/libopencv_ml.dll.a /C/msys64/clang64/lib/libopencv_shape.dll.a /C/msys64/clang64/lib/libopencv_ximgproc.dll.a /C/msys64/clang64/lib/libopencv_video.dll.a /C/msys64/clang64/lib/libopencv_imgcodecs.dll.a /C/msys64/clang64/lib/libopencv_objdetect.dll.a /C/msys64/clang64/lib/libopencv_calib3d.dll.a /C/msys64/clang64/lib/libopencv_dnn.dll.a /C/msys64/clang64/lib/libopencv_features2d.dll.a /C/msys64/clang64/lib/libopencv_flann.dll.a /C/msys64/clang64/lib/libopencv_photo.dll.a /C/msys64/clang64/lib/libopencv_imgproc.dll.a /C/msys64/clang64/lib/libopencv_core.dll.a /C/msys64/clang64/lib/libopenxr_loader.dll.a /C/msys64/clang64/lib/libopenvr_api.dll.a /C/msys64/clang64/lib/libvtkxdmfcore.dll.a /C/msys64/clang64/lib/libtheora.dll.a /C/msys64/clang64/lib/libtheoradec.dll.a /C/msys64/clang64/lib/libtheoraenc.dll.a /C/msys64/clang64/lib/libogg.dll.a /C/msys64/clang64/lib/libxml2.dll.a /C/msys64/clang64/lib/libicuuc.dll.a /C/msys64/clang64/lib/libgl2ps.dll.a /C/msys64/clang64/lib/libhpdf.dll.a /C/msys64/clang64/lib/libjsoncpp.dll.a /C/msys64/clang64/lib/libnetcdf.dll.a /C/msys64/clang64/lib/libhdf5_hl.dll.a /C/msys64/clang64/lib/libhdf5.dll.a /C/msys64/clang64/lib/libz.dll.a /C/msys64/clang64/lib/libblosc.dll.a /C/msys64/clang64/lib/libzstd.dll.a /C/msys64/clang64/lib/libbz2.dll.a /C/msys64/clang64/lib/libcurl.dll.a /C/msys64/clang64/lib/libxml2.dll.a /C/msys64/clang64/lib/libcgns.dll.a /C/msys64/clang64/lib/libhdf5.dll.a /C/msys64/clang64/lib/libhdf5_hl.dll.a /C/msys64/clang64/lib/libboost_serialization-mt.dll.a /C/msys64/clang64/lib/libQt6Sql.dll.a /C/msys64/clang64/lib/libQt6Quick.dll.a /C/msys64/clang64/lib/libQt6QmlModels.dll.a -luser32 /C/msys64/clang64/lib/libQt6Qml.dll.a /C/msys64/clang64/lib/libQt6Network.dll.a -lshell32 /C/msys64/clang64/lib/libQt6OpenGLWidgets.dll.a /C/msys64/clang64/lib/libQt6Widgets.dll.a /C/msys64/clang64/lib/libQt6OpenGL.dll.a /C/msys64/clang64/lib/libQt6Gui.dll.a -ld3d11 -ldxgi -ldxguid -ldcomp /C/msys64/clang64/lib/libQt6Core.dll.a -lmpr -luserenv /C/msys64/clang64/lib/libvtkInfovisBoostGraphAlgorithms.dll.a /C/msys64/clang64/lib/libproj.dll.a /C/msys64/clang64/lib/libsqlite3.dll.a /C/msys64/clang64/lib/libopenslide.dll.a /C/msys64/clang64/lib/libglew32.dll.a -lopengl32 /C/msys64/clang64/lib/libpython3.10.dll.a /C/msys64/clang64/lib/libvtkInfovisCore.dll.a /C/msys64/clang64/lib/libvtkFiltersExtraction.dll.a /C/msys64/clang64/lib/libvtkParallelDIY.dll.a /C/msys64/clang64/lib/libvtkIOXML.dll.a /C/msys64/clang64/lib/libvtkIOXMLParser.dll.a /C/msys64/clang64/lib/libexpat.dll.a /C/msys64/clang64/lib/libvtkParallelCore.dll.a /C/msys64/clang64/lib/libvtkIOLegacy.dll.a /C/msys64/clang64/lib/libvtkIOCore.dll.a /C/msys64/clang64/lib/liblzma.dll.a /C/msys64/clang64/lib/liblz4.dll.a /C/msys64/clang64/lib/libdouble-conversion.dll.a /C/msys64/clang64/lib/libvtkFiltersStatistics.dll.a /C/msys64/clang64/lib/libvtkImagingSources.dll.a /C/msys64/clang64/lib/libvtkIOImage.dll.a /C/msys64/clang64/lib/libtiff.dll.a /C/msys64/clang64/lib/libpng.dll.a /C/msys64/clang64/lib/libvtkmetaio.dll.a /C/msys64/clang64/lib/libjpeg.dll.a /C/msys64/clang64/lib/libvtkDICOMParser.dll.a /C/msys64/clang64/lib/libvtkRenderingFreeType.dll.a /C/msys64/clang64/lib/libfreetype.dll.a /C/msys64/clang64/lib/libz.dll.a /C/msys64/clang64/lib/libvtkRenderingCore.dll.a /C/msys64/clang64/lib/libvtkFiltersSources.dll.a /C/msys64/clang64/lib/libvtkCommonColor.dll.a /C/msys64/clang64/lib/libvtkImagingCore.dll.a /C/msys64/clang64/lib/libvtkFiltersGeometry.dll.a /C/msys64/clang64/lib/libvtkFiltersGeneral.dll.a /C/msys64/clang64/lib/libfmt.dll.a /C/msys64/clang64/lib/libvtkCommonComputationalGeometry.dll.a /C/msys64/clang64/lib/libvtkFiltersCore.dll.a /C/msys64/clang64/lib/libvtkCommonExecutionModel.dll.a /C/msys64/clang64/lib/libvtkCommonDataModel.dll.a /C/msys64/clang64/lib/libpugixml.dll.a /C/msys64/clang64/lib/libvtkCommonSystem.dll.a /C/msys64/clang64/lib/libvtkCommonMisc.dll.a /C/msys64/clang64/lib/libvtkCommonTransforms.dll.a /C/msys64/clang64/lib/libvtkCommonMath.dll.a /C/msys64/clang64/lib/libvtkkissfft.dll.a /C/msys64/clang64/lib/libvtkCommonCore.dll.a /C/msys64/clang64/lib/libvtkloguru.dll.a -pthread /C/msys64/clang64/lib/libtbb12.dll.a /C/msys64/clang64/lib/libomp.dll.a /C/msys64/clang64/lib/libvtksys.dll.a -lws2_32 -lpsapi -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 

# Rule to build all files generated by this target.
CMakeFiles/lab6.dir/build: lab6.exe
.PHONY : CMakeFiles/lab6.dir/build

CMakeFiles/lab6.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab6.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab6.dir/clean

CMakeFiles/lab6.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MSYS Makefiles" /C/Users/olive/Documents/MCHA4400/lab6 /C/Users/olive/Documents/MCHA4400/lab6 /C/Users/olive/Documents/MCHA4400/lab6/build /C/Users/olive/Documents/MCHA4400/lab6/build /C/Users/olive/Documents/MCHA4400/lab6/build/CMakeFiles/lab6.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab6.dir/depend

