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
include _deps/nanobench-build/CMakeFiles/nanobench.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/nanobench-build/CMakeFiles/nanobench.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/nanobench-build/CMakeFiles/nanobench.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/nanobench-build/CMakeFiles/nanobench.dir/flags.make

_deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj: _deps/nanobench-build/CMakeFiles/nanobench.dir/flags.make
_deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj: _deps/nanobench-src/src/test/app/nanobench.cpp
_deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj: _deps/nanobench-build/CMakeFiles/nanobench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/C/Users/olive/Documents/MCHA4400/lab6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj"
	cd /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build && /C/msys64/clang64/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj -MF CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj.d -o CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj -c /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-src/src/test/app/nanobench.cpp

_deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.i"
	cd /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build && /C/msys64/clang64/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-src/src/test/app/nanobench.cpp > CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.i

_deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.s"
	cd /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build && /C/msys64/clang64/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-src/src/test/app/nanobench.cpp -o CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.s

# Object files for target nanobench
nanobench_OBJECTS = \
"CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj"

# External object files for target nanobench
nanobench_EXTERNAL_OBJECTS =

_deps/nanobench-build/libnanobench.a: _deps/nanobench-build/CMakeFiles/nanobench.dir/src/test/app/nanobench.cpp.obj
_deps/nanobench-build/libnanobench.a: _deps/nanobench-build/CMakeFiles/nanobench.dir/build.make
_deps/nanobench-build/libnanobench.a: _deps/nanobench-build/CMakeFiles/nanobench.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/C/Users/olive/Documents/MCHA4400/lab6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libnanobench.a"
	cd /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build && $(CMAKE_COMMAND) -P CMakeFiles/nanobench.dir/cmake_clean_target.cmake
	cd /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nanobench.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/nanobench-build/CMakeFiles/nanobench.dir/build: _deps/nanobench-build/libnanobench.a
.PHONY : _deps/nanobench-build/CMakeFiles/nanobench.dir/build

_deps/nanobench-build/CMakeFiles/nanobench.dir/clean:
	cd /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build && $(CMAKE_COMMAND) -P CMakeFiles/nanobench.dir/cmake_clean.cmake
.PHONY : _deps/nanobench-build/CMakeFiles/nanobench.dir/clean

_deps/nanobench-build/CMakeFiles/nanobench.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MSYS Makefiles" /C/Users/olive/Documents/MCHA4400/lab6 /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-src /C/Users/olive/Documents/MCHA4400/lab6/build /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build /C/Users/olive/Documents/MCHA4400/lab6/build/_deps/nanobench-build/CMakeFiles/nanobench.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/nanobench-build/CMakeFiles/nanobench.dir/depend

