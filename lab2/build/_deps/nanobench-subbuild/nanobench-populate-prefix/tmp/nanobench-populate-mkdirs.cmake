# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-src"
  "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-build"
  "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix"
  "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/tmp"
  "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
  "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src"
  "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/olive/Documents/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
