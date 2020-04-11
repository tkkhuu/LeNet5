# This module defines the following variables:
#
# ::
#
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT_FOUND
#
# ::
#
#   TensorRT_VERSION_STRING - version (x.y.z)
#   TensorRT_VERSION_MAJOR  - major version (x)
#   TensorRT_VERSION_MINOR  - minor version (y)
#   TensorRT_VERSION_PATCH  - patch version (z)
#
# Hints
# ^^^^^
# A user may set ``TensorRT_ROOT`` to an installation root to tell this module where to look.
#

set(_TensorRT_SEARCHES)

if(TensorRT_ROOT)
  set(_TensorRT_SEARCH_ROOT PATHS ${TensorRT_ROOT} NO_DEFAULT_PATH)
  list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_ROOT)
endif()

# appends some common paths
set(_TensorRT_SEARCH_NORMAL
  PATHS "/usr"
)
list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_NORMAL)

# Include dir
foreach(search ${_TensorRT_SEARCHES})
  find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h ${${search}} PATH_SUFFIXES include)
endforeach()

if(NOT TensorRT_LIBRARY)
  foreach(search ${_TensorRT_SEARCHES})
    find_library(TensorRT_LIBRARY NAMES nvinfer ${${search}} PATH_SUFFIXES lib)
  endforeach()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TensorRT REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR VERSION_VAR TensorRT_VERSION_STRING)



if(TensorRT_FOUND)
  
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
  
  file(STRINGS "${TensorRT_INCLUDE_DIRS}/NvInferVersion.h" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

  foreach(TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_${TYPE} ${TRT_TYPE_STRING})
  endforeach(TYPE)

  foreach(TYPE MAJOR MINOR PATCH)
    string(REGEX MATCH "NV_TENSORRT_SONAME_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
  endforeach(TYPE)
  set(TensorRT_VERSION_STRING "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}.${TRT_BUILD}" CACHE STRING "TRT project version")

  if(NOT TensorRT_LIBRARIES)
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})
  endif()

  if(NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}")
    set_property(TARGET TensorRT::TensorRT APPEND PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")
  endif()
endif()