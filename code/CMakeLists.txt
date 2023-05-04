    cmake_minimum_required(VERSION 3.14)
set(CMAKE_BUILD_TYPE Debug)
    # include(ExternalProject)

# set project name and languages
project(Ping_Pong CXX C Fortran CUDA)

#set(CMAKE_CXX_FLAGS "-DMPICH_SKIP_MPICXX -arch=sm_35")
#set(CMAKE_CUDA_FLAGS "-DMPICH_SKIP_MPICXX -arch=sm_35")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/kokkos/bin/nvcc_wrapper -ccbin xlc++")

# MPI
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)

if (ENABLE_TEMPI)
    set(TEMPI_ENABLE_KAHIP OFF)
    add_subdirectory(tempi)
endif(ENABLE_TEMPI)

if (ENABLE_WRAP)
    # wrap.py setup -- grab the add_wrapped_file macro.
    set(WRAP ${PROJECT_SOURCE_DIR}/wrap/wrap.py)
    set(OLD_MPI_CXX_COMPILER $MPI_CXX_COMPILER)
    set(MPI_CXX_COMPILER "mpicxx")
    include(wrap/WrapConfig.cmake)
    add_wrapped_file(nvtx_pmpi.cpp src/nvtx.w)
    set(MPI_CXX_COMPILER $OLD_MPI_CXX_COMPILER)
    add_library(nvtx_pmpi SHARED nvtx_pmpi.cpp)
    #LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/CUP-ECS/ping-pong-gpu/build-mvapich2/tempi/libtempi.so
    target_link_libraries(ping_pong PRIVATE nvToolsExt)
endif(ENABLE_WRAP)

add_executable(ping_pong src/Main.cpp)
target_link_libraries(ping_pong PUBLIC Kokkos::kokkos)
target_link_libraries(ping_pong PUBLIC MPI::MPI_CXX)

#set_target_properties(ping_pong PROPERTIES CUDA_ARCHITECTURES "80")
if (ENABLE_TEMPI)
    target_link_libraries(ping_pong PRIVATE tempi::tempi)
endif(ENABLE_TEMPI)

#target_link_libraries(ping_pong PUBLIC nvtx_pmpi)

# set install destination
install(TARGETS ping_pong
        RUNTIME DESTINATION)

# print build strings
message("Ping Pong Build Time ${BUILD_TIME}")
