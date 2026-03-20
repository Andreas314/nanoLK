cmake_minimum_required(VERSION 3.10)

project(nanoLK)

# Use GCC (optional, only if you want to force it)
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

# Use modern C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

# Add source files
set(SOURCES
    src/main.cpp
    src/nanoLK.cpp
)

# Build executable
add_executable(nanoLK ${SOURCES})

# Include headers from src
target_include_directories(nanoLK PRIVATE src)
target_link_libraries(nanoLK PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(nanoLK PUBLIC OpenMP::OpenMP_CXX)
