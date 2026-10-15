set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

set(GVHF_MD_SOURCES
  md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu
  md_indices.cu md_j_driver.cu md_pairdata.c 
  contract_int3c2e.cu
  pbc_md_contract_j.cu
  contract_Et.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../fixed_point.cu
)
add_library(gvhf_md SHARED ${GVHF_MD_SOURCES})

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

set_target_properties(gvhf_md PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gvhf_md OpenMP::OpenMP_C)

if(BUILD_FIXED_POINT)
  add_library(gvhf_md_fp SHARED ${GVHF_MD_SOURCES})
  set_target_properties(gvhf_md_fp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(gvhf_md_fp PRIVATE FIXED_POINT_ACCUM=1 FIXED_POINT_SHIFT=${FIXED_POINT_SHIFT})
  target_link_libraries(gvhf_md_fp OpenMP::OpenMP_C)
endif()
