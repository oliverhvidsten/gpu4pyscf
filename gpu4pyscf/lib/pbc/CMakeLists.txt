set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

set(PBC_SOURCES
  pbc_driver.cu
  ft_ao.cu ft_ao_ip1.cu contract_ft_ao.cu
  fill_int3c2e.cu contract_int3c2e.cu ejk_int3c2e_ip1.cu
  fill_int2c2e.cu fill_int2c2e_ip1.cu e_int2c2e_ip1.cu
  contract_int3c2e_pvp.cu
  overlap.cu ppnl.cu
  supmol_sr_estimator.cu
  rys_roots_dat.cu
  rys_contract_k.cu rys_contract_jk_ip1.cu
  rys_contract_j.cu
  unrolled_rys_k.cu
  nr_eval_gto.cu
  sorting.c write_cderi.c int3c2e_recontract.c
  screen.cu
  fill_triu.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../fixed_point.cu
)
add_library(pbc SHARED ${PBC_SOURCES})

set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(pbc OpenMP::OpenMP_C)

if(BUILD_FIXED_POINT)
  add_library(pbc_fp SHARED ${PBC_SOURCES})
  set_target_properties(pbc_fp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(pbc_fp PRIVATE FIXED_POINT_ACCUM=1 FIXED_POINT_SHIFT=${FIXED_POINT_SHIFT})
  target_link_libraries(pbc_fp OpenMP::OpenMP_C)
endif()
