# Copyright (C) Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.

include_directories ("${PROJECT_SOURCE_DIR}/backend")
add_subdirectory (backend)

# put together all ArgoDSM source files
set(argo_sources "argo.cpp")

set(allocator_sources allocators.cpp)
foreach(src ${allocator_sources})
	list(APPEND argo_sources allocators/${src})
endforeach(src)

set(env_sources env.cpp)
foreach(src ${env_sources})
	list(APPEND argo_sources env/${src})
endforeach(src)

set(synchronization_sources
	synchronization.cpp
	cohort_lock.cpp
	intranode/mcs_lock.cpp
)
foreach(src ${synchronization_sources})
	list(APPEND argo_sources synchronization/${src})
endforeach(src)


# exactly one of these must be enabled. below is some code to ensure this.
option(ARGO_VM_SHM
	"Handle virtual addresses using POSIX shared memory. Size-limited." ON)
option(ARGO_VM_ANONYMOUS
	"Handle virtual addresses using anonymously-mapped memory. Slow." OFF)
option(ARGO_VM_MEMFD
	"Handle virtual addresses using an anonymous memory file. Requires kernel 3.17+." OFF)
option(ARGO_VM_PMEM
	"Handle virtual addresses using memory mapped files for shared and persistent memory." OFF)

set(ARGO_VM_COUNT 0 CACHE INTERNAL "Check for multiple virtual address handlers")
unset(vm_libs)

if(ARGO_VM_SHM)
	set(vm_sources shm.cpp)
	set(vm_libs rt)
	math(EXPR ARGO_VM_COUNT "${ARGO_VM_COUNT} + 1")
endif(ARGO_VM_SHM)

if(ARGO_VM_ANONYMOUS)
	set(vm_sources anonymous.cpp)
	math(EXPR ARGO_VM_COUNT "${ARGO_VM_COUNT} + 1")
endif(ARGO_VM_ANONYMOUS)

if(ARGO_VM_MEMFD)
	set(vm_sources memfd.cpp)
	math(EXPR ARGO_VM_COUNT "${ARGO_VM_COUNT} + 1")
endif(ARGO_VM_MEMFD)

if(ARGO_VM_PMEM)
	set(vm_sources pmem.cpp)
	set(ARGO_USE_LIBMEMKIND 1)
	math(EXPR ARGO_VM_COUNT "${ARGO_VM_COUNT} + 1")
endif(ARGO_VM_PMEM)

if(NOT ARGO_VM_COUNT EQUAL 1)
	message(FATAL_ERROR "${ARGO_VM_COUNT} virtual address handlers selected. Please select exactly one virtual address handler.")
endif(NOT ARGO_VM_COUNT EQUAL 1)

foreach(src ${vm_sources})
	list(APPEND argo_sources virtual_memory/${src})
endforeach(src)

# add the frontend library
add_library(argo SHARED ${argo_sources})

option(ARGO_USE_LIBNUMA
	"Use libnuma to determine NUMA structure within ArgoDSM" ON)
if(ARGO_USE_LIBNUMA)
	target_link_libraries(argo numa)
	add_definitions(-DARGO_USE_LIBNUMA)
endif(ARGO_USE_LIBNUMA)

if(ARGO_USE_LIBMEMKIND EQUAL 1)
	# include internal jemalloc/memkind headers (should not have been required!)
	set(INTERNALS "/home/ianevlavis/packages/memkind-1.10.1")
	find_path(INTERNALS_INCLUDE_DIR NAMES libmemkind.la PATHS ${INTERNALS})
	if(INTERNALS_INCLUDE_DIR)
		target_include_directories(argo PRIVATE ${INTERNALS_INCLUDE_DIR}/jemalloc/include)
		target_include_directories(argo PRIVATE ${INTERNALS_INCLUDE_DIR}/include)
	endif()
	# include installed memkind headers/lib (required)
	set(MEMKIND "/home/ianevlavis/memkind")
	find_path(MEMKIND_INCLUDE_DIR NAMES memkind.h PATHS ${MEMKIND}/include)
	if(MEMKIND_INCLUDE_DIR)
		target_include_directories(argo PRIVATE ${MEMKIND_INCLUDE_DIR})
	endif()
	find_path(MEMKIND_LIB_DIR NAMES libmemkind.so PATHS ${MEMKIND}/lib)
	if(MEMKIND_LIB_DIR)
		target_link_libraries(argo ${MEMKIND_LIB_DIR}/libmemkind.so)
	endif()
	add_definitions(-DARGO_USE_LIBMEMKIND)
endif(ARGO_USE_LIBMEMKIND EQUAL 1)

target_link_libraries(argo ${vm_libs})

#install (TARGETS argo DESTINATION bin)

install(TARGETS argo
	COMPONENT "Runtime"
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)

install(DIRECTORY .
	DESTINATION include/argo
	COMPONENT "Development"
	FILES_MATCHING
		PATTERN "*.h"
		PATTERN "*.hpp")
