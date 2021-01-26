/**
 * @file
 * @brief This file provides facilities for handling virtual memory and virtual address space
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#ifndef argo_virtual_memory_virtual_memory_hpp
#define argo_virtual_memory_virtual_memory_hpp argo_virtual_memory_virtual_memory_hpp

#include <utility>

#ifdef ARGO_USE_LIBMEMKIND
#include <memkind.h>
#include <memkind/internal/memkind_pmem.h>
#include <memkind/internal/memkind_private.h>
#endif

namespace argo {
	namespace virtual_memory {
		/**
		 * @brief initialize the ArgoDSM virtual address space
		 * @todo virtual address space handling should be wrapped into an object
		 */
		void init();

		/**
		 * @brief get a pointer to the ArgoDSM virtual memory
		 * @return the pointer to ArgoDSM virtual memory
		 */
		void* start_address();

		/**
		 * @brief get size of the ArgoDSM virtual memory
		 * @return the size of the ArgoDSM virtual memory
		 */
		std::size_t size();

#ifdef ARGO_USE_LIBMEMKIND
		/**
		 * @brief destroy the kind object associated with persistent memory
		 */
		void destroy_kind_pmem();
#endif

		/**
		 * @brief allocate memory that can be mapped into ArgoDSM virtual address space later
		 * @param alignment the alignment of the allocation
		 * @param size size of the allocation
		 * @param smem memory to allocate space (0: shared memory, 1: persistent memory)
		 * @return a pointer to the new memory allocation
		 * @details this will allocate memory that is guaranteed to work with map_memory().
		 *          Any memory allocated through other means may not be possible to map
		 *          into the visible ArgoDSM virtual memory space later.
		 */
		void* allocate_mappable(std::size_t alignment, std::size_t size, int smem = 0);

		/**
		 * @brief map memory into ArgoDSM virtual address space
		 * @param addr the address to map to
		 * @param size the size of the mapping
		 * @param offset the offset into the backing memory
		 * @param prot protection flags for the mapping
		 * @param smem memory to allocate space (0: shared memory, 1: persistent memory)
		 */
		void map_memory(void* addr, std::size_t size, std::size_t offset, int prot, int smem = 0);
	} // namespace virtual_memory
} // namespace argo

#endif
