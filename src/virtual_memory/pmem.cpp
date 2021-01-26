/**
 * @file
 * @brief This file implements the virtual memory and virtual address handling
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>
#include "virtual_memory.hpp"

namespace {
	/* file constants */
	/** @todo hardcoded start address */
	const char* ARGO_START = (char*) 0x200000000000l;
	/** @todo hardcoded end address */
	const char* ARGO_END   = (char*) 0x600000000000l;
	/** @todo hardcoded size */
	const ptrdiff_t ARGO_SIZE = ARGO_END - ARGO_START;
	/** @todo hardcoded maximum size */
	const ptrdiff_t ARGO_SIZE_LIMIT = 0x80000000000l;

	/** @brief error message string */
	const std::string msg_alloc_fail = "ArgoDSM could not allocate mappable memory";
	/** @brief error message string */
	const std::string msg_mmap_fail = "ArgoDSM failed to map in virtual address space.";
	/** @brief error message string */
	const std::string msg_main_mmap_fail = "ArgoDSM failed to set up virtual memory. Please report a bug.";

	/* smem variables */
	/** @brief a file descriptor for backing the virtual address space used by ArgoDSM */
	int fd_smem;
	/** @brief the size of the ArgoDSM virtual address space */
	std::size_t avail_smem;

	/* pmem variables */
	/** @brief the kind object associated with persistent memory */
	memkind_t pmem_kind = NULL;
	/** @brief the size of the ArgoDSM virtual address space */
	std::size_t avail_pmem;

	/* vmem variables */
	/** @brief the address at which the virtual address space used by ArgoDSM starts */
	void* start_addr;
	/** @brief the size of the ArgoDSM virtual address space */
	std::size_t avail;
}

namespace argo {
	namespace virtual_memory {
		void init_smem() {
			/* find maximum filesize */
			struct statvfs b;
			statvfs("/dev/shm", &b);
			avail_smem = b.f_bavail * b.f_bsize;
			if(avail_smem > static_cast<unsigned long>(ARGO_SIZE_LIMIT)) {
				avail_smem = ARGO_SIZE_LIMIT;
			}
			std::string filename = "/argocache" + std::to_string(getpid());
			fd_smem = shm_open(filename.c_str(), O_RDWR|O_CREAT, 0644);
			if(shm_unlink(filename.c_str())) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
			if(ftruncate(fd_smem, avail_smem)) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
		}

		void init_pmem() {
			/* find maximum filesize */
			struct statvfs b;
			statvfs("/mnt/pmem0", &b);
			avail_pmem = b.f_bavail * b.f_bsize;
			if(avail_pmem > static_cast<unsigned long>(ARGO_SIZE_LIMIT)) {
				avail_pmem = ARGO_SIZE_LIMIT;
			}
			/* file is unique, no need to follow the above naming convention */
			std::string filename_psm = "/mnt/pmem0";
			/* if '0', allocation automatically grows when it runs out of space */
			if (memkind_create_pmem(filename_psm.c_str(), 0, &pmem_kind)) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
			/* change file mode to be sure, since mkstemp() default is 0600 (optional) */
			struct memkind_pmem *priv = (memkind_pmem*)pmem_kind->priv;
			if (fchmod(priv->fd, 0644)) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
		}

		void init_vmem() {
			/** @todo check desired range is free */
			avail = avail_smem + avail_pmem;
			constexpr int flags = MAP_ANONYMOUS|MAP_SHARED|MAP_FIXED|MAP_NORESERVE;
			start_addr = ::mmap((void*)ARGO_START, avail, PROT_NONE, flags, -1, 0);
			if(start_addr == MAP_FAILED) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
		}

		void init() {
			init_smem();
			init_pmem();
			init_vmem();
		}

		void* start_address() {
			return start_addr;
		}

		std::size_t size() {
			return avail;
		}

		void destroy_kind_pmem() {
			if (memkind_destroy_kind(pmem_kind)) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
		}

		void* allocate_mappable(std::size_t alignment, std::size_t size, int smem) {
			void* p;
			auto r = (smem == 0)
				? posix_memalign(&p, alignment, size)
				: memkind_posix_memalign(pmem_kind, (void **)&p, alignment, size);
			if(r || p == nullptr) {
				std::cerr << msg_alloc_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(r)), msg_alloc_fail);
				return nullptr;
			}
			return p;
		}

		void map_memory(void* addr, std::size_t size, std::size_t offset, int prot, int smem) {
			struct memkind_pmem *priv = (memkind_pmem*)pmem_kind->priv;
			auto p = (smem == 0)
				? ::mmap(addr, size, prot, MAP_SHARED|MAP_FIXED, fd_smem, offset)
				: ::mmap(addr, size, prot, MAP_SHARED|MAP_FIXED, priv->fd, offset);
			if(p == MAP_FAILED) {
				std::cerr << msg_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_mmap_fail);
				exit(EXIT_FAILURE);
			}
		}
	} // namespace virtual_memory
} // namespace argo
