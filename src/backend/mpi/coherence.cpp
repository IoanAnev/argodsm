/**
 * @file
 * @brief This file implements selective coherence mechanisms for ArgoDSM
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#include "../backend.hpp"
#include "swdsm.h"
#include "write_buffer.hpp"
#include "mpi_mutex.hpp"
#include "virtual_memory/virtual_memory.hpp"
#include <vector>

// EXTERNAL VARIABLES FROM BACKEND
/**
 * @brief This is needed to access page information from the cache
 * @deprecated Should be replaced with a cache API
 */
extern control_data *cacheControl;
/**
 * @brief pyxis_dir is needed to access and modify the pyxis directory
 * @deprecated Should eventually be handled by a cache module
 */
extern std::uint64_t *pyxis_dir;
/**
 * @brief A vector containing cache locks
 * @deprecated Should eventually be handled by a cache module
 */
extern std::vector<cache_lock> cache_locks;
/**
 * @brief A sync lock that acquires shared read or exclusive
 * write access to the whole cache.
 */
extern pthread_rwlock_t sync_lock;
/**
 * @brief sharer locks that protect concurrent access from the same node
 * @deprecated Should be done in a cache module
 */
extern mpi_mutex **mpi_mutex_sharer;
/**
 * @brief Needed to update argo statistics
 * @deprecated Should be replaced by API calls to a stats module
 */
extern argo_statistics stats;
/**
 * @brief Needed to update information about cache pages touched
 * @deprecated Should eventually be handled by a cache module
 */
extern argo_byte *touchedcache;
/**
 * @brief workcomm is needed to poke the MPI system during one sided RMA
 */
extern MPI_Comm workcomm;
/**
 * @brief Write buffer to ensure selectively handled pages can be removed
 * @deprecated This should eventually be handled by a cache module
 */
extern std::vector<write_buffer<std::size_t>> argo_write_buffer;

namespace argo {
	namespace backend {
		void _selective_acquire(void *addr, std::size_t size){
			// Skip selective acquire if the size of the region is 0
			if(size == 0){
				return;
			}

			const std::size_t block_size = page_size*CACHELINE;
			const std::size_t start_address = reinterpret_cast<std::size_t>(argo::virtual_memory::start_address());
			const std::size_t page_misalignment = reinterpret_cast<std::size_t>(addr)%block_size;
			std::size_t argo_address =
				((reinterpret_cast<std::size_t>(addr)-start_address)/block_size)*block_size;
			const node_id_t node_id = argo::backend::node_id();
			const std::size_t node_id_bit = static_cast<std::size_t>(1) << node_id;

			// Lock relevant mutexes. Start statistics timekeeping
			double t1 = MPI_Wtime();
			pthread_rwlock_rdlock(&sync_lock);

			// Iterate over all pages to selectively invalidate
			for(std::size_t page_address = argo_address;
					page_address < argo_address + page_misalignment + size;
					page_address += block_size){
				const node_id_t homenode_id = peek_homenode(page_address);
				// This page should be skipped in the following cases
				// 1. The page is node local so no acquire is necessary
				// 2. The page has not yet been first-touched and trying
				// to perform an acquire would first-touch the page
				if(	homenode_id == node_id ||
					homenode_id == argo::data_distribution::invalid_node_id) {
					continue;
				}

				const std::size_t cache_index = getCacheIndex(page_address);
				const std::size_t classification_index = get_classification_index(page_address);
				cache_locks[cache_index].lock();

				// If the page is dirty, downgrade it
				if(cacheControl[cache_index].dirty == DIRTY){
					mprotect((char*)start_address + page_address, block_size, PROT_READ);
					for(int i = 0; i <CACHELINE; i++){
						store_page_diff(cache_index+i,page_address+page_size*i);
					}
					argo_write_buffer[get_write_buffer(cache_index)].erase(cache_index);
					cacheControl[cache_index].dirty = CLEAN;
				}
				// Make sure to sync writebacks
				unlock_windows();

				// Optimization to keep pages in cache if they do not
				// need to be invalidated.
				mpi_mutex_sharer[node_id]->lock_shared();
				if(
						// node is single writer
						(pyxis_dir[classification_index+1] == node_id_bit)
						||
						// No writer and assert that the node is a sharer
						((pyxis_dir[classification_index+1] == 0) &&
						 ((pyxis_dir[classification_index] & node_id_bit) == node_id_bit))
				  ){
					mpi_mutex_sharer[node_id]->unlock_shared();
					touchedcache[cache_index]=1;
					//nothing - we keep the pages, SD is done in flushWB
				}
				else{ //multiple writer or SO, invalidate the page
					mpi_mutex_sharer[node_id]->unlock_shared();
					cacheControl[cache_index].dirty=CLEAN;
					cacheControl[cache_index].state = INVALID;
					touchedcache[cache_index]=0;
					mprotect((char*)start_address + page_address, block_size, PROT_NONE);
				}
				cache_locks[cache_index].unlock();
			}
			double t2 = MPI_Wtime();

			// Poke the MPI system to force progress
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,workcomm,&flag,MPI_STATUS_IGNORE);

			// Release relevant mutexes
			pthread_rwlock_unlock(&sync_lock);

			std::lock_guard<std::mutex> ssi_time_lock(stats.ssi_time_mutex);
			stats.ssi_time += t2-t1;
		}

		void _selective_release(void *addr, std::size_t size){
			// Skip selective release if the size of the region is 0
			if(size == 0){
				return;
			}

			const std::size_t block_size = page_size*CACHELINE;
			const std::size_t start_address = reinterpret_cast<std::size_t>(argo::virtual_memory::start_address());
			const std::size_t page_misalignment = reinterpret_cast<std::size_t>(addr)%block_size;
			std::size_t argo_address =
				((reinterpret_cast<std::size_t>(addr)-start_address)/block_size)*block_size;
			const node_id_t node_id = argo::backend::node_id();

			// Lock relevant mutexes. Start statistics timekeeping
			double t1 = MPI_Wtime();
			pthread_rwlock_rdlock(&sync_lock);

			// Iterate over all pages to selectively downgrade
			for(std::size_t page_address = argo_address;
					page_address < argo_address + page_misalignment + size;
					page_address += block_size){
				const node_id_t homenode_id = peek_homenode(page_address);
				// selective_release should be skipped in the following cases
				// 1. The page is node local so no release is necessary
				// 2. The page has not yet been first-touched and trying
				// to perform a release would first-touch the page
				if(	homenode_id == node_id ||
					homenode_id == argo::data_distribution::invalid_node_id) {
					continue;
				}

				const std::size_t cache_index = getCacheIndex(page_address);
				cache_locks[cache_index].lock();

				// If the page is dirty, downgrade it
				if(cacheControl[cache_index].dirty == DIRTY){
					mprotect((char*)start_address + page_address, block_size, PROT_READ);
					for(int i = 0; i <CACHELINE; i++){
						store_page_diff(cache_index+i,page_address+page_size*i);
					}
					argo_write_buffer[get_write_buffer(cache_index)].erase(cache_index);
					cacheControl[cache_index].dirty = CLEAN;
				}
				// Make sure to sync writebacks
				unlock_windows();
				cache_locks[cache_index].unlock();
			}
			double t2 = MPI_Wtime();

			// Poke the MPI system to force progress
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,workcomm,&flag,MPI_STATUS_IGNORE);

			// Release relevant mutexes
			pthread_rwlock_unlock(&sync_lock);

			std::lock_guard<std::mutex> ssd_time_lock(stats.ssd_time_mutex);
			stats.ssd_time += t2-t1;
		}
	} //namespace backend
} //namespace argo
