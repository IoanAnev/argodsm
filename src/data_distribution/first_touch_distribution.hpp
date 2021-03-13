/**
 * @file
 * @brief This file implements the first-touch data distribution
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#ifndef argo_first_touch_distribution_hpp
#define argo_first_touch_distribution_hpp argo_first_touch_distribution_hpp

#include <mutex>

#include "base_distribution.hpp"

/**
 * @note backend.hpp is not included here as it includes global_ptr.hpp,
 *       which means that the data distribution definitions precede the
 *       backend definitions, and that is why we need to forward here.
 */
namespace argo {
	/* forward argo::backend function declarations */
	namespace backend {
		node_id_t node_id();
		/* forward argo::backend::atomic function declarations */
		namespace atomic {
			void _store_public_owners_dir(const void* desired,
				const std::size_t size, const std::size_t rank, const std::size_t disp);
			void _store_local_owners_dir(const std::size_t desired,
				const std::size_t rank, const std::size_t disp);
			void _store_local_offsets_tbl(const std::size_t desired,
				const std::size_t rank, const std::size_t disp);
			void _load_local_owners_dir(void* output_buffer,
				const std::size_t rank, const std::size_t disp);
			void _load_local_offsets_tbl(void* output_buffer,
				const std::size_t rank, const std::size_t disp);
			void _compare_exchange_owners_dir(const void* desired, const void* expected, void* output_buffer,
				const std::size_t size, const std::size_t rank, const std::size_t disp);
			void _compare_exchange_offsets_tbl(const void* desired, const void* expected, void* output_buffer,
				const std::size_t size, const std::size_t rank, const std::size_t disp);
		} // namespace atomic
	} // namespace backend
} // namespace argo

namespace argo {
	namespace data_distribution {
		/**
		 * @brief the first-touch data distribution
		 * @details gives ownership of a page to the node that first touched it.
		 * @note if a node's backing store size is not sufficient, it passes the
		 *       ownership of the page to a node that can host it.
		 */
		template<int instance>
		class first_touch_distribution : public base_distribution<instance> {
			private:
				/**
				 * @brief try and claim ownership of a page
				 * @param addr address in the global address space
				 */
				static void first_touch (const std::size_t& addr);
				/** @brief protects the owners directory */
				std::mutex owners_mutex;

			public:
				virtual node_id_t homenode (char* const ptr) {
					std::size_t homenode, ownership;
					static const std::size_t rank = argo::backend::node_id();
					static const std::size_t global_null = base_distribution<instance>::total_size + 1;
					const std::size_t addr = (ptr - base_distribution<instance>::start_address) / granularity * granularity;
					const std::size_t owners_dir_window_index = 3 * (addr / granularity);

					std::unique_lock<std::mutex> def_lock(owners_mutex, std::defer_lock);
					def_lock.lock();
					/* fetch the ownership value for the page from the local window */
					argo::backend::atomic::_load_local_owners_dir(&ownership, rank, owners_dir_window_index+2);
					/* if there has been no attempt in claiming this page, try to claim it */
					if (ownership == global_null) { first_touch(addr); }
					/* spin in case an update from a remote node has not been reflected in the local window */
					do {
						argo::backend::atomic::_load_local_owners_dir(&homenode, rank, owners_dir_window_index);
					} while (homenode == global_null);
					def_lock.unlock();

					if(homenode >= static_cast<std::size_t>(base_distribution<instance>::nodes)) {
						exit(EXIT_FAILURE);
					}
					return homenode;
				}

				virtual std::size_t local_offset (char* const ptr) {
					std::size_t offset;
					static const std::size_t rank = argo::backend::node_id();
					static const std::size_t global_null = base_distribution<instance>::total_size + 1;
					const std::size_t drift = (ptr - base_distribution<instance>::start_address) % granularity;
					const std::size_t addr = (ptr - base_distribution<instance>::start_address) / granularity * granularity;
					const std::size_t owners_dir_window_index = 3 * (addr / granularity);

					std::unique_lock<std::mutex> def_lock(owners_mutex, std::defer_lock);
					def_lock.lock();
					/* spin in case an update from a remote node has not been reflected in the local window */
					do {
						argo::backend::atomic::_load_local_owners_dir(&offset, rank, owners_dir_window_index+1);
					} while (offset == global_null);
					def_lock.unlock();
					offset += drift;

					if(offset >= base_distribution<instance>::size_per_node) {
						exit(EXIT_FAILURE);
					}
					return offset;
				}
		};

		template<int instance>
		void first_touch_distribution<instance>::first_touch (const std::size_t& addr) {
			/* variables for CAS */
			std::size_t offset, homenode, result;
			static const std::size_t rank = argo::backend::node_id();
			static const std::size_t global_null = base_distribution<instance>::total_size + 1;
			const std::size_t owners_dir_window_index = 3 * (addr / granularity);
			/* decentralize CAS */
			const std::size_t cas_node = (addr / granularity) % base_distribution<instance>::nodes;
	    
			/* check/try to acquire ownership of the page with CAS to process' cas_node index */
			argo::backend::atomic::_compare_exchange_owners_dir(&rank, &global_null, &result, sizeof(std::size_t), cas_node, owners_dir_window_index+2);

			/* this process was the first one to deposit its rank */
			if (result == global_null) {
				/* iterate through the nodes to find a valid offset for the page, starting from the currently running one */
				std::size_t n, searched;
				for(n = rank, searched = 0; searched < static_cast<std::size_t>(base_distribution<instance>::nodes);
						n = (n + 1) % base_distribution<instance>::nodes, searched++) {
					bool succeeded = false;
					/* load backing offset for the node from the offsets table */
					argo::backend::atomic::_load_local_offsets_tbl(&offset, rank, n);
					/* check if the backing store size of this node is sufficient */
					while (offset < base_distribution<instance>::size_per_node) {
						/* try to claim a valid offset */
						const std::size_t incr_offset = offset + granularity;
						argo::backend::atomic::_compare_exchange_offsets_tbl(&incr_offset, &offset, &result, sizeof(std::size_t), n, n);
						if (result == offset) { succeeded = true; homenode = n; break; }
						offset = result;
					}
					/* cache the offset returned from a remote CAS operation */
					if (n != rank) {
						argo::backend::atomic::_store_local_offsets_tbl(offset, rank, n);
					}
					/* succeeded, leave */
					if (succeeded) break;
				}

				/* mark the page in the local window */
				argo::backend::atomic::_store_local_owners_dir(homenode, rank, owners_dir_window_index);
				argo::backend::atomic::_store_local_owners_dir(offset, rank, owners_dir_window_index+1);
				argo::backend::atomic::_store_local_owners_dir(rank, rank, owners_dir_window_index+2);

				/* mark the page in the public windows */
				for(n = 0; n < static_cast<std::size_t>(base_distribution<instance>::nodes); n++) {
					if (n != rank) {
						argo::backend::atomic::_store_public_owners_dir(&homenode, sizeof(std::size_t), n, owners_dir_window_index);
						argo::backend::atomic::_store_public_owners_dir(&offset, sizeof(std::size_t), n, owners_dir_window_index+1);
						argo::backend::atomic::_store_public_owners_dir(&rank, sizeof(std::size_t), n, owners_dir_window_index+2);
					}
				}
			}
		}
	} // namespace data_distribution
} // namespace argo

#endif /* argo_first_touch_distribution_hpp */
