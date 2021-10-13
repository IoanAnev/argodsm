/**
 * @file
 * @brief This file implements the MPI-backend of ArgoDSM
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */
#include<cstddef>
#include<vector>
#include <memory>
#include <numeric>

#include "env/env.hpp"
#include "signal/signal.hpp"
#include "virtual_memory/virtual_memory.hpp"
#include "data_distribution/global_ptr.hpp"
#include "swdsm.h"
#include "write_buffer.hpp"
#include "mpi_lock.hpp"

namespace dd = argo::data_distribution;
namespace vm = argo::virtual_memory;
namespace sig = argo::signal;
namespace env = argo::env;

/** @brief For matching threads to more sensible thread IDs */
pthread_t tid[NUM_THREADS] = {0};

/*Barrier*/
/** @brief  Locks access to part that does SD in the global barrier */
pthread_mutex_t barriermutex = PTHREAD_MUTEX_INITIALIZER;
/** @brief Thread local barrier used to first wait for all local threads in the global barrier*/
pthread_barrier_t *threadbarrier;


/*Pagecache*/
/** @brief  Size of the cache in number of pages*/
std::size_t cachesize;
/** @brief  The maximum number of pages load_cache_entry will fetch remotely */
std::size_t load_size;
/** @brief  Offset off the cache in the backing file*/
std::size_t cacheoffset;
/** @brief  Keeps state, tag and dirty bit of the cache*/
control_data * cacheControl;
/** @brief  keeps track of readers and writers*/
std::uint64_t *globalSharers;
/** @brief  size of pyxis directory*/
std::size_t classificationSize;
/** @brief  Tracks if a page is touched this epoch*/
argo_byte * touchedcache;
/** @brief  The local page cache*/
char* cacheData;
/** @brief Copy of the local cache to keep twinpages for later being able to DIFF stores */
char * pagecopy;
/** @brief The number of pages protected by an MPI Window */
std::size_t win_granularity;
/** @brief Pointer to locks protecting the page cache */
std::vector<cache_lock> cache_locks;
/** @brief Mutex ensuring that only one thread can perform sync */
pthread_rwlock_t sync_lock = PTHREAD_RWLOCK_INITIALIZER;

/*Writebuffer*/
/** @brief A write buffer storing cache indices */
std::vector<write_buffer<std::size_t>> argo_write_buffer;

/*MPI and Comm*/
/** @brief  A copy of MPI_COMM_WORLD group to split up processes into smaller groups*/
/** @todo This can be removed now when we are only running 1 process per ArgoDSM node */
MPI_Group startgroup;
/** @brief  A group of all processes that are executing the main thread */
/** @todo This can be removed now when we are only running 1 process per ArgoDSM node */
MPI_Group workgroup;
/** @brief Communicator can be replaced with MPI_COMM_WORLD*/
MPI_Comm workcomm;
/** @brief MPI window for communicating pyxis directory*/
std::vector<std::vector<MPI_Win>> sharer_windows;
/** @brief The number of sharer windows (first dimension) */
std::size_t num_sharer_windows;
/** @brief MPI windows for reading and writing data in global address space */
std::vector<std::vector<MPI_Win>> data_windows;
/** @brief The number of data windows (first dimension) */
std::size_t num_data_windows;
/**
 * @brief Mutex to protect concurrent access to same window from within node
 * @note  First index corresponds to window, second to remote node
 **/
mpi_lock **mpi_lock_sharer;
/**
 * @brief Mutex to protect concurrent access to same window from within node
 * @note  First index corresponds to window, second to remote node
 **/
mpi_lock **mpi_lock_data;
/**
 * @brief Keep track of locked windows
 */
thread_local std::vector<std::pair<int, int>> locked_windows;
/** @brief MPI data structure for sending cache control data*/
MPI_Datatype mpi_control_data;
/** @brief MPI data structure for a block containing an ArgoDSM cacheline of pages */
MPI_Datatype cacheblock;
/** @brief number of MPI processes / ArgoDSM nodes */
int numtasks;
/** @brief  rank/process ID in the MPI/ArgoDSM runtime*/
int rank;
/** @brief rank/process ID in the MPI/ArgoDSM runtime*/
argo::node_id_t workrank;
/** @brief tracking which windows are used for reading and writing global address space*/

/*Loading and Prefetching*/
/**
 * @brief load into cache helper function
 * @param aligned_access_offset memory offset to load into the cache
 * @pre aligned_access_offset must be aligned as CACHELINE*pagesize
 */
void load_cache_entry(std::size_t aligned_access_offset);

//void sharer_op(int lock_type, int rank, int offset,
//		std::function<void(const std::size_t win_index)> op);
//
//std::size_t get_sharer_win_index(int offset);


/*Common*/
/** @brief  Points to start of global address space*/
void* startAddr;
/** @brief  Points to start of global address space this process is serving */
char* globalData;
/** @brief  Size of global address space*/
std::size_t size_of_all;
/** @brief  Size of this process part of global address space*/
std::size_t size_of_chunk;
/** @brief  size of a page */
static const unsigned int pagesize = 4096;
/** @brief  Magic value for invalid cacheindices */
std::uintptr_t GLOBAL_NULL;
/** @brief  Statistics */
argo_statistics stats;

/*First-Touch policy*/
/** @brief  Holds the owner and backing offset of a page */
std::uintptr_t *global_owners_dir;
/** @brief  Holds the backing offsets of the nodes */
std::uintptr_t *global_offsets_tbl;
/** @brief  Size of the owners directory */
std::size_t owners_dir_size;
/** @brief  MPI window for communicating owners directory */
MPI_Win owners_dir_window;
/** @brief  MPI window for communicating offsets table */
MPI_Win offsets_tbl_window;
/** @brief  Spinlock to avoid "spinning" on the semaphore */
std::mutex spin_mutex;

namespace {
	/** @brief constant for invalid ArgoDSM node */
	constexpr std::uint64_t invalid_node = static_cast<std::uint64_t>(-1);
}

std::size_t isPowerOf2(std::size_t x){
	std::size_t retval =  ((x & (x - 1)) == 0); //Checks if x is power of 2 (or zero)
	return retval;
}

int argo_get_local_tid(){
	int i;
	for(i = 0; i < NUM_THREADS; i++){
		if(pthread_equal(tid[i],pthread_self())){
			return i;
		}
	}
	return 0;
}

int argo_get_global_tid(){
	int i;
	for(i = 0; i < NUM_THREADS; i++){
		if(pthread_equal(tid[i],pthread_self())){
			return ((getID()*NUM_THREADS) + i);
		}
	}
	return 0;
}


void argo_register_thread(){
	int i;
	for(i = 0; i < NUM_THREADS; i++){
		if(tid[i] == 0){
			tid[i] = pthread_self();
			break;
		}
	}
	pthread_barrier_wait(&threadbarrier[NUM_THREADS]);
}


void argo_pin_threads(){

  cpu_set_t cpuset;
  int s;
  argo_register_thread();
  CPU_ZERO(&cpuset);
  int pinto = argo_get_local_tid();
  CPU_SET(pinto, &cpuset);

  s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (s != 0){
    printf("PINNING ERROR\n");
    argo_finalize();
  }
}


std::size_t getCacheIndex(std::uintptr_t addr){
	std::size_t index = (addr/pagesize) % cachesize;
	return index;
}

void init_mpi_struct(void){
	//init our struct coherence unit to work in mpi.
	const int blocklen[3] = { 1,1,1};
	MPI_Aint offsets[3];
	offsets[0] = 0;  offsets[1] = sizeof(argo_byte)*1;  offsets[2] = sizeof(argo_byte)*2;

	MPI_Datatype types[3] = {MPI_BYTE,MPI_BYTE,MPI_UNSIGNED_LONG};
	MPI_Type_create_struct(3,blocklen, offsets, types, &mpi_control_data);

	MPI_Type_commit(&mpi_control_data);
}


void init_mpi_cacheblock(void){
	//init our struct coherence unit to work in mpi.
	MPI_Type_contiguous(pagesize*CACHELINE,MPI_BYTE,&cacheblock);
	MPI_Type_commit(&cacheblock);
}

/**
 * @brief align an offset into a memory region to the beginning of its size block
 * @param offset the unaligned offset
 * @param size the size of each block
 * @return the beginning of the block of size size where offset is located
 */
inline std::size_t align_backwards(std::size_t offset, std::size_t size) {
	return (offset / size) * size;
}

void handler(int sig, siginfo_t *si, void *context){
	UNUSED_PARAM(sig);
#ifndef REG_ERR
	UNUSED_PARAM(context);
#endif /* REG_ERR */
	double t1 = MPI_Wtime();
	std::uintptr_t tag;
	argo_byte owner,state;

	/* compute offset in distributed memory in bytes, always positive */
	const std::size_t access_offset = static_cast<char*>(si->si_addr) - static_cast<char*>(startAddr);

	/* The type of miss triggering the handler is unknown */
	sig::access_type miss_type = sig::access_type::undefined;
#ifdef REG_ERR
	/* On x86, get and decode the error number from the context */
	const ucontext_t* ctx = static_cast<ucontext_t*>(context);
	auto err_num = ctx->uc_mcontext.gregs[REG_ERR];
	assert(err_num & X86_PF_USER);	//Assert signal from user space
	assert(err_num < X86_PF_RSVD); 	//Assert signal is from read or write access
	/* This could be further decoded by using X86_PF_PROT to detect
	 * whether the fault originated from no page found (0) or from
	 * a protection fault (1), but is not needed for this purpose. */
	/* Assign correct type to the miss */
	miss_type = (err_num & X86_PF_WRITE) ? sig::access_type::write : sig::access_type::read;
#endif /* REG_ERR */

	/* align access offset to cacheline */
	const std::size_t aligned_access_offset = align_backwards(access_offset, CACHELINE*pagesize);
	std::size_t classidx = get_classification_index(aligned_access_offset);

	/* compute start pointer of cacheline. char* has byte-wise arithmetics */
	char* const aligned_access_ptr = static_cast<char*>(startAddr) + aligned_access_offset;
	std::size_t startIndex = getCacheIndex(aligned_access_offset);

	/* Get homenode and offset, protect with ibsem if first touch */
	argo::node_id_t homenode;
	std::size_t offset;
	if(dd::is_first_touch_policy()){
		std::lock_guard<std::mutex> lock(spin_mutex);
		homenode = get_homenode(aligned_access_offset);
		offset = get_offset(aligned_access_offset);
	}else{
		homenode = get_homenode(aligned_access_offset);
		offset = get_offset(aligned_access_offset);
	}

	std::uint64_t id = static_cast<std::uint64_t>(1) << getID();
	std::uint64_t invid = ~id;
	std::size_t sharer_win_offset = get_sharer_win_offset(classidx);

	/* Acquire shared sync lock and first cache index lock */
	pthread_rwlock_rdlock(&sync_lock);
	cache_locks[startIndex].lock();

	/* page is local */
	if(homenode == (getID())){
		std::uint64_t sharers, prevsharer;
		{
			auto op = [&](std::size_t) {
				prevsharer = (globalSharers[classidx])&id;
			};
			sharer_op(MPI_LOCK_SHARED, workrank, classidx, op);
		}
		if(prevsharer != id){
			{
				auto op = [&](std::size_t) {
					sharers = globalSharers[classidx];
					globalSharers[classidx] |= id;
				};
				sharer_op(MPI_LOCK_EXCLUSIVE, workrank, classidx, op);
			}
			if(sharers != 0 && sharers != id && isPowerOf2(sharers)){
				std::uint64_t ownid = sharers&invid;
				argo::node_id_t owner = workrank;
				for(argo::node_id_t n = 0; n < numtasks; n++){
					if((static_cast<std::uint64_t>(1)<<n)==ownid){
						owner = n; //just get rank...
						break;
					}
				}
				if(owner==workrank){
					throw "bad owner in local access";
				}
				else{
					/* update remote private holder to shared */
					{
						auto op = [&](std::size_t win_index){
							MPI_Accumulate(&id, 1, MPI_LONG, owner, sharer_win_offset,1,MPI_LONG,MPI_BOR,sharer_windows[win_index][owner]); };
						sharer_op(MPI_LOCK_EXCLUSIVE, owner, classidx, op);
					}
				}
			}
			/* set page to permit reads and map it to the page cache */
			/** @todo Set cache offset to a variable instead of calculating it here */
			vm::map_memory(aligned_access_ptr, pagesize*CACHELINE, cacheoffset+offset, PROT_READ);

		}
		else{
			/* Do not register as writer if this is a confirmed read miss */
			if(miss_type == sig::access_type::read) {
				sem_post(&ibsem);
				pthread_mutex_unlock(&cachemutex);
				return;
			}

			/* get current sharers/writers and then add your own id */
			std::uint64_t sharers, writers;
			{
				auto op = [&](std::size_t){
					sharers = globalSharers[classidx];
					writers = globalSharers[classidx+1];
					globalSharers[classidx+1] |= id;
				};
				sharer_op(MPI_LOCK_EXCLUSIVE, workrank, classidx, op);
			}

			/* remote single writer */
			if(writers != id && writers != 0 && isPowerOf2(writers&invid)){
				for(argo::node_id_t n = 0; n < numtasks; n++){
					if((static_cast<std::uint64_t>(1)<<n)==(writers&invid)){
						owner = n; //just get rank...
						break;
					}
				}
				{
					auto op = [&](std::size_t win_index){
						MPI_Accumulate(&id, 1, MPI_LONG, owner, sharer_win_offset+1,1,MPI_LONG,MPI_BOR,sharer_windows[win_index][owner]);
					};
					sharer_op(MPI_LOCK_EXCLUSIVE, owner, classidx, op);
				}
			}
			else if(writers == id || writers == 0){
				for(argo::node_id_t n = 0; n < numtasks; n++){
					if(n != workrank && ((static_cast<unsigned long>(1)<<n)&sharers) != 0){
						{
							auto op = [&](std::size_t win_index){
								MPI_Accumulate(&id, 1, MPI_LONG, n, sharer_win_offset+1,1,MPI_LONG,MPI_BOR,sharer_windows[win_index][n]);
							};
							sharer_op(MPI_LOCK_EXCLUSIVE, n, classidx, op);
						}
					}
				}
			}

			/* set page to permit read/write and map it to the page cache */
			vm::map_memory(aligned_access_ptr, pagesize*CACHELINE, cacheoffset+offset, PROT_READ|PROT_WRITE);

		}
		/* Unlock shared sync lock and cache index lock */
		cache_locks[startIndex].unlock();
		pthread_rwlock_unlock(&sync_lock);
		return;
	}

	state  = cacheControl[startIndex].state;
	tag = cacheControl[startIndex].tag;
	bool performed_load = false;

	/* Fetch the correct page if necessary */
	if(state == INVALID || (tag != aligned_access_offset && tag != GLOBAL_NULL)) {
		load_cache_entry(aligned_access_offset);
		performed_load = true;
	}

	/* If miss is known to originate from a read access, or if the
	 * access type is unknown but a load has already been performed
	 * in this handler, exit here to avoid false write misses */
	if(miss_type == sig::access_type::read ||
		(miss_type == sig::access_type::undefined && performed_load)) {
		assert(cacheControl[startIndex].state == VALID);
		assert(cacheControl[startIndex].tag == aligned_access_offset);
		//pthread_mutex_unlock(&cachemutex);
		//sync_mutex.unlock_shared();
		pthread_rwlock_unlock(&sync_lock);
		double t2 = MPI_Wtime();
		stats.loadtime+=t2-t1;
		return;
	}

	std::uintptr_t line = startIndex / CACHELINE;
	line *= CACHELINE;

	if(cacheControl[line].dirty == DIRTY){
		cache_locks[startIndex].unlock();
		pthread_rwlock_unlock(&sync_lock);
		return;
	}

	touchedcache[line] = 1;
	cacheControl[line].dirty = DIRTY;

	std::uint64_t writers, sharers;
	{
		auto op = [&](std::size_t){
			writers = globalSharers[classidx+1];
			sharers = globalSharers[classidx];
		};
		sharer_op(MPI_LOCK_SHARED, workrank, classidx, op);
	}

	/* Either already registered write - or 1 or 0 other writers already cached */
	if(writers != id && isPowerOf2(writers)){
		{
			auto op = [&](std::size_t){
				globalSharers[classidx+1] |= id; //register locally
			};
			sharer_op(MPI_LOCK_EXCLUSIVE, workrank, classidx, op);
		}

		/* register and get latest sharers / writers */
		{
			auto op = [&](std::size_t win_index){
				MPI_Get_accumulate(&id, 1,MPI_LONG,&writers,1,MPI_LONG,homenode,
					sharer_win_offset+1,1,MPI_LONG,MPI_BOR,sharer_windows[win_index][homenode]);
				MPI_Get(&sharers,1, MPI_LONG, homenode, sharer_win_offset, 1,MPI_LONG,sharer_windows[win_index][homenode]);
			};
			sharer_op(MPI_LOCK_SHARED, homenode, classidx+1, op);
			//TODO: Test if this can still be SHARED
			//TODO: We can remove one MPI operation here by using a bitmask and storing both
		}
				
		/* We get result of accumulation before operation so we need to account for that */
		writers |= id;
		/* Just add the (potentially) new sharers fetched to local copy */
		{
			auto op = [&](std::size_t){
				globalSharers[classidx] |= sharers;
			};
			sharer_op(MPI_LOCK_EXCLUSIVE, workrank, classidx, op);
		}

		/* check if we need to update */
		if(writers != id && writers != 0 && isPowerOf2(writers&invid)){
			for(argo::node_id_t n = 0; n < numtasks; n++){
				if((static_cast<std::uint64_t>(1)<<n)==(writers&invid)){
					owner = n; //just get rank...
					break;
				}
			}
			sharer_op(MPI_LOCK_EXCLUSIVE, owner, classidx+1, [&](std::size_t win_index){
					MPI_Accumulate(&id, 1, MPI_LONG, owner, sharer_win_offset+1,1,MPI_LONG,MPI_BOR,sharer_windows[win_index][owner]);
					});
		}
		else if(writers==id || writers==0){
			for(argo::node_id_t n = 0; n < numtasks; n++){
				if(n != workrank && ((static_cast<unsigned long>(1)<<n)&sharers) != 0){
					sharer_op(MPI_LOCK_EXCLUSIVE, n, classidx+1, [&](std::size_t win_index){
							MPI_Accumulate(&id, 1, MPI_LONG, n, sharer_win_offset+1,1,MPI_LONG,MPI_BOR,sharer_windows[win_index][n]);
							});
				}
			}
		}
	}
	unsigned char* copy = reinterpret_cast<unsigned char*>(pagecopy + line*pagesize);
	memcpy(copy,aligned_access_ptr,CACHELINE*pagesize);
	mprotect(aligned_access_ptr, pagesize*CACHELINE,PROT_WRITE|PROT_READ);
	cache_locks[startIndex].unlock();
	pthread_rwlock_unlock(&sync_lock);
	double t2 = MPI_Wtime();
	stats.storetime += t2-t1;
	// TODO: Check if this actually needs to be outside
	argo_write_buffer[get_write_buffer(startIndex)].add(startIndex);
	return;
}


argo::node_id_t get_homenode(std::uintptr_t addr){
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<std::uintptr_t>(startAddr)), true, false);
	return gptr.node();
}

argo::node_id_t peek_homenode(std::uintptr_t addr) {
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<std::uintptr_t>(startAddr)), false, false);
	return gptr.peek_node();
}

std::size_t get_offset(std::uintptr_t addr){
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<std::uintptr_t>(startAddr)), false, true);
	return gptr.offset();
}

std::size_t peek_offset(std::uintptr_t addr) {
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<std::uintptr_t>(startAddr)), false, false);
	return gptr.peek_offset();
}

void load_cache_entry(std::uintptr_t aligned_access_offset) {

	/* If it's not an ArgoDSM address, do not handle it */
	if(aligned_access_offset >= size_of_all){
		//TODO: Must probably unlock here?
		printf("WARNING: UNSAFE CASE!!!\n");
		return;
	}

	const std::size_t block_size = pagesize*CACHELINE;
	/* Check that the precondition holds true */
	assert((aligned_access_offset % block_size) == 0);

	/* Assign node bit IDs */
	const std::uint64_t node_id_bit = static_cast<std::uint64_t>(1) << getID();
	const std::uint64_t node_id_inv_bit = ~node_id_bit;

	/* Calculate start values and store some parameters */
	const std::size_t cache_index = getCacheIndex(aligned_access_offset);
	const std::size_t start_index = align_backwards(cache_index, CACHELINE);
	std::size_t end_index = start_index+CACHELINE;
	const argo::node_id_t load_node = get_homenode(aligned_access_offset);
	const std::size_t load_offset = get_offset(aligned_access_offset);
	const std::size_t load_sharer_win_index = get_sharer_win_index(
			get_classification_index(aligned_access_offset));
	const std::size_t load_data_win_index = get_data_win_index(load_offset);


	/* Return if requested cache entry is already up to date. */
	if(cacheControl[start_index].tag == aligned_access_offset &&
			cacheControl[start_index].state != INVALID){
		cache_locks[start_index].unlock();
		return;
	}

	/* Adjust end_index to ensure the whole chunk to fetch is on the same node */
	for(std::size_t i = start_index+CACHELINE, p = CACHELINE;
					i < start_index+load_size;
					i+=CACHELINE, p+=CACHELINE){
		const std::uintptr_t temp_addr = aligned_access_offset + p*block_size;
		/* Increase end_index if it is within bounds and on the same node */
		if(temp_addr < size_of_all && i < cachesize){
			const argo::node_id_t temp_node = peek_homenode(temp_addr);
			const std::size_t temp_offset = peek_offset(temp_addr);
			const std::size_t temp_sharer_win_index = get_sharer_win_index(
					get_classification_index(temp_addr));
			const std::size_t temp_data_win_index = get_data_win_index(temp_offset);

			if(temp_node == load_node &&
					temp_offset == (load_offset + p*block_size) &&
					temp_sharer_win_index == load_sharer_win_index &&
					temp_data_win_index == load_data_win_index){
				end_index+=CACHELINE;
			}else{
				break;
			}
		}else{
			/* Stop when either condition is not satisfied */
			break;
		}
	}

	bool new_sharer = false;
	const std::size_t fetch_size = end_index - start_index;
	const std::size_t classification_size = fetch_size*2;

	/* For each page to load, true if page should be cached else false */
	std::vector<bool> pages_to_load(fetch_size);
	/* For each page to update in the cache, true if page has
	 * already been handled else false */
	std::vector<bool> handled_pages(fetch_size);
	/* Contains classification index for each page to load */
	std::vector<std::size_t> classification_index_array(fetch_size);
	/* Store sharer state from local node temporarily */
	std::vector<std::uintptr_t> local_sharers(fetch_size);
	/* Store content of remote Pyxis directory temporarily */
	std::vector<std::uintptr_t> remote_sharers(classification_size);
	/* Store updates to be made to remote Pyxis directory */
	std::vector<std::uintptr_t> sharer_bit_mask(classification_size);
	/* Temporarily store remotely fetched cache data */
	std::vector<char> temp_data(fetch_size*pagesize);

	/* Write back existing cache entries if needed */
	for(std::size_t idx = start_index, p = 0; idx < end_index; idx+=CACHELINE, p+=CACHELINE){
		/* Address and pointer to the data being loaded */
		const std::size_t temp_addr = aligned_access_offset + p*block_size;

		if(cache_locks[idx].try_lock() || idx == start_index){
			/* Skip updating pages that are already present and valid in the cache */
			if(cacheControl[idx].tag == temp_addr && cacheControl[idx].state != INVALID){
				pages_to_load[p] = false;
				cache_locks[idx].unlock();
				continue;
			}else{
				pages_to_load[p] = true;
			}
		}else{
			pages_to_load[p] = false;
			continue;
		}

		/* If another page occupies the cache index, begin to evict it. */
		if((cacheControl[idx].tag != temp_addr) && (cacheControl[idx].tag != GLOBAL_NULL)){
			void* old_ptr = static_cast<char*>(startAddr) + cacheControl[idx].tag;
			void* temp_ptr = static_cast<char*>(startAddr) + temp_addr;

			/* If the page is dirty, write it back */
			if(cacheControl[idx].dirty == DIRTY){
				mprotect(old_ptr,block_size,PROT_READ);
				for(std::size_t j = 0; j < CACHELINE; j++){
					storepageDIFF(idx+j,pagesize*j+(cacheControl[idx].tag));
				}
				argo_write_buffer[get_write_buffer(idx)].erase(idx);
			}
			/* Ensure the writeback has finished */
			unlock_windows();

			/* Clean up cache and protect memory */
			cacheControl[idx].state = INVALID;
			cacheControl[idx].tag = temp_addr;
			cacheControl[idx].dirty = CLEAN;
			vm::map_memory(temp_ptr, block_size, pagesize*idx, PROT_NONE);
			mprotect(old_ptr,block_size,PROT_NONE);
		}
	}

	/* Initialize classification_index_array */
	for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
		const std::size_t temp_addr = aligned_access_offset + i*block_size;
		classification_index_array[i] = get_classification_index(temp_addr);
	}

	/* Increase stat counter as load will be performed */
	stats.loads++;

	/* Get globalSharers info from local node and add self to it */
	sharer_op(MPI_LOCK_SHARED, workrank, classification_index_array[0],
			[&](std::size_t){
		for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
			if(pages_to_load[i]){
				/* Check local pyxis directory if we are sharer of the page */
				local_sharers[i] = (globalSharers[classification_index_array[i]])&node_id_bit;
				if(local_sharers[i] == 0){
					sharer_bit_mask[i*2] = node_id_bit;
					new_sharer = true; //At least one new sharer detected
				}
			}
		}
	});

	std::size_t sharer_win_offset = get_sharer_win_offset(classification_index_array[0]);
	/* If this node is a new sharer of at least one of the pages */
	if(new_sharer){
		/* Register this node as sharer of all newly shared pages in the load_node's
		 * globalSharers directory using one MPI call. When this call returns,
		 * remote_sharers contains remote globalSharers directory values prior to
		 * this call. */
		sharer_op(MPI_LOCK_SHARED, load_node, classification_index_array[0],
				[&](std::size_t win_index){
				MPI_Get_accumulate(sharer_bit_mask.data(), classification_size, MPI_LONG,
						remote_sharers.data(), classification_size, MPI_LONG,
						load_node, sharer_win_offset, classification_size,
						MPI_LONG, MPI_BOR, sharer_windows[win_index][load_node]);
				});
	}

	/* Register the received remote globalSharers information locally */
	sharer_op(MPI_LOCK_EXCLUSIVE, workrank, classification_index_array[0],
			[&](std::size_t){
		for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
			if(pages_to_load[i]){
						globalSharers[classification_index_array[i]] |= remote_sharers[i*2];
						globalSharers[classification_index_array[i]] |= node_id_bit; //Also add self
						globalSharers[classification_index_array[i]+1] |= remote_sharers[(i*2)+1];
			}
		}
	});

	/* If any owner of a page we loaded needs to downgrade from private
	 * to shared, we need to notify it */
	for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
		/* Skip pages that are not loaded or already handled */
		if(pages_to_load[i] && !handled_pages[i]){
			std::fill(sharer_bit_mask.begin(), sharer_bit_mask.end(), 0);
			const std::uintptr_t owner_id_bit =
				remote_sharers[i*2]&node_id_inv_bit; // remove own bit

			/* If there is exactly one other owner, and we are not sharer */
			if(isPowerOf2(owner_id_bit) && owner_id_bit != 0 && local_sharers[i] == 0){
				std::uintptr_t owner = invalid_node; // initialize to failsafe value
				for(int n = 0; n < numtasks; n++) {
					if((static_cast<std::uintptr_t>(1)<<n)==owner_id_bit) {
						owner = n; //just get rank...
						break;
					}
				}
				sharer_bit_mask[i*2] = node_id_bit;

				/* Check if any of the remaining pages need downgrading on the same node */
				for(std::size_t j = i+CACHELINE; j < fetch_size; j+=CACHELINE){
					if(pages_to_load[j] && !handled_pages[j]){
						if((remote_sharers[j*2]&node_id_inv_bit) == owner_id_bit &&
								local_sharers[j] == 0){
							sharer_bit_mask[j*2] = node_id_bit;
							handled_pages[j] = true; //Ensure these are marked as completed
						}
					}
				}

				/* Downgrade all relevant pages on the owner node from private to shared */
				sharer_op(MPI_LOCK_EXCLUSIVE, owner, classification_index_array[0],
						[&](std::size_t win_index){
						MPI_Accumulate(sharer_bit_mask.data(), classification_size, MPI_LONG, owner,
								sharer_win_offset, classification_size, MPI_LONG,
								MPI_BOR, sharer_windows[win_index][owner]);
					});
			}
		}
	}

	/* Finally, get the cache data and store it temporarily */
	std::size_t win_index = get_data_win_index(load_offset);
	std::size_t win_offset = get_data_win_offset(load_offset);
	mpi_lock_data[win_index][load_node].lock(MPI_LOCK_SHARED, load_node, data_windows[win_index][load_node]);
	MPI_Get(temp_data.data(), fetch_size, cacheblock,
					load_node, win_offset, fetch_size, cacheblock, data_windows[win_index][load_node]);
	mpi_lock_data[win_index][load_node].unlock(load_node, data_windows[win_index][load_node]);

	/* Update the cache */
	for(std::size_t idx = start_index, p = 0; idx < end_index; idx+=CACHELINE, p+=CACHELINE){
		/* Update only the pages necessary */
		if(pages_to_load[p]){
			/* Insert the data in the node cache */
			memcpy(&cacheData[idx*block_size], &temp_data[p*block_size], block_size);

			const std::size_t temp_addr = aligned_access_offset + p*block_size;
			void* temp_ptr = static_cast<char*>(startAddr) + temp_addr;

			/* If this is the first time inserting in to this index, perform vm map */
			if(cacheControl[idx].tag == GLOBAL_NULL){
				vm::map_memory(temp_ptr, block_size, pagesize*idx, PROT_READ);
				cacheControl[idx].tag = temp_addr;
			}else{
				/* Else, just mprotect the region */
				mprotect(temp_ptr, block_size, PROT_READ);
			}
			touchedcache[idx] = 1;
			cacheControl[idx].state = VALID;
			cacheControl[idx].dirty=CLEAN;
			cache_locks[idx].unlock();
		}
	}
}


void initmpi(){
	int ret,initialized,thread_status;
	int thread_level = MPI_THREAD_MULTIPLE;
	MPI_Initialized(&initialized);
	if (!initialized){
		ret = MPI_Init_thread(NULL,NULL,thread_level,&thread_status);
	}
	else{
		printf("MPI was already initialized before starting ArgoDSM - shutting down\n");
		exit(EXIT_FAILURE);
	}

	if (ret != MPI_SUCCESS || thread_status != thread_level) {
		printf ("MPI not able to start properly\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
		exit(EXIT_FAILURE);
	}

	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	init_mpi_struct();
	init_mpi_cacheblock();
}

argo::node_id_t getID(){
	return workrank;
}
argo::node_id_t argo_get_nid(){
	return workrank;
}

unsigned int argo_get_nodes(){
	return numtasks;
}
unsigned int getThreadCount(){
	return NUM_THREADS;
}

/**
 * @brief aligns an offset (into a memory region) to the beginning of its
 * subsequent size block if it is not already aligned to a size block.
 * @param offset the unaligned offset
 * @param size the size of each block
 * @return the aligned offset
 */
std::size_t align_forwards(std::size_t offset, std::size_t size){
	return (offset == 0) ? offset : (1 + ((offset-1) / size))*size;
}

void argo_initialize(std::size_t argo_size, std::size_t cache_size){
	initmpi();
	double init_start = MPI_Wtime();

	/** Standardise the ArgoDSM memory space */
	argo_size = std::max(argo_size, static_cast<std::size_t>(pagesize*numtasks));
	argo_size = align_forwards(argo_size, pagesize*CACHELINE*numtasks*dd::policy_padding());

	startAddr = vm::start_address();
#ifdef ARGO_PRINT_STATISTICS
	printf("maximum virtual memory: %ld GiB\n", vm::size() >> 30);
#endif

	threadbarrier = (pthread_barrier_t *) malloc(sizeof(pthread_barrier_t)*(NUM_THREADS+1));
	for(std::size_t i = 1; i <= NUM_THREADS; i++){
		pthread_barrier_init(&threadbarrier[i],NULL,i);
	}

	/** Get the number of pages to load from the env module */
	load_size = env::load_size();
	/** Limit cache_size to at most argo_size */
	cachesize = std::min(argo_size, cache_size);
	/** Round the number of cache pages upwards */
	cachesize = align_forwards(cachesize, pagesize*CACHELINE);
	/** At least two pages are required to prevent endless eviction loops */
	cachesize = std::max(cachesize, static_cast<std::size_t>(pagesize*CACHELINE*2));
	cachesize /= pagesize;
	cache_locks.resize(cachesize);

	win_granularity = env::mpi_win_granularity();


	classificationSize = 2*(argo_size/pagesize);

	int *workranks = (int *) malloc(sizeof(int)*numtasks);
	int *procranks = (int *) malloc(sizeof(int)*2);
	int workindex = 0;

	for(argo::node_id_t i = 0; i < numtasks; i++){
		workranks[workindex++] = i;
		procranks[0]=i;
		procranks[1]=i+1;
	}

	MPI_Comm_group(MPI_COMM_WORLD, &startgroup);
	MPI_Group_incl(startgroup,numtasks,workranks,&workgroup);
	MPI_Comm_create(MPI_COMM_WORLD,workgroup,&workcomm);
	MPI_Group_rank(workgroup,&workrank);


	//Allocate local memory for each node,
	size_of_all = argo_size; //total distr. global memory
	GLOBAL_NULL=size_of_all+1;
	size_of_chunk = argo_size/(numtasks); //part on each node
	sig::signal_handler<SIGSEGV>::install_argo_handler(&handler);

	std::size_t cacheControlSize = sizeof(control_data)*cachesize;
	std::size_t gwritersize = classificationSize*sizeof(long);
	cacheControlSize = align_forwards(cacheControlSize, pagesize);
	gwritersize = align_forwards(gwritersize, pagesize);

	owners_dir_size = 3*(argo_size/pagesize);
	std::size_t owners_dir_size_bytes = owners_dir_size*sizeof(std::size_t);
	owners_dir_size_bytes = align_forwards(owners_dir_size_bytes, pagesize);

	std::size_t offsets_tbl_size = numtasks;
	std::size_t offsets_tbl_size_bytes = offsets_tbl_size*sizeof(std::size_t);
	offsets_tbl_size_bytes = align_forwards(offsets_tbl_size_bytes, pagesize);

	cacheoffset = pagesize*cachesize+cacheControlSize;

	globalData = static_cast<char*>(vm::allocate_mappable(pagesize, size_of_chunk));
	cacheData = static_cast<char*>(vm::allocate_mappable(pagesize, cachesize*pagesize));
	cacheControl = static_cast<control_data*>(vm::allocate_mappable(pagesize, cacheControlSize));

	touchedcache = (argo_byte *)malloc(cachesize);
	if(touchedcache == NULL){
		printf("malloc error out of memory\n");
		exit(EXIT_FAILURE);
	}

	pagecopy = static_cast<char*>(vm::allocate_mappable(pagesize, cachesize*pagesize));
	globalSharers = static_cast<std::uint64_t*>(vm::allocate_mappable(pagesize, gwritersize));

	if (dd::is_first_touch_policy()) {
		global_owners_dir = static_cast<std::uintptr_t*>(vm::allocate_mappable(pagesize, owners_dir_size_bytes));
		global_offsets_tbl = static_cast<std::uintptr_t*>(vm::allocate_mappable(pagesize, offsets_tbl_size_bytes));
	}

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	MPI_Barrier(MPI_COMM_WORLD);

	void* tmpcache;
	tmpcache=cacheData;
	vm::map_memory(tmpcache, pagesize*cachesize, 0, PROT_READ|PROT_WRITE);

	std::size_t current_offset = pagesize*cachesize;
	tmpcache=cacheControl;
	vm::map_memory(tmpcache, cacheControlSize, current_offset, PROT_READ|PROT_WRITE);

	current_offset += cacheControlSize;
	tmpcache=globalData;
	vm::map_memory(tmpcache, size_of_chunk, current_offset, PROT_READ|PROT_WRITE);

	current_offset += size_of_chunk;
	tmpcache=globalSharers;
	vm::map_memory(tmpcache, gwritersize, current_offset, PROT_READ|PROT_WRITE);

	if (dd::is_first_touch_policy()) {
		current_offset += gwritersize;
		tmpcache=global_owners_dir;
		vm::map_memory(tmpcache, owners_dir_size_bytes, current_offset, PROT_READ|PROT_WRITE);
		current_offset += owners_dir_size_bytes;
		tmpcache=global_offsets_tbl;
		vm::map_memory(tmpcache, offsets_tbl_size_bytes, current_offset, PROT_READ|PROT_WRITE);
	}

	argo_write_buffer.resize(env::write_buffer_count());

	num_data_windows = std::ceil(size_of_chunk/static_cast<double>(pagesize*CACHELINE*win_granularity));
	// Create one data_window per page chunk
	// TODO: Do we need the double dimensions or can each window be reused for another node?
	data_windows.resize(num_data_windows, std::vector<MPI_Win>(numtasks));
	for(std::size_t i = 0; i < num_data_windows; i++){
		std::size_t data_offset = i*pagesize*win_granularity;
		for(int j = 0; j < numtasks; j++){
			MPI_Win_create(&globalData[data_offset], pagesize*win_granularity*sizeof(argo_byte),
					1, MPI_INFO_NULL, MPI_COMM_WORLD, &data_windows[i][j]);
		}
	}
	// Locks to protect the globalData windows from concurrent local access
	mpi_lock_data = new mpi_lock*[num_data_windows];
	for(std::size_t i = 0; i < num_data_windows; i++){
		mpi_lock_data[i] = new mpi_lock[numtasks];
	}

	// Create one sharer_window per page chunk
	num_sharer_windows = std::ceil((classificationSize/2)/static_cast<double>(win_granularity));
	sharer_windows.resize(num_sharer_windows, std::vector<MPI_Win>(numtasks));
	for(std::size_t i = 0; i < num_sharer_windows; i++){
		std::size_t sharer_offset = i*2*win_granularity;
		for(int j = 0; j < numtasks; j++){
			MPI_Win_create(&globalSharers[sharer_offset],
					2*win_granularity*sizeof(unsigned long),
					sizeof(unsigned long), MPI_INFO_NULL,
					MPI_COMM_WORLD, &sharer_windows[i][j]);
		}
	}
	// Locks to protect the sharer windows from concurrent local access
	mpi_lock_sharer = new mpi_lock*[num_sharer_windows];
	for(std::size_t i = 0; i < num_sharer_windows; i++){
		mpi_lock_sharer[i] = new mpi_lock[numtasks];
	}


	if (dd::is_first_touch_policy()) {
		MPI_Win_create(global_owners_dir, owners_dir_size_bytes, sizeof(std::uintptr_t),
									 MPI_INFO_NULL, MPI_COMM_WORLD, &owners_dir_window);
		MPI_Win_create(global_offsets_tbl, offsets_tbl_size_bytes, sizeof(std::uintptr_t),
									 MPI_INFO_NULL, MPI_COMM_WORLD, &offsets_tbl_window);
	}

	memset(pagecopy, 0, cachesize*pagesize);
	memset(touchedcache, 0, cachesize);
	memset(globalData, 0, size_of_chunk*sizeof(argo_byte));
	memset(cacheData, 0, cachesize*pagesize);
	memset(globalSharers, 0, gwritersize);
	memset(cacheControl, 0, cachesize*sizeof(control_data));

	if (dd::is_first_touch_policy()) {
		memset(global_owners_dir, 0, owners_dir_size_bytes);
		memset(global_offsets_tbl, 0, offsets_tbl_size_bytes);
	}

	for(std::size_t i = 0; i < cachesize; i++){
		cacheControl[i].tag = GLOBAL_NULL;
		cacheControl[i].state = INVALID;
		cacheControl[i].dirty = CLEAN;
	}

	argo_reset_coherence(1);
	double init_end = MPI_Wtime();
	stats.inittime = init_end - init_start;
	stats.exectime = init_end;
}

void argo_finalize(){
	swdsm_argo_barrier(1);
	if(getID() == 0){
		printf("ArgoDSM shutting down\n");
	}
	swdsm_argo_barrier(1);
	stats.exectime = MPI_Wtime() - stats.exectime;
	mprotect(startAddr,size_of_all,PROT_WRITE|PROT_READ);
	MPI_Barrier(MPI_COMM_WORLD);

	print_statistics();

	MPI_Barrier(MPI_COMM_WORLD);

	// Free data windows
	for(auto &win_index : data_windows){
		for( auto window : win_index){
			MPI_Win_free(&window);
		}
	}
	// Free sharer windows
	for(auto &win_index : sharer_windows){
		for( auto window : win_index){
			MPI_Win_free(&window);
		}
	}
	if (dd::is_first_touch_policy()) {
		MPI_Win_free(&owners_dir_window);
		MPI_Win_free(&offsets_tbl_window);
	}
	MPI_Comm_free(&workcomm);
	MPI_Finalize();
	return;
}

void self_invalidation(){
	int flushed = 0;
	std::uint64_t id = static_cast<std::uint64_t>(1) << getID();

	double t1 = MPI_Wtime();
	for(std::size_t i = 0; i < cachesize; i+=CACHELINE){
		if(touchedcache[i] != 0){
			std::uintptr_t distrAddr = cacheControl[i].tag;
			std::uintptr_t lineAddr = distrAddr/(CACHELINE*pagesize);
			lineAddr*=(pagesize*CACHELINE);
			std::size_t classidx = get_classification_index(lineAddr);
			argo_byte dirty = cacheControl[i].dirty;

			if(flushed == 0 && dirty == DIRTY){
				for(auto &write_buffer : argo_write_buffer){
					write_buffer.flush();
				};
				flushed = 1;
			}
			std::size_t win_index = get_sharer_win_index(classidx);
			mpi_lock_sharer[win_index][workrank].lock(MPI_LOCK_SHARED, workrank, sharer_windows[win_index][workrank]);
			if(
				 // node is single writer
				 (globalSharers[classidx+1]==id)
				 ||
				 // No writer and assert that the node is a sharer
				 ((globalSharers[classidx+1]==0) && ((globalSharers[classidx]&id)==id))
				 ){
				mpi_lock_sharer[win_index][workrank].unlock(workrank, sharer_windows[win_index][workrank]);
				touchedcache[i] =1;
				/*nothing - we keep the pages, SD is done in flushWB*/
			}
			else{ //multiple writer or SO
				mpi_lock_sharer[win_index][workrank].unlock(workrank, sharer_windows[win_index][workrank]);
				cacheControl[i].dirty=CLEAN;
				cacheControl[i].state = INVALID;
				touchedcache[i] =0;
				mprotect((char*)startAddr + lineAddr, pagesize*CACHELINE, PROT_NONE);
			}
		}
	}
	double t2 = MPI_Wtime();
	stats.selfinvtime += (t2-t1);
}

void swdsm_argo_barrier(int n){ //BARRIER
	double time1,time2;
	pthread_t barrierlockholder;
	time1 = MPI_Wtime();
	pthread_barrier_wait(&threadbarrier[n]);
	if(argo_get_nodes()==1){
		time2 = MPI_Wtime();
		stats.barriers++;
		stats.barriertime += (time2-time1);
		return;
	}

	if(pthread_mutex_trylock(&barriermutex) == 0){
		barrierlockholder = pthread_self();
		pthread_rwlock_wrlock(&sync_lock);
		for(auto &write_buffer : argo_write_buffer){
			write_buffer.flush();
		};
		MPI_Barrier(workcomm);
		self_invalidation();
		pthread_rwlock_unlock(&sync_lock);
	}

	pthread_barrier_wait(&threadbarrier[n]);
	if(pthread_equal(barrierlockholder,pthread_self())){
		pthread_mutex_unlock(&barriermutex);
		time2 = MPI_Wtime();
		stats.barriers++;
		stats.barriertime += (time2-time1);
	}
}

void argo_reset_coherence(int n){
	stats.writebacks = 0;
	stats.stores = 0;
	memset(touchedcache, 0, cachesize);

	for(std::size_t i=0; i<classificationSize; i+=win_granularity){
		sharer_op(MPI_LOCK_EXCLUSIVE, workrank, i, [&](std::size_t){
				for(j = i; j < i+win_granularity; j++){
					globalSharers[j] = 0;
				}
			});
	}
	
	if (dd::is_first_touch_policy()) {
		/**
		 * @note initialize the first-touch directory with a magic value,
		 *       in order to identify if the indices are touched or not.
		 */
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, owners_dir_window);
		for(std::size_t i = 0; i < owners_dir_size; i++) {
			global_owners_dir[i] = GLOBAL_NULL;
		}
		MPI_Win_unlock(workrank, owners_dir_window);

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, offsets_tbl_window);
		for(argo::node_id_t n = 0; n < numtasks; n++) {
			global_offsets_tbl[n] = 0;
		}
		MPI_Win_unlock(workrank, offsets_tbl_window);
	}
	swdsm_argo_barrier(n);
	mprotect(startAddr,size_of_all,PROT_NONE);
	swdsm_argo_barrier(n);
	clearStatistics();
}

void argo_acquire(){
	int flag;
	pthread_rwlock_wrlock(&sync_lock);
	self_invalidation();
	MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,workcomm,&flag,MPI_STATUS_IGNORE);
	pthread_rwlock_unlock(&sync_lock);
}


void argo_release(){
	int flag;
	pthread_rwlock_wrlock(&sync_lock);
	for(auto &write_buffer : argo_write_buffer){
		write_buffer.flush();
	};
	MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,workcomm,&flag,MPI_STATUS_IGNORE);
	pthread_rwlock_unlock(&sync_lock);
}

void argo_acq_rel(){
	argo_acquire();
	argo_release();
}

void clearStatistics(){
	stats.selfinvtime = 0;
	stats.loadtime = 0;
	stats.storetime = 0;
	stats.locktime=0;
	stats.inittime=0;
	stats.exectime=0;
	stats.barriertime = 0;
	stats.stores = 0;
	stats.writebacks = 0;
	stats.loads = 0;
	stats.barriers = 0;
	stats.locks = 0;
	stats.ssitime = 0;
	stats.ssdtime = 0;
}

void storepageDIFF(std::size_t index, std::uintptr_t addr){
	int cnt = 0;
	
	// This might differ depending on allocation policy, must take into account
	const argo::node_id_t homenode = get_homenode(addr);
	const std::size_t offset = get_offset(addr);
	const std::size_t win_index = get_data_win_index(offset);
	const std::size_t win_offset = get_data_win_offset(offset);

	char * copy = (char *)(pagecopy + index*pagesize);
	char * real = (char *)startAddr+addr;
	size_t drf_unit = sizeof(char);

	if(!have_lock(win_index, homenode)){
		mpi_lock_data[win_index][homenode].lock(MPI_LOCK_EXCLUSIVE, homenode, data_windows[win_index][homenode]);
		add_to_locked(win_index, homenode);
	}

	std::size_t i;
	for(i = 0; i < pagesize; i+=drf_unit){
		int branchval;
		for(std::size_t j = i; j < i+drf_unit; j++){
			branchval = real[j] != copy[j];
			if(branchval != 0){
				break;
			}
		}
		if(branchval != 0){
			cnt+=drf_unit;
		}
		else{
			if(cnt > 0){
				MPI_Put(&real[i-cnt], cnt, MPI_BYTE, homenode, win_offset+(i-cnt), cnt, MPI_BYTE, data_windows[win_index][homenode]);
				cnt = 0;
			}
		}
	}
	if(cnt > 0){
		MPI_Put(&real[i-cnt], cnt, MPI_BYTE, homenode, win_offset+(i-cnt), cnt, MPI_BYTE, data_windows[win_index][homenode]);
	}
	stats.stores++;
}

/** Define some colors */
#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

void print_statistics(){
	/**
	 * Store statistics for the cache lock
	 */
	double cache_lock_time = 0;
	for( auto &cache_lock : cache_locks ) {
		cache_lock_time += cache_lock.get_lock_time();
	}

	/**
	 * Store statistics for the write buffers
	 */
	double flush_time(0), write_back_time(0), buffer_lock_time(0);
	std::size_t max_page_count(0), max_partial_flush_count(0);
	std::vector<double> page_count, partial_flush_count;

	// Iterate over all buffers
	for( auto &write_buffer : argo_write_buffer) {
		// Add total times
		flush_time += write_buffer.get_flush_time();
		write_back_time += write_buffer.get_write_back_time();
		buffer_lock_time += write_buffer.get_buffer_lock_time();

		// Store number of pages added per buffer and max number
		std::size_t pages = write_buffer.get_page_count();
		page_count.push_back(pages);
		if(pages > max_page_count) {
			max_page_count = pages;
		}

		// Store number of flushes per buffer and max number
		std::size_t flushes = write_buffer.get_partial_flush_count();
		partial_flush_count.push_back(flushes);
		if(flushes > max_partial_flush_count) {
			max_partial_flush_count = flushes;
		}
	}
	// Calculate the standard deviation of the load weights
	double stddev_page_count = 0;
	if(max_page_count > 0) {
		for(auto &e : page_count) {
			// Normalize to max load
			e /= max_page_count;
		}
		stddev_page_count = stddev(page_count);
	}

	// Calculate the standard deviation of the load weights
	double stddev_partial_flush_count = 0;
	if(max_partial_flush_count > 0) {
		for(auto &e : partial_flush_count) {
			// Normalize to max load
			e /= max_partial_flush_count;
		}
		stddev_partial_flush_count = stddev(partial_flush_count);
	}


	/**
	 *	Store MPI lock statistics for the data lock
	 */
	std::size_t data_num_locks(0);
	double data_spin_lock_time(0), data_spin_avg_lock_time(0), data_spin_max_lock_time(0);
	double data_spin_hold_time(0), data_spin_avg_hold_time(0), data_spin_max_hold_time(0);

	double data_mpi_lock_time(0), data_mpi_avg_lock_time(0), data_mpi_max_lock_time(0);
	double data_mpi_unlock_time(0), data_mpi_avg_unlock_time(0), data_mpi_max_unlock_time(0);
	double data_mpi_hold_time(0), data_mpi_avg_hold_time(0), data_mpi_max_hold_time(0);

	for(std::size_t i=0; i<num_data_windows; i++){
		for(int j=0; j<numtasks; j++){
			/* Get number of locks */
			data_num_locks += mpi_lock_data[i][j].get_num_locks();

			/* Get spin lock stats */
			data_spin_lock_time 		+= 	mpi_lock_data[i][j].get_spin_lock_time();
			data_spin_hold_time 		+= 	mpi_lock_data[i][j].get_spin_hold_time();
			if(mpi_lock_data[i][j].get_max_spin_lock_time() > data_spin_max_lock_time){
				data_spin_max_lock_time = mpi_lock_data[i][j].get_max_spin_lock_time();
			}
			if(mpi_lock_data[i][j].get_max_spin_hold_time() > data_spin_max_hold_time){
				data_spin_max_hold_time = mpi_lock_data[i][j].get_max_spin_hold_time();
			}

			/* Get mpi lock stats */
			data_mpi_lock_time 			+= 	mpi_lock_data[i][j].get_mpi_lock_time();
			data_mpi_unlock_time 		+= 	mpi_lock_data[i][j].get_mpi_unlock_time();
			data_mpi_hold_time 			+=	mpi_lock_data[i][j].get_mpi_hold_time();
			if(mpi_lock_data[i][j].get_max_mpi_lock_time() > data_mpi_max_lock_time){
				data_mpi_max_lock_time = mpi_lock_data[i][j].get_max_mpi_lock_time();
			}
			if(mpi_lock_data[i][j].get_max_mpi_unlock_time() > data_mpi_max_unlock_time){
				data_mpi_max_unlock_time = mpi_lock_data[i][j].get_max_mpi_unlock_time();
			}
			if(mpi_lock_data[i][j].get_max_mpi_hold_time() > data_mpi_max_hold_time){
				data_mpi_max_hold_time = mpi_lock_data[i][j].get_max_mpi_hold_time();
			}
		}
	}
	/** Get averages */
	data_spin_avg_lock_time 	= data_spin_lock_time / data_num_locks;
	data_spin_avg_hold_time 	= data_spin_hold_time / data_num_locks;
	data_mpi_avg_lock_time 		= data_mpi_lock_time / data_num_locks;
	data_mpi_avg_unlock_time 	= data_mpi_unlock_time / data_num_locks;
	data_mpi_avg_hold_time 		= data_mpi_hold_time / data_num_locks;

	/**
	 *	Store MPI lock statistics for the sharer lock
	 */
	std::size_t sharer_num_locks(0);
	double sharer_spin_lock_time(0), sharer_spin_avg_lock_time(0), sharer_spin_max_lock_time(0);
	double sharer_spin_hold_time(0), sharer_spin_avg_hold_time(0), sharer_spin_max_hold_time(0);

	double sharer_mpi_lock_time(0), sharer_mpi_avg_lock_time(0), sharer_mpi_max_lock_time(0);
	double sharer_mpi_unlock_time(0), sharer_mpi_avg_unlock_time(0), sharer_mpi_max_unlock_time(0);
	double sharer_mpi_hold_time(0), sharer_mpi_avg_hold_time(0), sharer_mpi_max_hold_time(0);

	for(std::size_t i=0; i<num_sharer_windows; i++){
		for(int j=0; j<numtasks; j++){
			/* Get number of locks */
			sharer_num_locks += mpi_lock_sharer[i][j].get_num_locks();

			/* Get spin lock stats */
			sharer_spin_lock_time 		+= 	mpi_lock_sharer[i][j].get_spin_lock_time();
			sharer_spin_hold_time 		+= 	mpi_lock_sharer[i][j].get_spin_hold_time();
			if(mpi_lock_sharer[i][j].get_max_spin_lock_time() > sharer_spin_max_lock_time){
				sharer_spin_max_lock_time = mpi_lock_sharer[i][j].get_max_spin_lock_time();
			}
			if(mpi_lock_sharer[i][j].get_max_spin_hold_time() > sharer_spin_max_hold_time){
				sharer_spin_max_hold_time = mpi_lock_sharer[i][j].get_max_spin_hold_time();
			}

			/* Get mpi lock stats */
			sharer_mpi_lock_time 			+= 	mpi_lock_sharer[i][j].get_mpi_lock_time();
			sharer_mpi_unlock_time 		+= 	mpi_lock_sharer[i][j].get_mpi_unlock_time();
			sharer_mpi_hold_time 			+=	mpi_lock_sharer[i][j].get_mpi_hold_time();
			if(mpi_lock_sharer[i][j].get_max_mpi_lock_time() > sharer_mpi_max_lock_time){
				sharer_mpi_max_lock_time = mpi_lock_sharer[i][j].get_max_mpi_lock_time();
			}
			if(mpi_lock_sharer[i][j].get_max_mpi_unlock_time() > sharer_mpi_max_unlock_time){
				sharer_mpi_max_unlock_time = mpi_lock_sharer[i][j].get_max_mpi_unlock_time();
			}
			if(mpi_lock_sharer[i][j].get_max_mpi_hold_time() > sharer_mpi_max_hold_time){
				sharer_mpi_max_hold_time = mpi_lock_sharer[i][j].get_max_mpi_hold_time();
			}
		}
	}
	/** Get averages */
	sharer_spin_avg_lock_time 	= sharer_spin_lock_time / sharer_num_locks;
	sharer_spin_avg_hold_time 	= sharer_spin_hold_time / sharer_num_locks;
	sharer_mpi_avg_lock_time 		= sharer_mpi_lock_time / sharer_num_locks;
	sharer_mpi_avg_unlock_time 	= sharer_mpi_unlock_time / sharer_num_locks;
	sharer_mpi_avg_hold_time 		= sharer_mpi_hold_time / sharer_num_locks;


	/** Nicely format and print the results */
	MPI_Barrier(MPI_COMM_WORLD);
	if(workrank==0){
		/** Adjust memory size */
		double mem_size_readable = size_of_all;
		std::vector<const char*> sizes = { "B ", "KB", "MB", "GB", "TB" };
		std::size_t order = 0;
		while (mem_size_readable >= 1024 && order < sizes.size()-1) {
			order++;
			mem_size_readable /= 1024;
		}

		printf("\n#################################" YEL" ArgoDSM statistics " RESET "##################################\n");
		printf("#  memory size: %12.2f%s  page size (p): %10dB   cache size: %13ldp\n",
				mem_size_readable, sizes[order], pagesize, cachesize);
		printf("#  write buffer size: %6ldp   write back size: %8ldp   write buffers: %10ld\n",
				env::write_buffer_size()/CACHELINE,
				env::write_buffer_write_back_size()/CACHELINE,
				env::write_buffer_count());
		printf("#  allocation policy: %6ld    policy block size: %6ldp   load size: %14ldp\n",
				env::allocation_policy(), env::allocation_block_size(), env::load_size());
		printf("#  active time: %12.4fs   init time: %14.4fs   CACHELINE: %14ldp\n",
				stats.exectime, stats.inittime, CACHELINE);
	}
	for(int i=0; i<numtasks; i++){
		MPI_Barrier(MPI_COMM_WORLD);
		if(i==workrank){
			printf("\n#" YEL "  ### PROCESS ID %d ###\n" RESET,workrank);

			/* Print remote access info */
			printf("#  " CYN "# Remote accesses\n" RESET);
			printf("#  read misses: %12lu    access time: %12.4fs\n",
					stats.loads, stats.loadtime);
			printf("#  write misses: %11lu    access time: %12.4fs\n",
					stats.stores, stats.storetime);

			/* Print coherence info */
			printf("#  " CYN "# Coherence actions\n" RESET);
			printf("#  locks held: %13d    barriers passed: %8lu    barrier time: %11.4fs\n",
					stats.locks, stats.barriers, stats.barriertime);
			printf("#  si time: %16.4fs   ssi time: %15.4fs   ssd time: %15.4fs\n",
					stats.selfinvtime, stats.ssitime, stats.ssdtime);

			/* Print write buffer info */
			printf("#  " CYN "# Write buffer\n" RESET);
			printf("#  flush time: %13.4fs   wrtbk time: %13.4fs   lock time: %14.4fs\n",
					flush_time, write_back_time, buffer_lock_time);
			printf("#  load stddev: %12.4f%%   flush stddev: %11.4f%%\n",
					stddev_page_count*100, stddev_partial_flush_count*100);

			/* Print cache lock info */
			printf("#  " CYN "# Cache lock\n" RESET);
			printf("#  cache lock time: %8.4fs\n",
					cache_lock_time);

			/* Print data lock info */
			printf("#  " CYN "# Data lock  \t(%zu locks held)\n" RESET, data_num_locks);
			printf("#  spin lock time: %9.4fs   avg lock time: %10.4fs   max lock time: %10.4fs\n",
					data_spin_lock_time, data_spin_avg_lock_time, data_spin_max_lock_time);
			printf("#  spin hold time: %9.4fs   avg hold time: %10.4fs   max hold time: %10.4fs\n",
					data_spin_hold_time, data_spin_avg_hold_time, data_spin_max_hold_time);
			printf("#  mpi lock time: %10.4fs   avg lock time: %10.4fs   max lock time: %10.4fs\n",
					data_mpi_lock_time, data_mpi_avg_lock_time, data_mpi_max_lock_time);
			printf("#  mpi unlock time: %8.4fs   avg unlock time: %8.4fs   max unlock time: %8.4fs\n",
					data_mpi_unlock_time, data_mpi_avg_unlock_time, data_mpi_max_unlock_time);
			printf("#  mpi hold time: %10.4fs   avg hold time: %10.4fs   max hold time: %10.4fs\n",
					data_mpi_hold_time, data_mpi_avg_hold_time, data_mpi_max_hold_time);


			/* Print sharer lock info */
			printf("#  " CYN "# Sharer lock  \t(%zu locks held)\n" RESET, sharer_num_locks);
			printf("#  spin lock time: %9.4fs   avg lock time: %10.4fs   max lock time: %10.4fs\n",
					sharer_spin_lock_time, sharer_spin_avg_lock_time, sharer_spin_max_lock_time);
			printf("#  spin hold time: %9.4fs   avg hold time: %10.4fs   max hold time: %10.4fs\n",
					sharer_spin_hold_time, sharer_spin_avg_hold_time, sharer_spin_max_hold_time);
			printf("#  mpi lock time: %10.4fs   avg lock time: %10.4fs   max lock time: %10.4fs\n",
					sharer_mpi_lock_time, sharer_mpi_avg_lock_time, sharer_mpi_max_lock_time);
			printf("#  mpi unlock time: %8.4fs   avg unlock time: %8.4fs   max unlock time: %8.4fs\n",
					sharer_mpi_unlock_time, sharer_mpi_avg_unlock_time, sharer_mpi_max_unlock_time);
			printf("#  mpi hold time: %10.4fs   avg hold time: %10.4fs   max hold time: %10.4fs\n",
					sharer_mpi_hold_time, sharer_mpi_avg_hold_time, sharer_mpi_max_hold_time);
			printf("\n");
			fflush(stdout);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void *argo_get_global_base(){return startAddr;}
size_t argo_get_global_size(){return size_of_all;}

std::size_t get_classification_index(std::uintptr_t addr){
	return (2*(addr/(pagesize*CACHELINE))) % classificationSize;
}

bool _is_cached(std::uintptr_t addr) {
	argo::node_id_t homenode;
	std::size_t aligned_address = align_backwards(
			addr-reinterpret_cast<std::size_t>(startAddr), CACHELINE*pagesize);
	homenode = peek_homenode(aligned_address);
	std::size_t cache_index = getCacheIndex(aligned_address);

	// Return true for pages which are either local or already cached
	return ((homenode == getID()) || (cacheControl[cache_index].tag == aligned_address &&
				cacheControl[cache_index].state == VALID));
}

void sharer_op(int lock_type, int rank, int offset,
		std::function<void(const std::size_t window_index)> op) {
	std::size_t win_index = get_sharer_win_index(offset);
	mpi_lock_sharer[win_index][rank].lock(lock_type, rank, sharer_windows[win_index][rank]);
	op(win_index);
	mpi_lock_sharer[win_index][rank].unlock(rank, sharer_windows[win_index][rank]);
}

std::size_t get_sharer_win_index(int classification_index){
	return (classification_index/2)/win_granularity;
}

std::size_t get_sharer_win_offset(int classification_index){
	return classification_index%(win_granularity*2);
}

std::size_t get_data_win_index(std::size_t offset){
	return (offset/(pagesize*CACHELINE))/win_granularity;
}

std::size_t get_data_win_offset(std::size_t offset){
	return offset%(win_granularity*(pagesize*CACHELINE));
}

void add_to_locked(int data_win_index, int homenode){
	locked_windows.emplace_back(std::make_pair(data_win_index, homenode));
}

void unlock_windows() {
	// Unlock all windows
	for(const auto &p : locked_windows){
		mpi_lock_data[p.first][p.second].unlock(
				p.second, data_windows[p.first][p.second]);
	}
	locked_windows.clear();
}

bool have_lock(int data_win_index, int homenode){
	auto it = std::find_if(locked_windows.begin(), locked_windows.end(),
			[&data_win_index, &homenode](const std::pair<int, int>& p) {
			return (p.first == data_win_index) && (p.second == homenode);
			});
	return it != locked_windows.end();
}

std::size_t get_write_buffer(std::size_t cache_index){
	// Skew the calculation to avoid poor load balancing
	return (cache_index + (cache_index / env::write_buffer_count()) + 1) % env::write_buffer_count();
}

double stddev(const std::vector<double> &v){
	double sum = std::accumulate(v.begin(), v.end(), 0.0);
	double mean = sum / v.size();

	std::vector<double> diff(v.size());
	std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	return std::sqrt(sq_sum / v.size());
}
