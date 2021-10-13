/**
 * @file
 * @brief This file provides an interface for the ArgoDSM write buffer.
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#ifndef argo_write_buffer_hpp
#define argo_write_buffer_hpp argo_write_buffer_hpp

#include <deque>
#include <iterator>
#include <algorithm>
#include <mutex>
#include <mpi.h>
#include <vector>
#include <thread>

#include "backend/backend.hpp"
#include "env/env.hpp"
#include "virtual_memory/virtual_memory.hpp"
#include "swdsm.h"

/**
 * @brief		Argo cache data structure
 * @deprecated 	prototype implementation, should be replaced with API calls
 */
extern control_data* cacheControl;

/** @brief Block size based on backend definition */
const std::size_t block_size = page_size*CACHELINE;

//TODO: Document
extern std::vector<cache_lock> cache_locks;

/**
 * @brief	A write buffer in FIFO style with the capability to erase any
 * element from the buffer while preserving ordering.
 * @tparam	T the type of the write buffer
 */
template<typename T>
class write_buffer
{
	private:
		/** @brief This container holds cache indexes that should be written back */
		std::deque<T> _buffer;

		/** @brief The maximum size of the write buffer */
		std::size_t _max_size;

		/**
		 * @brief The number of elements to write back once attempting to
		 * add an element to an already full write buffer.
		 */
		std::size_t _write_back_size;

		/** @brief	Mutex to protect the write buffer from simultaneous accesses */
		mutable std::mutex _buffer_mutex;
		mutable std::mutex _parallel_flush_mutex;

		/** @brief Time keeping for the write buffer */
		double _flush_time;
		double _write_back_time;
		double _buffer_lock_time;

		/** @brief Stat keeping for the write buffer */
		std::size_t _page_count;
		std::size_t _partial_flush_count;

		/**
		 * @brief	Check if the write buffer is empty
		 * @return	True if empty, else False
		 */
		bool empty() {
			return _buffer.empty();
		}

		/**
		 * @brief	Get the size of the buffer
		 * @return	The size of the buffer
		 */
		size_t size() {
			return _buffer.size();
		}

		/**
		 * @brief	Get the buffer element at index i
		 * @param	i The requested buffer index
		 * @return	The element at index i of type T
		 */
		T at(std::size_t i){
			return _buffer.at(i);
		}

		/**
		 * @brief	Check if val exists in the buffer
		 * @param	val The value to check for
		 * @return	True if val exists in the buffer, else False
		 */
		bool has(T val) {
			typename std::deque<T>::iterator it = std::find(_buffer.begin(),
					_buffer.end(), val);
			return (it != _buffer.end());
		}

		/**
		 * @brief	Constructs a new element and emplaces it at the back of the buffer
		 * @param	args Properties of the new element to emplace back
		 */
		template<typename... Args>
			void emplace_back( Args&&... args) {
				_buffer.emplace_back(std::forward<Args>(args)...);
			}

		/**
		 * @brief	Removes the front element from the buffer and returns it
		 * @return	The element that was popped from the buffer
		 */
		T pop() {
			auto elem = std::move(_buffer.front());
			_buffer.pop_front();
			return elem;
		}

		/**
		 * @brief	Sorts all elements by home node id in ascending order
		 */
		void sort() {
			std::sort(_buffer.begin(), _buffer.end(),
					[](const T& l, const T& r) {
				return get_homenode(cacheControl[l].tag) < get_homenode(cacheControl[r].tag);
			});
		}

		/**
		 * @brief	Sorts the first _write_back_size elements by home node id in ascending order
		 */
		void sort_first() {
			assert(_buffer.size() >= _write_back_size);
			std::sort(_buffer.begin(), _buffer.begin()+_write_back_size,
					[](const T& l, const T& r) {
				return get_homenode(cacheControl[l].tag) < get_homenode(cacheControl[r].tag);
			});
		}

		/**
		 * @brief	Flushes first _write_back_size elements of the  ArgoDSM 
		 * 			write buffer to memory
		 * @pre		Require write_buffer_mutex to be held
		 */
		void flush_partial() {
			double t_start = MPI_Wtime();
			// Sort the first _write_back_size elements
			sort_first();

			// For each element, handle the corresponding ArgoDSM page
			for(std::size_t i = 0; i < _write_back_size; i++) {
				// The code below should be replaced with a cache API
				// call to write back a cached page
				std::size_t cache_index = pop();
				cache_locks[cache_index].lock();
				std::uintptr_t page_address = cacheControl[cache_index].tag;
				void* page_ptr = static_cast<char*>(
						argo::virtual_memory::start_address()) + page_address;

				// Write back the page
				mprotect(page_ptr, block_size, PROT_READ);
				cacheControl[cache_index].dirty=CLEAN;
				for(int i=0; i < CACHELINE; i++){
					storepageDIFF(cache_index+i,page_size*i+page_address);
				}
				cache_locks[cache_index].unlock();
				// Close any windows used to write back data
				// This should be replaced with an API call
				// TODO: Any impact of moving this outside?
				unlock_windows();
			}
			double t_stop = MPI_Wtime();
			_write_back_time += t_stop-t_start;
		}

		/**
		 * @brief	Iterates over the buffer and writes back elements in chunks
		 * 			until there are no elements left.
		 */
		void process_buffer(){
			std::size_t block_size = argo::env::mpi_win_granularity();

			// Continue until the buffer is empty
			while(!empty()) {
				std::vector<std::size_t> cache_indices;
				{
					// Under protected access, get block_size number of indices
					std::lock_guard<std::mutex> pop_lock(_parallel_flush_mutex);
					for(std::size_t i=0; i<block_size; i++){
						if(!empty()){
							cache_indices.push_back(pop());
						}else{
							break;
						}
					}
				}

				// For each index, handle the corresponding ArgoDSM page
				for(auto cache_index : cache_indices) {
					std::uintptr_t page_address = cacheControl[cache_index].tag;
					void* page_ptr = static_cast<char*>(
							argo::virtual_memory::start_address()) + page_address;

					// Write back the page and clean up cache
					mprotect(page_ptr, block_size, PROT_READ);
					cacheControl[cache_index].dirty=CLEAN;
					for(int i=0; i < CACHELINE; i++){
						storepageDIFF(cache_index+i,page_size*i+page_address);
					}
					// The windows must be unlocked for concurrency
					unlock_windows();
				}
			}
		}

	public:
		/**
		 * @brief	Constructor
		 */
		write_buffer()	:
			_max_size(argo::env::write_buffer_size()/CACHELINE),
			_write_back_size(argo::env::write_buffer_write_back_size()/CACHELINE),
			_flush_time(0),
			_write_back_time(0),
			_buffer_lock_time(0),
			_page_count(0),
			_partial_flush_count(0) { }

		/**
		 * @brief	Copy constructor
		 * @param	other The write_buffer object to copy from
		 * @note	with c++14 we could use std::shared_lock for lock_other
		 */
		write_buffer(const write_buffer & other) {
			// Ensure protection of data
			std::lock_guard<std::mutex> lock_other(other._buffer_mutex);
			// Copy data
			_buffer = other._buffer;
			_max_size = other._max_size;
			_write_back_size = other._write_back_size;
			_flush_time = other._flush_time;
			_write_back_time = other._write_back_time;
			_buffer_lock_time = other._buffer_lock_time;
			_page_count = other._page_count;
			_partial_flush_count = other._partial_flush_count;
		}

		/**
		 * @brief	Copy assignment operator
		 * @param	other the write_buffer object to copy assign from
		 * @return	reference to the created copy
		 * @note	with c++14 we could use std::shared_lock for lock_other
		 */
		write_buffer& operator=(const write_buffer & other) {
			if(&other != this) {
				// Ensure protection of data
				std::unique_lock<std::mutex> lock_this(_buffer_mutex, std::defer_lock);
				std::unique_lock<std::mutex> lock_other(other._buffer_mutex, std::defer_lock);
				std::lock(lock_this, lock_other);
				// Copy data
				_buffer = other._buffer;
				_max_size = other._max_size;
				_write_back_size = other._write_back_size;
				_flush_time = other._flush_time;
				_write_back_time = other._write_back_time;
				_buffer_lock_time = other._buffer_lock_time;
				_page_count = other._page_count;
				_partial_flush_count = other._partial_flush_count;
			}
			return *this;
		}

		/**
		 * @brief	Move constructor
		 * @param	other the write_buffer object to move from
		 */
		write_buffer(write_buffer && other) {
			// Ensure protection of data
			std::lock_guard<std::mutex> lock_other(other._buffer_mutex);

			// Copy data
			_buffer = std::move(other._buffer);
			_max_size = std::move(other._max_size);
			_write_back_size = std::move(other._write_back_size);
			_flush_time = std::move(other._flush_time);
			_write_back_time = std::move(other._write_back_time);
			_buffer_lock_time = std::move(other._buffer_lock_time);
			_page_count = std::move(other._page_count);
			_partial_flush_count = std::move(other._partial_flush_count);
		}

		/**
		 * @brief	Move assignment operator
		 * @param	other the write_buffer object to move assign from
		 * @return	reference to the moved object
		 */
		write_buffer& operator=(write_buffer && other) {
			if (&other != this) {
				// Ensure protection of data
				std::unique_lock<std::mutex> lock_this(_buffer_mutex, std::defer_lock);
				std::unique_lock<std::mutex> lock_other(other._buffer_mutex, std::defer_lock);
				std::lock(lock_this, lock_other);

				_buffer = std::move(other._buffer);
				_max_size = std::move(other._max_size);
				_write_back_size = std::move(other._write_back_size);
				_flush_time = std::move(other._flush_time);
				_write_back_time = std::move(other._write_back_time);
				_buffer_lock_time = std::move(other._buffer_lock_time);
				_page_count = std::move(other._page_count);
				_partial_flush_count = std::move(other._partial_flush_count);
			}
			return *this;
		}


		/**
		 * @brief	If val exists in buffer, delete it. Else, do nothing.
		 * @param	val The value of type T to erase
		 */
		void erase(T val) {
			double t_start = MPI_Wtime();
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			_buffer_lock_time += MPI_Wtime() - t_start;

			typename std::deque<T>::iterator it = std::find(_buffer.begin(),
					_buffer.end(), val);
			if(it != _buffer.end()){
				_buffer.erase(it);
			}
		}

		/**
		 * @brief	Flushes the ArgoDSM write buffer to memory
		 */
		void flush() {
			double t_start = MPI_Wtime();
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			_buffer_lock_time += MPI_Wtime() - t_start;

			// If it's empty we don't need to do anything
			if(empty()){
				double t_stop = MPI_Wtime();
				_flush_time += t_stop-t_start;
				return;
			}
			// Otherwise, sort the buffer
			sort();

			// Start an appropriate amount of workers
			std::size_t hwthreads = std::thread::hardware_concurrency();
			std::size_t nthreads = hwthreads > 1 ? hwthreads/2 : 1;
			//TODO: This needs to be configurable for SWnodes
			std::vector<std::thread> threads;
			for(std::size_t n=0; n<nthreads; n++){
				threads.emplace_back(&write_buffer::process_buffer, this);
			}
			// Wait for them all to finish
			for(auto& t : threads) {
				if (t.joinable()) {
					t.join();
				}
			}

			// Update timer statistics
			double t_stop = MPI_Wtime();
			_flush_time += t_stop-t_start;
		}

		/**
		 * @brief	Adds a new element to the write buffer
		 * @param	val The value of type T to add to the buffer
		 */
		void add(T val) {
			double t_start = MPI_Wtime();
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			_buffer_lock_time += MPI_Wtime() - t_start;

			// If already present in the buffer, do nothing
			if(has(val)){
				return;
			}

			// If the buffer is full, write back _write_back_size indices
			if(size() >= _max_size){
				flush_partial();
				_partial_flush_count++;
			}

			// Add val to the back of the buffer
			emplace_back(val);
			_page_count++;
		}

		/**
		 * @brief	Get the time spent flushing the write buffer
		 * @return	The time in seconds
		 */
		double get_flush_time() {
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			return _flush_time;
		}

		/**
		 * @brief	Get the time spent partially flushing the write buffer
		 * @return	The time in seconds
		 */
		double get_write_back_time() {
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			return _write_back_time;
		}

		/**
		 * @brief	Get the time spent waiting for the write buffer lock
		 * @return	The time in seconds
		 */
		double get_buffer_lock_time() {
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			return _buffer_lock_time;
		}

		/**
		 * @brief get buffer size
		 */
		std::size_t get_size() {
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			return _buffer.size();
		}

		/**
		 * @brief get total number of pages added
		 */
		std::size_t get_page_count() {
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			return _page_count;
		}

		/**
		 * @brief get the number of times partially flushed
		 */
		std::size_t get_partial_flush_count() {
			std::lock_guard<std::mutex> lock(_buffer_mutex);
			return _partial_flush_count;
		}

}; //class

#endif /* argo_write_buffer_hpp */
