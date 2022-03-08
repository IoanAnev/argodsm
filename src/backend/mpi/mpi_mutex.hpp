#include <mutex>
#include <condition_variable>
#include <climits>
#include <mpi.h>

#ifndef argo_mpi_mutex_hpp
#define argo_mpi_mutex_hpp argo_mpi_mutex_hpp

class mpi_mutex
{
	int			node_;
	MPI_Win*	window_;
	std::mutex	mut_;
	std::condition_variable gate1_;
	std::condition_variable gate2_;
	unsigned	state_;

    static const unsigned write_entered_ = 1U <<
		(sizeof(unsigned)*CHAR_BIT - 1);
    static const unsigned n_readers_ = ~write_entered_;

public:

    mpi_mutex(int node, MPI_Win* window) : node_(node), window_(window), state_(0) {}

	// Exclusive ownership

	void lock();
	bool try_lock();
	void unlock();

	// Shared ownership

	void lock_shared();
	bool try_lock_shared();
	void unlock_shared();
};

#endif /* argo_mpi_mutex_hpp */
