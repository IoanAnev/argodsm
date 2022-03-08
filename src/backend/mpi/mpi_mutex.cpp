#include "mpi_mutex.hpp"

// Exclusive ownership

void mpi_mutex::lock() {
	// Mutex ensures atomicity of lock operations
	std::unique_lock<std::mutex> lk(mut_);

	while (state_ & write_entered_)
		gate1_.wait(lk);
	state_ |= write_entered_;
	while (state_ & n_readers_)
		gate2_.wait(lk);

	// Exclusive lock acquired
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, node_, 0, *window_);
}

bool mpi_mutex::try_lock() {
	std::unique_lock<std::mutex> lk(mut_, std::try_to_lock);
	if (lk.owns_lock() && state_ == 0) {
		state_ = write_entered_;
		// Exclusive lock acquired
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, node_, 0, *window_);
		return true;
	}
	return false;
}

void mpi_mutex::unlock() {
	{
		std::lock_guard<std::mutex> _(mut_);
		state_ = 0;
		MPI_Win_unlock(node_, *window_);
	}
	gate1_.notify_all();
}

// Shared ownership

void mpi_mutex::lock_shared() {
	std::unique_lock<std::mutex> lk(mut_);
	while ((state_ & write_entered_) || (state_ & n_readers_) == n_readers_)
		gate1_.wait(lk);
	unsigned num_readers = (state_ & n_readers_) + 1;
	state_ &= ~n_readers_;
	state_ |= num_readers;

	// Shared lock acquired
	if(num_readers == 1) {
		// "Global" lock taken
		MPI_Win_lock(MPI_LOCK_SHARED, node_, 0, *window_);
	} else {
		// "Local" lock hooked on to
	}
}

bool mpi_mutex::try_lock_shared() {
	std::unique_lock<std::mutex> lk(mut_, std::try_to_lock);
	unsigned num_readers = state_ & n_readers_;
	if (lk.owns_lock() && !(state_ & write_entered_) && num_readers != n_readers_) {
		++num_readers;
		state_ &= ~n_readers_;
		state_ |= num_readers;

		// Shared lock acquired
		if(num_readers == 1) {
			// "Global" lock taken
			MPI_Win_lock(MPI_LOCK_SHARED, node_, 0, *window_);
		} else {
			// "Local" lock hooked on to
		}

		return true;
	}
	return false;
}

void mpi_mutex::unlock_shared() {
	std::lock_guard<std::mutex> _(mut_);
	unsigned num_readers = (state_ & n_readers_) - 1;
	state_ &= ~n_readers_;
	state_ |= num_readers;

	// Writer waiting
	if (state_ & write_entered_) {
		// We are last reader to leave
		if (num_readers == 0) {
			// Unlock MPI Window
			MPI_Win_unlock(node_, *window_);
			// Notify writer waiting at gate 2
			gate2_.notify_one();
		}
		// We are not last reader, flush MPI operations
		else {
			// TODO: Can we just flush locally?
			MPI_Win_flush(node_, *window_);
		}
	}
	// Only readers waiting
	else
	{
		// Perform unlock or flush
		if(num_readers == 0) {
			MPI_Win_unlock(node_, *window_);
		} else {
			MPI_Win_flush(node_, *window_);
		}

		// Notify first reader waiting at gate 1
		if (num_readers == n_readers_ - 1)
			gate1_.notify_one();
	}
}
