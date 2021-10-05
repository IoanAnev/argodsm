# Copyright (C) Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.

add_library(argobackend-mpi SHARED mpi.cpp swdsm.cpp coherence.cpp mpi_lock.cpp)

install(TARGETS argobackend-mpi
	COMPONENT "Runtime"
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
