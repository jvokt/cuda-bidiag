### CUDA Implementation of Golub-Kahan-Lanczos Bidiagonalization ###

* Implemented the algorithm in CUDA, ran it on random examples, and ran it on the MovieLens database. 
* Used the files from NIST called mmio.c and mmio.h to read a file in Matrix Market format. 
* Used cusparseXcoo2csr to convert the matrix from Coordinate format (arrays of equal length for rows, cols, and values), to Compressed Sparse Row format
* Made significant use of cuBLAS and cuSPARSE during the main algorithm