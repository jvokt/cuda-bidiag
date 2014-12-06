INCLUDES  := -I. -I${CUDA_INC_PATH}
LIBRARIES := -L${CUDA_LIB_PATH_64}

LIBRARIES += -lcublas -lcusparse

all: build

build: cudaBidiag

mmio.o: mmio.c
	g++ -O3 $(INCLUDES) -o mmio.o -c mmio.c 

main.o: main.cpp
	nvcc -m64 $(INCLUDES) -o main.o -c main.cpp

cudaBidiag: mmio.o main.o
	nvcc -m64 -o cudaBidiag mmio.o main.o $(LIBRARIES)

run: build
	./cudaBidiag

clean:
	rm -f cudaBidiag main.o mmio.o

clobber: clean
