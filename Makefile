CC=nvcc

GLUT_LIBS=-lGL -lGLU -lglut
CU_LIBS=-I"/usr/local/cuda/samples/common/inc"

LDLIBS=${GLUT_LIBS} ${CU_LIBS}

FILES=main.o floatmath.o boid.o


all: fishes

fishes: ${FILES}
	${CC} ${FILES} -o fishes ${LDLIBS}

main.o:	main.cu
	${CC} -c main.cu -o main.o ${LDLIBS}

boid.o: boid.cu boid.cuh
	${CC} -c boid.cu -o boid.o ${CU_LIBS} -dc

floatmath.o: floatmath.cu floatmath.cuh
	${CC} -c floatmath.cu -o floatmath.o ${CU_LIBS} -dc

clean:
	rm -f fishes *.o
