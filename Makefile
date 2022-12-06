all: hamming_one

hamming_one: kernel.cu
	nvcc kernel.cu -o 'Hamming One'

clean:
	rm 'Hamming One'
