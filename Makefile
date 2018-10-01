#
# Makefile
#


SRCNT=matmul_double.cu
SRCT=matmul_double_t.cu

EXECNT=$(SRCNT:.cu=)
EXECT=$(SRCT:.cu=)

OBJSNT=$(SRCNT:.cu=.o)
OBJST=$(SRCT:.cu=.o)

NVCCFLAGS=--gpu-architecture=compute_60 --gpu-code=sm_60 -O3

CC=nvcc

all: $(EXECNT) 
	@echo "Change line 18 in Makefile to \"all: \$$(EXECNT) \$$(EXECT)\" to build both T and NT versions"

$(EXECNT): $(OBJSNT)
	$(CC) $(NVCCFLAGS) $^  -o $@

$(EXECT): $(OBJST)
	$(CC) $(NVCCFLAGS) $^ -o $@

%.o : %.cu
	$(CC)  $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(EXECNT) $(EXECT) $(OBJSNT) $(OBJST)


# vim:ft=make
#
