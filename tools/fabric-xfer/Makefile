# fabric-xfer: gcc 13, libibverbs-dev (stock Spark). `make` builds the tool; `make test` needs no RDMA headers.
CC      ?= gcc
CFLAGS  ?= -O2
CFLAGS  += -std=gnu11 -Wall -Wextra
CORE     = fx_core.c fx_engine.c
TESTS    = tests/test_core tests/test_engine

all: fabric-xfer

fabric-xfer: fabric_xfer.c $(CORE) fx_verbs.c fx.h
	$(CC) $(CFLAGS) -o $@ fabric_xfer.c $(CORE) fx_verbs.c -libverbs -lpthread

tests/test_core: tests/test_core.c $(CORE) fx_loop.c fx.h tests/t.h
	$(CC) $(CFLAGS) -o $@ tests/test_core.c $(CORE) fx_loop.c -lpthread

tests/test_engine: tests/test_engine.c $(CORE) fx_loop.c fx.h tests/t.h
	$(CC) $(CFLAGS) -o $@ tests/test_engine.c $(CORE) fx_loop.c -lpthread

test: $(TESTS)
	./tests/test_core && ./tests/test_engine && bash tests/test_wrapper.sh

clean:
	rm -f fabric-xfer $(TESTS)

.PHONY: all test clean
