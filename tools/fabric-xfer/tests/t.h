#ifndef T_H
#define T_H
#include <stdio.h>
static int t_fail, t_count;
#define CHECK(c) do { t_count++; if (!(c)) { t_fail++; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
#define T_DONE() do { printf("%s: %d checks, %d failed\n", __FILE__, t_count, t_fail); return t_fail ? 1 : 0; } while (0)
#endif
