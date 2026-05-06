#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

/* Workaround for clone3 kernel bug on Tegra 6.8.12-tegra
 * libnvcuextend.so creates threads via clone3 which SEGV_MAPERR at NULL
 * Blocking its load allows CUDA runtime to work normally */
void *dlopen(const char *filename, int flag) {
    if (filename && strstr(filename, "nvcuextend")) {
        return NULL;
    }
    return ((void *(*)(const char *, int))dlsym(RTLD_NEXT, "dlopen"))(filename, flag);
}
