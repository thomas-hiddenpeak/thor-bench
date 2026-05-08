#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>

typedef void* (*dlopen_fn)(const char*, int);
static dlopen_fn real_dlopen = NULL;

void* dlopen(const char* filename, int flags) {
    if (real_dlopen == NULL) real_dlopen = (dlopen_fn)dlsym(RTLD_NEXT, "dlopen");
    if (filename && strstr(filename, "nvcuextend") != NULL) return NULL;
    return real_dlopen(filename, flags);
}
