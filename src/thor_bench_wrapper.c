/*
 * thor_bench_wrapper.c — Thin loader that sets LD_PRELOAD for nocudaextend.so
 * and execs thor_bench_impl. This is how `./build/thor_bench` works bare.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>
#include <sys/stat.h>

int main(int argc, char* argv[]) {
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        perror("readlink /proc/self/exe");
        return 1;
    }
    exe_path[len] = '\0';

    char* dir = dirname(exe_path);

    char nocudaextend_path[PATH_MAX];
    snprintf(nocudaextend_path, sizeof(nocudaextend_path), "%s/nocudaextend.so", dir);

    struct stat st;
    if (stat(nocudaextend_path, &st) != 0) {
        char impl_path[PATH_MAX];
        snprintf(impl_path, sizeof(impl_path), "%s/thor_bench_impl", dir);
        execv(impl_path, argv);
        perror("execv thor_bench_impl");
        return 1;
    }

    const char* existing = getenv("LD_PRELOAD");
    char new_preload[PATH_MAX + 512];
    if (existing && *existing) {
        snprintf(new_preload, sizeof(new_preload), "%s:%s", nocudaextend_path, existing);
    } else {
        snprintf(new_preload, sizeof(new_preload), "%s", nocudaextend_path);
    }
    setenv("LD_PRELOAD", new_preload, 1);

    /* Exec the real binary */
    char impl_path[PATH_MAX];
    snprintf(impl_path, sizeof(impl_path), "%s/thor_bench_impl", dir);
    execv(impl_path, argv);
    perror("execv thor_bench_impl");
    return 1;
}
