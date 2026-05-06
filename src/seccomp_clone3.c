#define _GNU_SOURCE
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef __NR_clone3
#define __NR_clone3 435
#endif
#ifndef __NR_seccomp
#define __NR_seccomp 277
#endif

int main(int argc, char *argv[]) {
    struct sock_filter insns[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, 0),
        { BPF_JMP | BPF_JEQ | BPF_K, 0, 1, __NR_clone3 },
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (ENOSYS & SECCOMP_RET_DATA)),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog prog = { .len = 4, .filter = insns };

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl");
        return 1;
    }

    long ret = syscall(__NR_seccomp, SECCOMP_SET_MODE_FILTER, 2, &prog);
    if (ret != 0) {
        ret = syscall(__NR_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog);
        if (ret != 0) {
            perror("seccomp");
            return 1;
        }
        fprintf(stderr, "Seccomp installed WITHOUT ERRNO flag\n");
    } else {
        fprintf(stderr, "Seccomp installed WITH ERRNO flag\n");
    }

    if (argc < 2) {
        long r = syscall(__NR_clone3, 0, 0);
        fprintf(stderr, "clone3 returned: %ld errno: %d\n", r, errno);
        return 0;
    }

    execvp(argv[1], &argv[1]);
    perror("execvp");
    return 127;
}
