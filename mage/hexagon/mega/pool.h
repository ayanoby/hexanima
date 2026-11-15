#pragma once
typedef void (*pool_fn)(void *arg, int idx, int n);
int pool_init(int nworkers);       /* spawn workers (spin-wait) */
void pool_start(pool_fn fn, void *arg);
void pool_wait(void);
void pool_run(pool_fn fn, void *arg);  /* start + wait */
void pool_quit(void);
int pool_n(void);
