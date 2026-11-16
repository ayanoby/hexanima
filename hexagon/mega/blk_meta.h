#pragma once
/* offsets (bytes) into the shared blob, passed as int32 meta[] */
enum { M_X = 0, M_CTX, M_MOD, M_GAIN, M_COS, M_SIN,
       M_WSQ, M_WSK, M_WSV, M_WSO, M_WCQ, M_WCK, M_WCV, M_WCO, M_WM1, M_WM2,
       M_H, M_Q, M_K, M_V, M_O, M_T, M_MID, M_CK, M_CV, M_NTOK, M_NCTX, M_STOP, M_NW, M_REL, M_FIRST, M_Q8, M_FORCE, M_W0, M_WN, M_PASS, M_NMETA };   /* windows (M_WN > 0): the call works on slabs [M_W0, M_W0+M_WN) of the tokens; H/Q/O/T/MID buffers are sized for one window, X/K/V/COS/SIN for all of them; M_PASS 1 = only K and V of the window into the global K/V */   /* M_FORCE: test only, (QB << 16) | C forces the chunked attention with these sizes (0 = automatic) */
