.p2align 4,,15
.global hmx_acc
.type hmx_acc, @function
hmx_acc:
{
    activation.hf = mxmem(R0, R1):deep;
    weight.hf = mxmem(R2, R3);
}
{
    jumpr R31;
}
.size hmx_acc, .-hmx_acc
#if __HEXAGON_ARCH__ < 73
.p2align 4,,15
.global hmx_store
.type hmx_store, @function
hmx_store:
{
    mxmem(R0, R1):after.hf = acc;
}
{
    jumpr R31;
}
.size hmx_store, .-hmx_store
#else
/* v73+: the accumulator is drained through a bias/scale block in VTCM (R2 = its address): bias = mxmem2, cvt.hf = acc(#2), mxmem = cvt (see Hexagon SDK libs/qhl_hmx) */
.p2align 4,,15
.global hmx_store_cvt
.type hmx_store_cvt, @function
hmx_store_cvt:
{
    R6 = #2
}
{
    bias = mxmem2(R2)
}
{
    cvt.hf = acc(R6)
}
{
    mxmem(R0, R1) = cvt;
}
{
    jumpr R31;
}
.size hmx_store_cvt, .-hmx_store_cvt
#endif
