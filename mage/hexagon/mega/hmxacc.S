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
