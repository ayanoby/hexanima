.p2align 4,,15
.global hmx8_tile
.type hmx8_tile, @function
hmx8_tile:
{
    activation.ub = mxmem(R0, R1):deep;
    weight.b = mxmem(R2, R3);
}
{
    mxmem(R4, R5):after.ub = acc;
}
{
    jumpr R31;
}
.size hmx8_tile, .-hmx8_tile
.p2align 4,,15
.global hmx8_acc
.type hmx8_acc, @function
hmx8_acc:
{
    activation.ub = mxmem(R0, R1):deep;
    weight.b = mxmem(R2, R3);
}
{
    jumpr R31;
}
.size hmx8_acc, .-hmx8_acc
.p2align 4,,15
.global hmx8_store
.type hmx8_store, @function
hmx8_store:
{
    mxmem(R0, R1):after.ub = acc;
}
{
    jumpr R31;
}
.size hmx8_store, .-hmx8_store
.p2align 4,,15
.global hmx8_store_before
.type hmx8_store_before, @function
hmx8_store_before:
{
    mxmem(R0, R1):before.ub = acc;
}
{
    jumpr R31;
}
.size hmx8_store_before, .-hmx8_store_before
.p2align 4,,15
.global hmx8_bias
.type hmx8_bias, @function
hmx8_bias:
{
    bias = mxmem(R0);
}
{
    jumpr R31;
}
.size hmx8_bias, .-hmx8_bias
.p2align 4,,15
.global hmx8_st_sat
.type hmx8_st_sat, @function
hmx8_st_sat:
{
    mxmem(R0, R1):after:sat.ub = acc;
}
{
    jumpr R31;
}
.size hmx8_st_sat, .-hmx8_st_sat
.p2align 4,,15
.global hmx8_st_uh21
.type hmx8_st_uh21, @function
hmx8_st_uh21:
{
    mxmem(R0, R1):after.uh = acc:2x1;
}
{
    jumpr R31;
}
.size hmx8_st_uh21, .-hmx8_st_uh21
.p2align 4,,15
.global hmx8_st_uh22
.type hmx8_st_uh22, @function
hmx8_st_uh22:
{
    mxmem(R0, R1):after.uh = acc:2x2;
}
{
    jumpr R31;
}
.size hmx8_st_uh22, .-hmx8_st_uh22
.p2align 4,,15
.global hmx8_st_ret
.type hmx8_st_ret, @function
hmx8_st_ret:
{
    mxmem(R0, R1):after:retain.ub = acc;
}
{
    jumpr R31;
}
.size hmx8_st_ret, .-hmx8_st_ret
.p2align 4,,15
.global hmx8_clr
.type hmx8_clr, @function
hmx8_clr:
{
    mxclracc
}
{
    jumpr R31;
}
.size hmx8_clr, .-hmx8_clr
.p2align 4,,15
.global hmx8_bias2
.type hmx8_bias2, @function
hmx8_bias2:
{
    bias = mxmem2(R0);
}
{
    jumpr R31;
}
.size hmx8_bias2, .-hmx8_bias2
.p2align 4,,15
.global hmx8_acc_cm
.type hmx8_acc_cm, @function
hmx8_acc_cm:
{
    activation.ub = mxmem(R0, R1):cm;
    weight.b = mxmem(R2, R3);
}
{
    jumpr R31;
}
.size hmx8_acc_cm, .-hmx8_acc_cm
.p2align 4,,15
.global hmx8_st_cmsat
.type hmx8_st_cmsat, @function
hmx8_st_cmsat:
{
    mxmem(R0, R1):after:cm:sat.ub = acc;
}
{
    jumpr R31;
}
.size hmx8_st_cmsat, .-hmx8_st_cmsat
.p2align 4,,15
.global hmx_clr
.type hmx_clr, @function
hmx_clr:
{
    mxclracc.hf
}
{
    jumpr R31;
}
.size hmx_clr, .-hmx_clr
