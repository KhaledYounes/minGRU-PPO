import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def log_add_exp(a, b):
    if libdevice.isnan(b):
        return b
    if libdevice.isnan(a):
        return a
    max_val = tl.maximum(a, b)
    min_val = tl.minimum(a, b)
    if (min_val == float("inf")) and (min_val == max_val):
        return max_val
    if min_val == float("-inf"):
        return max_val
    diff = min_val - max_val
    return max_val + libdevice.log1p(tl.exp(diff))


@triton.jit
def fwd_kernel(logc_ptr, logv_ptr, t_ptr, z_ptr, out_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask = offs_t < T
    logc = tl.load(logc_ptr + pid * T + offs_t, mask=mask, other=0.0)
    logv = tl.load(logv_ptr + pid * T + offs_t, mask=mask, other=0.0)
    a_star = tl.cumsum(logc, axis=0)
    temp = logv - a_star
    z = tl.associative_scan(temp, axis=0, combine_fn=log_add_exp)
    out = tl.exp(a_star + z)
    tl.store(t_ptr + pid * T + offs_t, temp, mask=mask)
    tl.store(z_ptr + pid * T + offs_t, z, mask=mask)
    tl.store(out_ptr + pid * T + offs_t, out, mask=mask)


@triton.jit
def bwd_kernel(t_ptr, z_ptr, out_ptr, grad_out_ptr, gradc_ptr, gradv_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask = offs_t < T
    temp = tl.load(t_ptr + pid * T + offs_t, mask=mask, other=0.0)
    z = tl.load(z_ptr + pid * T + offs_t, mask=mask, other=0.0)
    h_out = tl.load(out_ptr + pid * T + offs_t, mask=mask, other=0.0)
    grad = tl.load(grad_out_ptr + pid * T + offs_t, mask=mask, other=0.0)
    g_logh = grad * h_out
    ell = libdevice.log(tl.abs(g_logh)) - z
    ninf = -float("inf")
    log_pos = tl.where(g_logh > 0, ell, ninf)
    log_neg = tl.where(g_logh < 0, ell, ninf)
    rp = tl.associative_scan(log_pos, axis=0, combine_fn=log_add_exp, reverse=True)
    rn = tl.associative_scan(log_neg, axis=0, combine_fn=log_add_exp, reverse=True)
    g_temp = tl.exp(temp + rp) - tl.exp(temp + rn)
    g_logv = g_temp
    g_ast = g_logh - g_temp
    g_logc = tl.cumsum(g_ast, axis=0, reverse=True)
    tl.store(gradc_ptr + pid * T + offs_t, g_logc, mask=mask)
    tl.store(gradv_ptr + pid * T + offs_t, g_logv, mask=mask)


class TritonParallelScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_c, log_v):
        B, T, H = log_c.shape
        log_c_flat = log_c.permute(0, 2, 1).contiguous().reshape(-1, T)
        log_v_flat = log_v.permute(0, 2, 1).contiguous().reshape(-1, T)
        rows = log_c_flat.shape[0]
        BLOCK_T = triton.next_power_of_2(T)
        temp = torch.empty_like(log_c_flat)
        zbuf = torch.empty_like(log_c_flat)
        out = torch.empty_like(log_c_flat)
        fwd_kernel[(rows,)](
            log_c_flat,
            log_v_flat,
            temp,
            zbuf,
            out,
            T,
            BLOCK_T,
            num_warps=8,
            num_stages=2,
        )
        ctx.save_for_backward(temp, zbuf, out)
        ctx.shape = (B, H, T)
        ctx.BLOCK_T = BLOCK_T
        return out.reshape(B, H, T).permute(0, 2, 1)

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T = ctx.shape
        temp, zbuf, out = ctx.saved_tensors
        grad_out_flat = grad_out.permute(0, 2, 1).contiguous().reshape(-1, T)
        gradc = torch.empty_like(out)
        gradv = torch.empty_like(out)
        bwd_kernel[(out.shape[0],)](
            temp,
            zbuf,
            out,
            grad_out_flat,
            gradc,
            gradv,
            T,
            ctx.BLOCK_T,
            num_warps=8,
            num_stages=2,
        )
        return (
            gradc.reshape(B, H, T).permute(0, 2, 1),
            gradv.reshape(B, H, T).permute(0, 2, 1),
        )


prefix_scan = TritonParallelScan.apply
