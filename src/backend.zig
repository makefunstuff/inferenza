const tensor = @import("tensor.zig");
const root = @import("root.zig");
const std = @import("std");

pub const Backend = struct {
    ptr: *const anyopaque,
    allocator: std.mem.Allocator,
    add_fn: *const fn (self: *const Backend, a: tensor.Tensor, b: tensor.Tensor) root.TensorError !tensor.Tensor,
    matmul_fn: *const fn (self: *const Backend, a: tensor.Tensor, b: tensor.Tensor) root.TensorError !tensor.Tensor,

    pub fn add(self: *const Backend, a: tensor.Tensor, b: tensor.Tensor) !tensor.Tensor {
        return self.add_fn(self, a, b);
    }

    pub fn matmul(self: *const Backend, a: tensor.Tensor, b: tensor.Tensor) !tensor.Tensor {
        return self.matmul_fn(self, a, b);
    }

    pub fn deinit(self: *const Backend) void {
        _ = self;
    }
};

pub const CpuBackend = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*const Backend {
        const self = try allocator.create(CpuBackend);
        self.allocator = allocator;
        
        const interface = try allocator.create(Backend);
        interface.* = .{
            .ptr = self,
            .allocator = allocator,
            .add_fn = add,
            .matmul_fn = matmul_impl,
        };
        return interface;
    }

    const V = 8; // SIMD lane width (AVX f32)
    const Vf32 = @Vector(V, f32);
    const TILE = 64; // L1 cache tile size

    fn matmul_impl(self: *const Backend, a: tensor.Tensor, b: tensor.Tensor) root.TensorError !tensor.Tensor {
        if (a.shape.len != 2 or b.shape.len != 2) return root.TensorError.DimensionMismatch;
        const m = a.shape[0];
        const k = a.shape[1];
        if (k != b.shape[0]) return root.TensorError.DimensionMismatch;
        const n = b.shape[1];

        const c = try self.allocator.alloc(f32, m * n);
        @memset(c, 0);

        const result_shape = try self.allocator.dupe(usize, &.{ m, n });
        const result_strides = try self.allocator.dupe(usize, &.{ n, 1 });

        // Tiled matmul: C[i,j] += A[i,p] * B[p,j]
        // Outer tiles for cache locality, inner SIMD across N dimension
        var jj: usize = 0;
        while (jj < n) : (jj += TILE) {
            const j_end = @min(jj + TILE, n);
            var pp: usize = 0;
            while (pp < k) : (pp += TILE) {
                const p_end = @min(pp + TILE, k);
                for (0..m) |i| {
                    const a_base = i * a.strides[0];
                    const c_base = i * n;
                    for (pp..p_end) |p| {
                        const a_val: Vf32 = @splat(a.data[a_base + p * a.strides[1]]);
                        const b_base = p * b.strides[0];
                        // SIMD: 8 columns of C at once
                        var j = jj;
                        while (j + V <= j_end) : (j += V) {
                            const b_vec: Vf32 = b.data[b_base + j ..][0..V].*;
                            const c_vec: Vf32 = c[c_base + j ..][0..V].*;
                            c[c_base + j ..][0..V].* = c_vec + a_val * b_vec;
                        }
                        // Scalar tail
                        while (j < j_end) : (j += 1) {
                            c[c_base + j] += a.data[a_base + p * a.strides[1]] * b.data[b_base + j * b.strides[1]];
                        }
                    }
                }
            }
        }

        return tensor.Tensor{
            .data = c,
            .shape = result_shape,
            .strides = result_strides,
            .backend = self,
        };
    }

    fn add(self: *const Backend, a: tensor.Tensor, b: tensor.Tensor) root.TensorError !tensor.Tensor {
        const len = a.data.len;
        if (len != b.data.len) return root.TensorError.DimensionMismatch;

        const result_data = try self.allocator.alloc(f32, len);
        for (0..len) |i| {
            result_data[i] = a.data[i] + b.data[i];
        }

        return tensor.Tensor{
            .data = result_data,
            .shape = a.shape,
            .strides = a.strides,
            .backend = self,
        };
    }
};

const testing = std.testing;

test "add: element-wise" {
    const alloc = testing.allocator;
    const be = try CpuBackend.init(alloc);
    defer alloc.destroy(@as(*const CpuBackend, @ptrCast(@alignCast(be.ptr))));
    defer alloc.destroy(be);

    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 5, 6, 7, 8 };
    const shape = &[_]usize{ 2, 2 };
    const strides = &[_]usize{ 2, 1 };

    const a = tensor.Tensor{ .data = &a_data, .shape = shape, .strides = strides, .backend = be };
    const b = tensor.Tensor{ .data = &b_data, .shape = shape, .strides = strides, .backend = be };

    const c = try a.add(b);
    defer alloc.free(c.data);

    try testing.expectEqualSlices(f32, &[_]f32{ 6, 8, 10, 12 }, c.data);
}

test "add: dimension mismatch" {
    const alloc = testing.allocator;
    const be = try CpuBackend.init(alloc);
    defer alloc.destroy(@as(*const CpuBackend, @ptrCast(@alignCast(be.ptr))));
    defer alloc.destroy(be);

    var a_data = [_]f32{ 1, 2, 3 };
    var b_data = [_]f32{ 1, 2 };
    const a = tensor.Tensor{ .data = &a_data, .shape = &[_]usize{3}, .strides = &[_]usize{1}, .backend = be };
    const b = tensor.Tensor{ .data = &b_data, .shape = &[_]usize{2}, .strides = &[_]usize{1}, .backend = be };

    try testing.expectError(root.TensorError.DimensionMismatch, a.add(b));
}

test "matmul: 2x3 * 3x2" {
    const alloc = testing.allocator;
    const be = try CpuBackend.init(alloc);
    defer alloc.destroy(@as(*const CpuBackend, @ptrCast(@alignCast(be.ptr))));
    defer alloc.destroy(be);

    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };

    const a = tensor.Tensor{ .data = &a_data, .shape = &[_]usize{ 2, 3 }, .strides = &[_]usize{ 3, 1 }, .backend = be };
    const b = tensor.Tensor{ .data = &b_data, .shape = &[_]usize{ 3, 2 }, .strides = &[_]usize{ 2, 1 }, .backend = be };

    const c = try a.matmul(b);
    defer alloc.free(c.data);
    defer alloc.free(@constCast(c.shape));
    defer alloc.free(@constCast(c.strides));

    try testing.expectEqualSlices(usize, &[_]usize{ 2, 2 }, c.shape);
    for ([_]f32{ 58, 64, 139, 154 }, c.data) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-4);
    }
}

test "matmul: identity" {
    const alloc = testing.allocator;
    const be = try CpuBackend.init(alloc);
    defer alloc.destroy(@as(*const CpuBackend, @ptrCast(@alignCast(be.ptr))));
    defer alloc.destroy(be);

    var a_data = [_]f32{ 3, 7, 2, 5 };
    var i_data = [_]f32{ 1, 0, 0, 1 };

    const a = tensor.Tensor{ .data = &a_data, .shape = &[_]usize{ 2, 2 }, .strides = &[_]usize{ 2, 1 }, .backend = be };
    const eye = tensor.Tensor{ .data = &i_data, .shape = &[_]usize{ 2, 2 }, .strides = &[_]usize{ 2, 1 }, .backend = be };

    const c = try a.matmul(eye);
    defer alloc.free(c.data);
    defer alloc.free(@constCast(c.shape));
    defer alloc.free(@constCast(c.strides));

    for (a_data[0..], c.data) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-4);
    }
}

test "matmul: non-aligned N (exercises SIMD tail)" {
    const alloc = testing.allocator;
    const be = try CpuBackend.init(alloc);
    defer alloc.destroy(@as(*const CpuBackend, @ptrCast(@alignCast(be.ptr))));
    defer alloc.destroy(be);

    const k = 13;
    var a_data: [k]f32 = undefined;
    var b_data: [k]f32 = undefined;
    var expected: f32 = 0;
    for (0..k) |idx| {
        const v: f32 = @floatFromInt(idx + 1);
        a_data[idx] = v;
        b_data[idx] = v;
        expected += v * v;
    }

    const a = tensor.Tensor{ .data = &a_data, .shape = &[_]usize{ 1, k }, .strides = &[_]usize{ k, 1 }, .backend = be };
    const b = tensor.Tensor{ .data = &b_data, .shape = &[_]usize{ k, 1 }, .strides = &[_]usize{ 1, 1 }, .backend = be };

    const c = try a.matmul(b);
    defer alloc.free(c.data);
    defer alloc.free(@constCast(c.shape));
    defer alloc.free(@constCast(c.strides));

    try testing.expectApproxEqAbs(expected, c.data[0], 1e-4);
}

test "matmul: dimension mismatch" {
    const alloc = testing.allocator;
    const be = try CpuBackend.init(alloc);
    defer alloc.destroy(@as(*const CpuBackend, @ptrCast(@alignCast(be.ptr))));
    defer alloc.destroy(be);

    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    const a = tensor.Tensor{ .data = &a_data, .shape = &[_]usize{ 2, 2 }, .strides = &[_]usize{ 2, 1 }, .backend = be };
    const b = tensor.Tensor{ .data = &b_data, .shape = &[_]usize{ 3, 2 }, .strides = &[_]usize{ 2, 1 }, .backend = be };

    try testing.expectError(root.TensorError.DimensionMismatch, a.matmul(b));
}
