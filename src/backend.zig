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

    fn matmul_impl(self: *const Backend, a: tensor.Tensor, b: tensor.Tensor) root.TensorError !tensor.Tensor {
        _ = self; _ = a; _ = b;
        return error.NotImplemented;
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
