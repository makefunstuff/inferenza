const std = @import("std");
const root = @import("root.zig");
const backend = @import("backend.zig");

pub const Tensor = struct {
    data: []f32,
    shape: []const usize,
    strides: []const usize,
    backend: *const backend.Backend,

    pub fn add(self: Tensor, other: Tensor) !Tensor {
        return self.backend.add(self, other);
    }

    pub fn matmul(self: Tensor, other: Tensor) !Tensor {
        return self.backend.matmul(self, other);
    }
};
