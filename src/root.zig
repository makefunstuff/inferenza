pub const TensorError = error{
    DimensionMismatch,
    OutOfMemory,
    InvalidShape,
    BackendError,
    NotImplemented,
};

pub const tensor = @import("tensor.zig");
pub const backend = @import("backend.zig");
