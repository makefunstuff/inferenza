const std = @import("std");
const inferenza = @import("inferenza");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    std.debug.print("Inferenza Engine Starting...\n", .{});

    // 1. Initialize the CPU Backend
    const cpu = try inferenza.backend.CpuBackend.init(allocator);
    defer cpu.deinit();

    // 2. Create some dummy data (Two 1D Tensors)
    const shape = [_]usize{ 3 };
    const strides = [_]usize{ 1 };
    
    const data_a = try allocator.alloc(f32, 3);
    defer allocator.free(data_a);

    const data_b = try allocator.alloc(f32, 3);
    defer allocator.free(data_b);

    const t1 = inferenza.tensor.Tensor{ 
        .data = data_a, 
        .shape = &shape, 
        .strides = &strides, 
        .backend = cpu 
    };
    const t2 = inferenza.tensor.Tensor{ 
        .data = data_b, 
        .shape = &shape, 
        .strides = &strides, 
        .backend = cpu 
    };

    // 3. Perform Inference (The Addition)
    std.debug.print("Performing t1 + t2 on CPU...\n", .{});
    const result = try t1.add(t2);
    defer allocator.free(result.data);

    // 4. Verify Result
    for (result.data, 0..) |val, i| {
        std.debug.print("Result[{d}] = {d}\n", .{ i, val });
    }
}
