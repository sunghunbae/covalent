#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()



extern "C" __global__ void _expand_naive_shifts_e946adb1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_shift_offset,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shifts,
    wp::array_t<wp::int32> var_shift_system_idx)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::vec_t<3, wp::int32>* var_4;
        wp::vec_t<3, wp::int32> var_5;
        wp::vec_t<3, wp::int32> var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        wp::range_t var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::range_t var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 2;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 2;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        bool var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        bool var_35;
        const wp::int32 var_36 = 0;
        bool var_37;
        const wp::int32 var_38 = 0;
        bool var_39;
        bool var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        wp::vec_t<3, wp::int32> var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32 var_50;
        //---------
        // forward
        // def _expand_naive_shifts(                                                              <L 69>
        // tid = wp.tid()                                                                         <L 100>
        var_0 = builtin_tid1d();
        // pos = shift_offset[tid]                                                                <L 101>
        var_1 = wp::address(var_shift_offset, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // _shift_range = shift_range[tid]                                                        <L 102>
        var_4 = wp::address(var_shift_range, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // for k0 in range(0, _shift_range[0] + 1):                                               <L 103>
        var_8 = wp::extract(var_5, var_7);
        var_10 = wp::add(var_8, var_9);
        var_12 = wp::range(var_11, var_10);
        start_for_0:;
            if (iter_cmp(var_12) == 0) goto end_for_0;
            var_13 = wp::iter_next(var_12);
            // for k1 in range(-_shift_range[1], _shift_range[1] + 1):                            <L 104>
            var_15 = wp::extract(var_5, var_14);
            var_16 = wp::neg(var_15);
            var_18 = wp::extract(var_5, var_17);
            var_20 = wp::add(var_18, var_19);
            var_21 = wp::range(var_16, var_20);
            start_for_2:;
                if (iter_cmp(var_21) == 0) goto end_for_2;
                var_22 = wp::iter_next(var_21);
                // for k2 in range(-_shift_range[2], _shift_range[2] + 1):                        <L 105>
                var_24 = wp::extract(var_5, var_23);
                var_25 = wp::neg(var_24);
                var_27 = wp::extract(var_5, var_26);
                var_29 = wp::add(var_27, var_28);
                var_30 = wp::range(var_25, var_29);
                start_for_4:;
                    if (iter_cmp(var_30) == 0) goto end_for_4;
                    var_31 = wp::iter_next(var_30);
                    // if k0 > 0 or (k0 == 0 and k1 > 0) or (k0 == 0 and k1 == 0 and k2 >= 0):       <L 106>
                    var_34 = (var_13 > var_33);
                    var_32 = var_34;
                    if (!var_32) {
                        var_37 = (var_13 == var_36);
                        var_35 = var_37;
                        if (var_35) {
                            var_39 = (var_22 > var_38);
                            var_35 = var_35 && var_39;
                        }
                        var_32 = var_32 || var_35;
                    }
                    if (!var_32) {
                        var_42 = (var_13 == var_41);
                        var_40 = var_42;
                        if (var_40) {
                            var_44 = (var_22 == var_43);
                            var_40 = var_40 && var_44;
                        }
                        if (var_40) {
                            var_46 = (var_31 >= var_45);
                            var_40 = var_40 && var_46;
                        }
                        var_32 = var_32 || var_40;
                    }
                    if (var_32) {
                        // shifts[pos] = wp.vec3i(k0, k1, k2)                                     <L 107>
                        var_47 = wp::vec_t<3, wp::int32>(var_13, var_22, var_31);
                        wp::array_store(var_shifts, var_2, var_47);
                        // shift_system_idx[pos] = tid                                            <L 108>
                        wp::array_store(var_shift_system_idx, var_2, var_0);
                        // pos += 1                                                               <L 109>
                        var_49 = wp::add(var_2, var_48);
                    }
                    var_50 = wp::where(var_32, var_49, var_2);
                    wp::assign(var_2, var_50);
                    goto start_for_4;
                end_for_4:;
                goto start_for_2;
            end_for_2:;
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _expand_naive_shifts_selective_216c85f5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_shift_offset,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shifts,
    wp::array_t<wp::int32> var_shift_system_idx,
    wp::array_t<bool> var_rebuild_flags)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        bool* var_1;
        bool var_2;
        bool var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::vec_t<3, wp::int32>* var_7;
        wp::vec_t<3, wp::int32> var_8;
        wp::vec_t<3, wp::int32> var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        const wp::int32 var_12 = 1;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        wp::range_t var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        wp::range_t var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 2;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::range_t var_33;
        wp::int32 var_34;
        bool var_35;
        const wp::int32 var_36 = 0;
        bool var_37;
        bool var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        bool var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        const wp::int32 var_46 = 0;
        bool var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        wp::vec_t<3, wp::int32> var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        wp::int32 var_53;
        //---------
        // forward
        // def _expand_naive_shifts_selective(                                                    <L 113>
        // tid = wp.tid()                                                                         <L 145>
        var_0 = builtin_tid1d();
        // if not rebuild_flags[tid]:                                                             <L 146>
        var_1 = wp::address(var_rebuild_flags, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::unot(var_3);
        if (var_2) {
            // return                                                                             <L 147>
            continue;
        }
        // pos = shift_offset[tid]                                                                <L 148>
        var_4 = wp::address(var_shift_offset, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // _shift_range = shift_range[tid]                                                        <L 149>
        var_7 = wp::address(var_shift_range, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for k0 in range(0, _shift_range[0] + 1):                                               <L 150>
        var_11 = wp::extract(var_8, var_10);
        var_13 = wp::add(var_11, var_12);
        var_15 = wp::range(var_14, var_13);
        start_for_1:;
            if (iter_cmp(var_15) == 0) goto end_for_1;
            var_16 = wp::iter_next(var_15);
            // for k1 in range(-_shift_range[1], _shift_range[1] + 1):                            <L 151>
            var_18 = wp::extract(var_8, var_17);
            var_19 = wp::neg(var_18);
            var_21 = wp::extract(var_8, var_20);
            var_23 = wp::add(var_21, var_22);
            var_24 = wp::range(var_19, var_23);
            start_for_3:;
                if (iter_cmp(var_24) == 0) goto end_for_3;
                var_25 = wp::iter_next(var_24);
                // for k2 in range(-_shift_range[2], _shift_range[2] + 1):                        <L 152>
                var_27 = wp::extract(var_8, var_26);
                var_28 = wp::neg(var_27);
                var_30 = wp::extract(var_8, var_29);
                var_32 = wp::add(var_30, var_31);
                var_33 = wp::range(var_28, var_32);
                start_for_5:;
                    if (iter_cmp(var_33) == 0) goto end_for_5;
                    var_34 = wp::iter_next(var_33);
                    // if k0 > 0 or (k0 == 0 and k1 > 0) or (k0 == 0 and k1 == 0 and k2 >= 0):       <L 153>
                    var_37 = (var_16 > var_36);
                    var_35 = var_37;
                    if (!var_35) {
                        var_40 = (var_16 == var_39);
                        var_38 = var_40;
                        if (var_38) {
                            var_42 = (var_25 > var_41);
                            var_38 = var_38 && var_42;
                        }
                        var_35 = var_35 || var_38;
                    }
                    if (!var_35) {
                        var_45 = (var_16 == var_44);
                        var_43 = var_45;
                        if (var_43) {
                            var_47 = (var_25 == var_46);
                            var_43 = var_43 && var_47;
                        }
                        if (var_43) {
                            var_49 = (var_34 >= var_48);
                            var_43 = var_43 && var_49;
                        }
                        var_35 = var_35 || var_43;
                    }
                    if (var_35) {
                        // shifts[pos] = wp.vec3i(k0, k1, k2)                                     <L 154>
                        var_50 = wp::vec_t<3, wp::int32>(var_16, var_25, var_34);
                        wp::array_store(var_shifts, var_5, var_50);
                        // shift_system_idx[pos] = tid                                            <L 155>
                        wp::array_store(var_shift_system_idx, var_5, var_0);
                        // pos += 1                                                               <L 156>
                        var_52 = wp::add(var_5, var_51);
                    }
                    var_53 = wp::where(var_35, var_52, var_5);
                    wp::assign(var_5, var_53);
                    goto start_for_5;
                end_for_5:;
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void _wrap_positions_single_kernel_4a7bdef1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_inv_cell,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions_wrapped,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::mat_t<3, 3, wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::mat_t<3, 3, wp::float32> var_4;
        const wp::int32 var_5 = 0;
        wp::mat_t<3, 3, wp::float32>* var_6;
        wp::mat_t<3, 3, wp::float32> var_7;
        wp::mat_t<3, 3, wp::float32> var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        const wp::int32 var_13 = 0;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 2;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::int32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        //---------
        // forward
        // def _wrap_positions_single_kernel(                                                     <L 1>
        // i = wp.tid()                                                                           <L 33>
        var_0 = builtin_tid1d();
        // _cell = cell[0]                                                                        <L 34>
        var_2 = wp::address(var_cell, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // _inv_cell = inv_cell[0]                                                                <L 35>
        var_6 = wp::address(var_inv_cell, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // _pos = positions[i]                                                                    <L 36>
        var_9 = wp::address(var_positions, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // _frac = _pos * _inv_cell                                                               <L 37>
        var_12 = wp::mul(var_10, var_7);
        // _int = wp.vec3i(                                                                       <L 38>
        // wp.int32(wp.floor(_frac[0])),                                                          <L 39>
        var_14 = wp::extract(var_12, var_13);
        var_15 = wp::floor(var_14);
        var_16 = wp::int32(var_15);
        // wp.int32(wp.floor(_frac[1])),                                                          <L 40>
        var_18 = wp::extract(var_12, var_17);
        var_19 = wp::floor(var_18);
        var_20 = wp::int32(var_19);
        // wp.int32(wp.floor(_frac[2])),                                                          <L 41>
        var_22 = wp::extract(var_12, var_21);
        var_23 = wp::floor(var_22);
        var_24 = wp::int32(var_23);
        var_25 = wp::vec_t<3, wp::int32>(var_16, var_20, var_24);
        // positions_wrapped[i] = _pos - type(_pos)(_int) * _cell                                 <L 43>
        var_26 = wp::vec_t<3, wp::float32>(var_25);
        var_27 = wp::mul(var_26, var_3);
        var_28 = wp::sub(var_10, var_27);
        wp::array_store(var_positions_wrapped, var_0, var_28);
        // per_atom_cell_offsets[i] = _int                                                        <L 44>
        wp::array_store(var_per_atom_cell_offsets, var_0, var_25);
    }
}



extern "C" __global__ void _wrap_positions_single_kernel_259846de_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_inv_cell,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions_wrapped,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::mat_t<3, 3, wp::float64>* var_2;
        wp::mat_t<3, 3, wp::float64> var_3;
        wp::mat_t<3, 3, wp::float64> var_4;
        const wp::int32 var_5 = 0;
        wp::mat_t<3, 3, wp::float64>* var_6;
        wp::mat_t<3, 3, wp::float64> var_7;
        wp::mat_t<3, 3, wp::float64> var_8;
        wp::vec_t<3, wp::float64>* var_9;
        wp::vec_t<3, wp::float64> var_10;
        wp::vec_t<3, wp::float64> var_11;
        wp::vec_t<3, wp::float64> var_12;
        const wp::int32 var_13 = 0;
        wp::float64 var_14;
        wp::float64 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::float64 var_18;
        wp::float64 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 2;
        wp::float64 var_22;
        wp::float64 var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::int32> var_25;
        wp::vec_t<3, wp::float64> var_26;
        wp::vec_t<3, wp::float64> var_27;
        wp::vec_t<3, wp::float64> var_28;
        //---------
        // forward
        // def _wrap_positions_single_kernel(                                                     <L 1>
        // i = wp.tid()                                                                           <L 33>
        var_0 = builtin_tid1d();
        // _cell = cell[0]                                                                        <L 34>
        var_2 = wp::address(var_cell, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // _inv_cell = inv_cell[0]                                                                <L 35>
        var_6 = wp::address(var_inv_cell, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // _pos = positions[i]                                                                    <L 36>
        var_9 = wp::address(var_positions, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // _frac = _pos * _inv_cell                                                               <L 37>
        var_12 = wp::mul(var_10, var_7);
        // _int = wp.vec3i(                                                                       <L 38>
        // wp.int32(wp.floor(_frac[0])),                                                          <L 39>
        var_14 = wp::extract(var_12, var_13);
        var_15 = wp::floor(var_14);
        var_16 = wp::int32(var_15);
        // wp.int32(wp.floor(_frac[1])),                                                          <L 40>
        var_18 = wp::extract(var_12, var_17);
        var_19 = wp::floor(var_18);
        var_20 = wp::int32(var_19);
        // wp.int32(wp.floor(_frac[2])),                                                          <L 41>
        var_22 = wp::extract(var_12, var_21);
        var_23 = wp::floor(var_22);
        var_24 = wp::int32(var_23);
        var_25 = wp::vec_t<3, wp::int32>(var_16, var_20, var_24);
        // positions_wrapped[i] = _pos - type(_pos)(_int) * _cell                                 <L 43>
        var_26 = wp::vec_t<3, wp::float64>(var_25);
        var_27 = wp::mul(var_26, var_3);
        var_28 = wp::sub(var_10, var_27);
        wp::array_store(var_positions_wrapped, var_0, var_28);
        // per_atom_cell_offsets[i] = _int                                                        <L 44>
        wp::array_store(var_per_atom_cell_offsets, var_0, var_25);
    }
}



extern "C" __global__ void _wrap_positions_single_kernel_19055b7b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_inv_cell,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions_wrapped,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::mat_t<3, 3, wp::float16>* var_2;
        wp::mat_t<3, 3, wp::float16> var_3;
        wp::mat_t<3, 3, wp::float16> var_4;
        const wp::int32 var_5 = 0;
        wp::mat_t<3, 3, wp::float16>* var_6;
        wp::mat_t<3, 3, wp::float16> var_7;
        wp::mat_t<3, 3, wp::float16> var_8;
        wp::vec_t<3, wp::float16>* var_9;
        wp::vec_t<3, wp::float16> var_10;
        wp::vec_t<3, wp::float16> var_11;
        wp::vec_t<3, wp::float16> var_12;
        const wp::int32 var_13 = 0;
        wp::float16 var_14;
        wp::float16 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::float16 var_18;
        wp::float16 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 2;
        wp::float16 var_22;
        wp::float16 var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::int32> var_25;
        wp::vec_t<3, wp::float16> var_26;
        wp::vec_t<3, wp::float16> var_27;
        wp::vec_t<3, wp::float16> var_28;
        //---------
        // forward
        // def _wrap_positions_single_kernel(                                                     <L 1>
        // i = wp.tid()                                                                           <L 33>
        var_0 = builtin_tid1d();
        // _cell = cell[0]                                                                        <L 34>
        var_2 = wp::address(var_cell, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // _inv_cell = inv_cell[0]                                                                <L 35>
        var_6 = wp::address(var_inv_cell, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // _pos = positions[i]                                                                    <L 36>
        var_9 = wp::address(var_positions, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // _frac = _pos * _inv_cell                                                               <L 37>
        var_12 = wp::mul(var_10, var_7);
        // _int = wp.vec3i(                                                                       <L 38>
        // wp.int32(wp.floor(_frac[0])),                                                          <L 39>
        var_14 = wp::extract(var_12, var_13);
        var_15 = wp::floor(var_14);
        var_16 = wp::int32(var_15);
        // wp.int32(wp.floor(_frac[1])),                                                          <L 40>
        var_18 = wp::extract(var_12, var_17);
        var_19 = wp::floor(var_18);
        var_20 = wp::int32(var_19);
        // wp.int32(wp.floor(_frac[2])),                                                          <L 41>
        var_22 = wp::extract(var_12, var_21);
        var_23 = wp::floor(var_22);
        var_24 = wp::int32(var_23);
        var_25 = wp::vec_t<3, wp::int32>(var_16, var_20, var_24);
        // positions_wrapped[i] = _pos - type(_pos)(_int) * _cell                                 <L 43>
        var_26 = wp::vec_t<3, wp::float16>(var_25);
        var_27 = wp::mul(var_26, var_3);
        var_28 = wp::sub(var_10, var_27);
        wp::array_store(var_positions_wrapped, var_0, var_28);
        // per_atom_cell_offsets[i] = _int                                                        <L 44>
        wp::array_store(var_per_atom_cell_offsets, var_0, var_25);
    }
}



extern "C" __global__ void _compute_naive_num_shifts_289b8d98_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::float32 var_cutoff,
    wp::array_t<bool> var_pbc,
    wp::array_t<wp::int32> var_num_shifts,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3, 3, wp::float32>* var_1;
        wp::mat_t<3, 3, wp::float32> var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::slice_t var_4;
        const wp::int32 var_5 = 0;
        wp::array_t<bool> var_6;
        wp::mat_t<3, 3, wp::float32> var_7;
        wp::mat_t<3, 3, wp::float32> var_8;
        const wp::int32 var_9 = 0;
        bool* var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        wp::vec_t<3, wp::float32> var_13;
        wp::float32 var_14;
        bool var_15;
        bool var_16;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 0;
        wp::float32 var_19;
        const wp::float32 var_20 = 0.0;
        wp::float32 var_21;
        bool var_22;
        wp::float32 var_23;
        bool var_24;
        const wp::int32 var_25 = 1;
        bool* var_26;
        bool var_27;
        const wp::int32 var_28 = 1;
        wp::vec_t<3, wp::float32> var_29;
        wp::float32 var_30;
        bool var_31;
        bool var_32;
        const wp::int32 var_33 = 1;
        const wp::int32 var_34 = 0;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::float32 var_37;
        bool var_38;
        wp::float32 var_39;
        bool var_40;
        const wp::int32 var_41 = 2;
        bool* var_42;
        bool var_43;
        const wp::int32 var_44 = 2;
        wp::vec_t<3, wp::float32> var_45;
        wp::float32 var_46;
        bool var_47;
        bool var_48;
        const wp::int32 var_49 = 2;
        const wp::int32 var_50 = 0;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.0;
        wp::float32 var_53;
        bool var_54;
        wp::float32 var_55;
        bool var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::int32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::int32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::int32 var_68;
        wp::vec_t<3, wp::int32> var_69;
        const wp::int32 var_70 = 2;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::int32 var_73;
        const wp::int32 var_74 = 1;
        wp::int32 var_75;
        const wp::int32 var_76 = 2;
        const wp::int32 var_77 = 2;
        wp::int32 var_78;
        wp::int32 var_79;
        const wp::int32 var_80 = 1;
        wp::int32 var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 1;
        wp::int32 var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        const wp::int32 var_90 = 2;
        wp::int32 var_91;
        wp::int32 var_92;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        //---------
        // forward
        // def _compute_naive_num_shifts(                                                         <L 1>
        // tid = wp.tid()                                                                         <L 44>
        var_0 = builtin_tid1d();
        // _cell = cell[tid]                                                                      <L 46>
        var_1 = wp::address(var_cell, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // _pbc = pbc[tid]                                                                        <L 47>
        var_4 = wp::slice_t(var_0, var_0, var_5);
        var_6 = wp::view(var_pbc, var_4);
        // _cell_inv = wp.transpose(wp.inverse(_cell))                                            <L 49>
        var_7 = wp::inverse(var_2);
        var_8 = wp::transpose(var_7);
        // _d_inv_0 = wp.length(_cell_inv[0]) if _pbc[0] else type(_cell_inv[0, 0])(0.0)          <L 50>
        var_10 = wp::address(var_6, var_9);
        var_11 = wp::load(var_10);
        if (var_11) {
            var_13 = wp::extract(var_8, var_12);
            var_14 = wp::length(var_13);
        }
        var_15 = wp::load(var_10);
        var_16 = wp::load(var_10);
        if (!var_16) {
            var_19 = wp::extract(var_8, var_17, var_18);
            var_21 = wp::float32(var_20);
        }
        var_22 = wp::load(var_10);
        var_24 = wp::load(var_10);
        var_23 = wp::where(var_24, var_14, var_21);
        // _d_inv_1 = wp.length(_cell_inv[1]) if _pbc[1] else type(_cell_inv[1, 0])(0.0)          <L 51>
        var_26 = wp::address(var_6, var_25);
        var_27 = wp::load(var_26);
        if (var_27) {
            var_29 = wp::extract(var_8, var_28);
            var_30 = wp::length(var_29);
        }
        var_31 = wp::load(var_26);
        var_32 = wp::load(var_26);
        if (!var_32) {
            var_35 = wp::extract(var_8, var_33, var_34);
            var_37 = wp::float32(var_36);
        }
        var_38 = wp::load(var_26);
        var_40 = wp::load(var_26);
        var_39 = wp::where(var_40, var_30, var_37);
        // _d_inv_2 = wp.length(_cell_inv[2]) if _pbc[2] else type(_cell_inv[2, 0])(0.0)          <L 52>
        var_42 = wp::address(var_6, var_41);
        var_43 = wp::load(var_42);
        if (var_43) {
            var_45 = wp::extract(var_8, var_44);
            var_46 = wp::length(var_45);
        }
        var_47 = wp::load(var_42);
        var_48 = wp::load(var_42);
        if (!var_48) {
            var_51 = wp::extract(var_8, var_49, var_50);
            var_53 = wp::float32(var_52);
        }
        var_54 = wp::load(var_42);
        var_56 = wp::load(var_42);
        var_55 = wp::where(var_56, var_46, var_53);
        // _s = wp.vec3i(                                                                         <L 53>
        // wp.int32(wp.ceil(_d_inv_0 * type(_d_inv_0)(cutoff))),                                  <L 54>
        var_57 = wp::float32(var_cutoff);
        var_58 = wp::mul(var_23, var_57);
        var_59 = wp::ceil(var_58);
        var_60 = wp::int32(var_59);
        // wp.int32(wp.ceil(_d_inv_1 * type(_d_inv_1)(cutoff))),                                  <L 55>
        var_61 = wp::float32(var_cutoff);
        var_62 = wp::mul(var_39, var_61);
        var_63 = wp::ceil(var_62);
        var_64 = wp::int32(var_63);
        // wp.int32(wp.ceil(_d_inv_2 * type(_d_inv_2)(cutoff))),                                  <L 56>
        var_65 = wp::float32(var_cutoff);
        var_66 = wp::mul(var_55, var_65);
        var_67 = wp::ceil(var_66);
        var_68 = wp::int32(var_67);
        var_69 = wp::vec_t<3, wp::int32>(var_60, var_64, var_68);
        // k1 = 2 * _s[1] + 1                                                                     <L 58>
        var_72 = wp::extract(var_69, var_71);
        var_73 = wp::mul(var_70, var_72);
        var_75 = wp::add(var_73, var_74);
        // k2 = 2 * _s[2] + 1                                                                     <L 59>
        var_78 = wp::extract(var_69, var_77);
        var_79 = wp::mul(var_76, var_78);
        var_81 = wp::add(var_79, var_80);
        // shift_range[tid] = _s                                                                  <L 60>
        wp::array_store(var_shift_range, var_0, var_69);
        // num_shifts[tid] = _s[0] * k1 * k2 + _s[1] * k2 + _s[2] + 1                             <L 61>
        var_83 = wp::extract(var_69, var_82);
        var_84 = wp::mul(var_83, var_75);
        var_85 = wp::mul(var_84, var_81);
        var_87 = wp::extract(var_69, var_86);
        var_88 = wp::mul(var_87, var_81);
        var_89 = wp::add(var_85, var_88);
        var_91 = wp::extract(var_69, var_90);
        var_92 = wp::add(var_89, var_91);
        var_94 = wp::add(var_92, var_93);
        wp::array_store(var_num_shifts, var_0, var_94);
    }
}



extern "C" __global__ void _compute_naive_num_shifts_ca2bc731_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::float64 var_cutoff,
    wp::array_t<bool> var_pbc,
    wp::array_t<wp::int32> var_num_shifts,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3, 3, wp::float64>* var_1;
        wp::mat_t<3, 3, wp::float64> var_2;
        wp::mat_t<3, 3, wp::float64> var_3;
        wp::slice_t var_4;
        const wp::int32 var_5 = 0;
        wp::array_t<bool> var_6;
        wp::mat_t<3, 3, wp::float64> var_7;
        wp::mat_t<3, 3, wp::float64> var_8;
        const wp::int32 var_9 = 0;
        bool* var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        wp::vec_t<3, wp::float64> var_13;
        wp::float64 var_14;
        bool var_15;
        bool var_16;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 0;
        wp::float64 var_19;
        wp::float64 var_20;
        bool var_21;
        wp::float64 var_22;
        bool var_23;
        const wp::int32 var_24 = 1;
        bool* var_25;
        bool var_26;
        const wp::int32 var_27 = 1;
        wp::vec_t<3, wp::float64> var_28;
        wp::float64 var_29;
        bool var_30;
        bool var_31;
        const wp::int32 var_32 = 1;
        const wp::int32 var_33 = 0;
        wp::float64 var_34;
        wp::float64 var_35;
        bool var_36;
        wp::float64 var_37;
        bool var_38;
        const wp::int32 var_39 = 2;
        bool* var_40;
        bool var_41;
        const wp::int32 var_42 = 2;
        wp::vec_t<3, wp::float64> var_43;
        wp::float64 var_44;
        bool var_45;
        bool var_46;
        const wp::int32 var_47 = 2;
        const wp::int32 var_48 = 0;
        wp::float64 var_49;
        wp::float64 var_50;
        bool var_51;
        wp::float64 var_52;
        bool var_53;
        wp::float64 var_54;
        wp::float64 var_55;
        wp::float64 var_56;
        wp::int32 var_57;
        wp::float64 var_58;
        wp::float64 var_59;
        wp::float64 var_60;
        wp::int32 var_61;
        wp::float64 var_62;
        wp::float64 var_63;
        wp::float64 var_64;
        wp::int32 var_65;
        wp::vec_t<3, wp::int32> var_66;
        const wp::int32 var_67 = 2;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::int32 var_70;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        const wp::int32 var_73 = 2;
        const wp::int32 var_74 = 2;
        wp::int32 var_75;
        wp::int32 var_76;
        const wp::int32 var_77 = 1;
        wp::int32 var_78;
        const wp::int32 var_79 = 0;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 1;
        wp::int32 var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        const wp::int32 var_87 = 2;
        wp::int32 var_88;
        wp::int32 var_89;
        const wp::int32 var_90 = 1;
        wp::int32 var_91;
        //---------
        // forward
        // def _compute_naive_num_shifts(                                                         <L 1>
        // tid = wp.tid()                                                                         <L 44>
        var_0 = builtin_tid1d();
        // _cell = cell[tid]                                                                      <L 46>
        var_1 = wp::address(var_cell, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // _pbc = pbc[tid]                                                                        <L 47>
        var_4 = wp::slice_t(var_0, var_0, var_5);
        var_6 = wp::view(var_pbc, var_4);
        // _cell_inv = wp.transpose(wp.inverse(_cell))                                            <L 49>
        var_7 = wp::inverse(var_2);
        var_8 = wp::transpose(var_7);
        // _d_inv_0 = wp.length(_cell_inv[0]) if _pbc[0] else type(_cell_inv[0, 0])(0.0)          <L 50>
        var_10 = wp::address(var_6, var_9);
        var_11 = wp::load(var_10);
        if (var_11) {
            var_13 = wp::extract(var_8, var_12);
            var_14 = wp::length(var_13);
        }
        var_15 = wp::load(var_10);
        var_16 = wp::load(var_10);
        if (!var_16) {
            var_19 = wp::extract(var_8, var_17, var_18);
            var_20 = 0.0;
        }
        var_21 = wp::load(var_10);
        var_23 = wp::load(var_10);
        var_22 = wp::where(var_23, var_14, var_20);
        // _d_inv_1 = wp.length(_cell_inv[1]) if _pbc[1] else type(_cell_inv[1, 0])(0.0)          <L 51>
        var_25 = wp::address(var_6, var_24);
        var_26 = wp::load(var_25);
        if (var_26) {
            var_28 = wp::extract(var_8, var_27);
            var_29 = wp::length(var_28);
        }
        var_30 = wp::load(var_25);
        var_31 = wp::load(var_25);
        if (!var_31) {
            var_34 = wp::extract(var_8, var_32, var_33);
            var_35 = 0.0;
        }
        var_36 = wp::load(var_25);
        var_38 = wp::load(var_25);
        var_37 = wp::where(var_38, var_29, var_35);
        // _d_inv_2 = wp.length(_cell_inv[2]) if _pbc[2] else type(_cell_inv[2, 0])(0.0)          <L 52>
        var_40 = wp::address(var_6, var_39);
        var_41 = wp::load(var_40);
        if (var_41) {
            var_43 = wp::extract(var_8, var_42);
            var_44 = wp::length(var_43);
        }
        var_45 = wp::load(var_40);
        var_46 = wp::load(var_40);
        if (!var_46) {
            var_49 = wp::extract(var_8, var_47, var_48);
            var_50 = 0.0;
        }
        var_51 = wp::load(var_40);
        var_53 = wp::load(var_40);
        var_52 = wp::where(var_53, var_44, var_50);
        // _s = wp.vec3i(                                                                         <L 53>
        // wp.int32(wp.ceil(_d_inv_0 * type(_d_inv_0)(cutoff))),                                  <L 54>
        var_54 = wp::float64(var_cutoff);
        var_55 = wp::mul(var_22, var_54);
        var_56 = wp::ceil(var_55);
        var_57 = wp::int32(var_56);
        // wp.int32(wp.ceil(_d_inv_1 * type(_d_inv_1)(cutoff))),                                  <L 55>
        var_58 = wp::float64(var_cutoff);
        var_59 = wp::mul(var_37, var_58);
        var_60 = wp::ceil(var_59);
        var_61 = wp::int32(var_60);
        // wp.int32(wp.ceil(_d_inv_2 * type(_d_inv_2)(cutoff))),                                  <L 56>
        var_62 = wp::float64(var_cutoff);
        var_63 = wp::mul(var_52, var_62);
        var_64 = wp::ceil(var_63);
        var_65 = wp::int32(var_64);
        var_66 = wp::vec_t<3, wp::int32>(var_57, var_61, var_65);
        // k1 = 2 * _s[1] + 1                                                                     <L 58>
        var_69 = wp::extract(var_66, var_68);
        var_70 = wp::mul(var_67, var_69);
        var_72 = wp::add(var_70, var_71);
        // k2 = 2 * _s[2] + 1                                                                     <L 59>
        var_75 = wp::extract(var_66, var_74);
        var_76 = wp::mul(var_73, var_75);
        var_78 = wp::add(var_76, var_77);
        // shift_range[tid] = _s                                                                  <L 60>
        wp::array_store(var_shift_range, var_0, var_66);
        // num_shifts[tid] = _s[0] * k1 * k2 + _s[1] * k2 + _s[2] + 1                             <L 61>
        var_80 = wp::extract(var_66, var_79);
        var_81 = wp::mul(var_80, var_72);
        var_82 = wp::mul(var_81, var_78);
        var_84 = wp::extract(var_66, var_83);
        var_85 = wp::mul(var_84, var_78);
        var_86 = wp::add(var_82, var_85);
        var_88 = wp::extract(var_66, var_87);
        var_89 = wp::add(var_86, var_88);
        var_91 = wp::add(var_89, var_90);
        wp::array_store(var_num_shifts, var_0, var_91);
    }
}



extern "C" __global__ void _compute_naive_num_shifts_b0707f7e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::float16 var_cutoff,
    wp::array_t<bool> var_pbc,
    wp::array_t<wp::int32> var_num_shifts,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3, 3, wp::float16>* var_1;
        wp::mat_t<3, 3, wp::float16> var_2;
        wp::mat_t<3, 3, wp::float16> var_3;
        wp::slice_t var_4;
        const wp::int32 var_5 = 0;
        wp::array_t<bool> var_6;
        wp::mat_t<3, 3, wp::float16> var_7;
        wp::mat_t<3, 3, wp::float16> var_8;
        const wp::int32 var_9 = 0;
        bool* var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        wp::vec_t<3, wp::float16> var_13;
        wp::float16 var_14;
        bool var_15;
        bool var_16;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 0;
        wp::float16 var_19;
        const wp::float32 var_20 = 0.0;
        wp::float16 var_21;
        bool var_22;
        wp::float16 var_23;
        bool var_24;
        const wp::int32 var_25 = 1;
        bool* var_26;
        bool var_27;
        const wp::int32 var_28 = 1;
        wp::vec_t<3, wp::float16> var_29;
        wp::float16 var_30;
        bool var_31;
        bool var_32;
        const wp::int32 var_33 = 1;
        const wp::int32 var_34 = 0;
        wp::float16 var_35;
        const wp::float32 var_36 = 0.0;
        wp::float16 var_37;
        bool var_38;
        wp::float16 var_39;
        bool var_40;
        const wp::int32 var_41 = 2;
        bool* var_42;
        bool var_43;
        const wp::int32 var_44 = 2;
        wp::vec_t<3, wp::float16> var_45;
        wp::float16 var_46;
        bool var_47;
        bool var_48;
        const wp::int32 var_49 = 2;
        const wp::int32 var_50 = 0;
        wp::float16 var_51;
        const wp::float32 var_52 = 0.0;
        wp::float16 var_53;
        bool var_54;
        wp::float16 var_55;
        bool var_56;
        wp::float16 var_57;
        wp::float16 var_58;
        wp::float16 var_59;
        wp::int32 var_60;
        wp::float16 var_61;
        wp::float16 var_62;
        wp::float16 var_63;
        wp::int32 var_64;
        wp::float16 var_65;
        wp::float16 var_66;
        wp::float16 var_67;
        wp::int32 var_68;
        wp::vec_t<3, wp::int32> var_69;
        const wp::int32 var_70 = 2;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::int32 var_73;
        const wp::int32 var_74 = 1;
        wp::int32 var_75;
        const wp::int32 var_76 = 2;
        const wp::int32 var_77 = 2;
        wp::int32 var_78;
        wp::int32 var_79;
        const wp::int32 var_80 = 1;
        wp::int32 var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 1;
        wp::int32 var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        const wp::int32 var_90 = 2;
        wp::int32 var_91;
        wp::int32 var_92;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        //---------
        // forward
        // def _compute_naive_num_shifts(                                                         <L 1>
        // tid = wp.tid()                                                                         <L 44>
        var_0 = builtin_tid1d();
        // _cell = cell[tid]                                                                      <L 46>
        var_1 = wp::address(var_cell, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // _pbc = pbc[tid]                                                                        <L 47>
        var_4 = wp::slice_t(var_0, var_0, var_5);
        var_6 = wp::view(var_pbc, var_4);
        // _cell_inv = wp.transpose(wp.inverse(_cell))                                            <L 49>
        var_7 = wp::inverse(var_2);
        var_8 = wp::transpose(var_7);
        // _d_inv_0 = wp.length(_cell_inv[0]) if _pbc[0] else type(_cell_inv[0, 0])(0.0)          <L 50>
        var_10 = wp::address(var_6, var_9);
        var_11 = wp::load(var_10);
        if (var_11) {
            var_13 = wp::extract(var_8, var_12);
            var_14 = wp::length(var_13);
        }
        var_15 = wp::load(var_10);
        var_16 = wp::load(var_10);
        if (!var_16) {
            var_19 = wp::extract(var_8, var_17, var_18);
            var_21 = wp::float16(var_20);
        }
        var_22 = wp::load(var_10);
        var_24 = wp::load(var_10);
        var_23 = wp::where(var_24, var_14, var_21);
        // _d_inv_1 = wp.length(_cell_inv[1]) if _pbc[1] else type(_cell_inv[1, 0])(0.0)          <L 51>
        var_26 = wp::address(var_6, var_25);
        var_27 = wp::load(var_26);
        if (var_27) {
            var_29 = wp::extract(var_8, var_28);
            var_30 = wp::length(var_29);
        }
        var_31 = wp::load(var_26);
        var_32 = wp::load(var_26);
        if (!var_32) {
            var_35 = wp::extract(var_8, var_33, var_34);
            var_37 = wp::float16(var_36);
        }
        var_38 = wp::load(var_26);
        var_40 = wp::load(var_26);
        var_39 = wp::where(var_40, var_30, var_37);
        // _d_inv_2 = wp.length(_cell_inv[2]) if _pbc[2] else type(_cell_inv[2, 0])(0.0)          <L 52>
        var_42 = wp::address(var_6, var_41);
        var_43 = wp::load(var_42);
        if (var_43) {
            var_45 = wp::extract(var_8, var_44);
            var_46 = wp::length(var_45);
        }
        var_47 = wp::load(var_42);
        var_48 = wp::load(var_42);
        if (!var_48) {
            var_51 = wp::extract(var_8, var_49, var_50);
            var_53 = wp::float16(var_52);
        }
        var_54 = wp::load(var_42);
        var_56 = wp::load(var_42);
        var_55 = wp::where(var_56, var_46, var_53);
        // _s = wp.vec3i(                                                                         <L 53>
        // wp.int32(wp.ceil(_d_inv_0 * type(_d_inv_0)(cutoff))),                                  <L 54>
        var_57 = wp::float16(var_cutoff);
        var_58 = wp::mul(var_23, var_57);
        var_59 = wp::ceil(var_58);
        var_60 = wp::int32(var_59);
        // wp.int32(wp.ceil(_d_inv_1 * type(_d_inv_1)(cutoff))),                                  <L 55>
        var_61 = wp::float16(var_cutoff);
        var_62 = wp::mul(var_39, var_61);
        var_63 = wp::ceil(var_62);
        var_64 = wp::int32(var_63);
        // wp.int32(wp.ceil(_d_inv_2 * type(_d_inv_2)(cutoff))),                                  <L 56>
        var_65 = wp::float16(var_cutoff);
        var_66 = wp::mul(var_55, var_65);
        var_67 = wp::ceil(var_66);
        var_68 = wp::int32(var_67);
        var_69 = wp::vec_t<3, wp::int32>(var_60, var_64, var_68);
        // k1 = 2 * _s[1] + 1                                                                     <L 58>
        var_72 = wp::extract(var_69, var_71);
        var_73 = wp::mul(var_70, var_72);
        var_75 = wp::add(var_73, var_74);
        // k2 = 2 * _s[2] + 1                                                                     <L 59>
        var_78 = wp::extract(var_69, var_77);
        var_79 = wp::mul(var_76, var_78);
        var_81 = wp::add(var_79, var_80);
        // shift_range[tid] = _s                                                                  <L 60>
        wp::array_store(var_shift_range, var_0, var_69);
        // num_shifts[tid] = _s[0] * k1 * k2 + _s[1] * k2 + _s[2] + 1                             <L 61>
        var_83 = wp::extract(var_69, var_82);
        var_84 = wp::mul(var_83, var_75);
        var_85 = wp::mul(var_84, var_81);
        var_87 = wp::extract(var_69, var_86);
        var_88 = wp::mul(var_87, var_81);
        var_89 = wp::add(var_85, var_88);
        var_91 = wp::extract(var_69, var_90);
        var_92 = wp::add(var_89, var_91);
        var_94 = wp::add(var_92, var_93);
        wp::array_store(var_num_shifts, var_0, var_94);
    }
}



extern "C" __global__ void _compute_inv_cells_kernel_ba5e7adc_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_inv_cell)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3, 3, wp::float32>* var_1;
        wp::mat_t<3, 3, wp::float32> var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        //---------
        // forward
        // def _compute_inv_cells_kernel(                                                         <L 1>
        // tid = wp.tid()                                                                         <L 18>
        var_0 = builtin_tid1d();
        // inv_cell[tid] = wp.inverse(cell[tid])                                                  <L 19>
        var_1 = wp::address(var_cell, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::inverse(var_3);
        wp::array_store(var_inv_cell, var_0, var_2);
    }
}



extern "C" __global__ void _compute_inv_cells_kernel_f7c58bdd_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_inv_cell)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3, 3, wp::float64>* var_1;
        wp::mat_t<3, 3, wp::float64> var_2;
        wp::mat_t<3, 3, wp::float64> var_3;
        //---------
        // forward
        // def _compute_inv_cells_kernel(                                                         <L 1>
        // tid = wp.tid()                                                                         <L 18>
        var_0 = builtin_tid1d();
        // inv_cell[tid] = wp.inverse(cell[tid])                                                  <L 19>
        var_1 = wp::address(var_cell, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::inverse(var_3);
        wp::array_store(var_inv_cell, var_0, var_2);
    }
}



extern "C" __global__ void _compute_inv_cells_kernel_3f1154f4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_inv_cell)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3, 3, wp::float16>* var_1;
        wp::mat_t<3, 3, wp::float16> var_2;
        wp::mat_t<3, 3, wp::float16> var_3;
        //---------
        // forward
        // def _compute_inv_cells_kernel(                                                         <L 1>
        // tid = wp.tid()                                                                         <L 18>
        var_0 = builtin_tid1d();
        // inv_cell[tid] = wp.inverse(cell[tid])                                                  <L 19>
        var_1 = wp::address(var_cell, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::inverse(var_3);
        wp::array_store(var_inv_cell, var_0, var_2);
    }
}



extern "C" __global__ void _selective_zero_num_neighbors_49ba334b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_num_neighbors,
    wp::array_t<wp::int32> var_batch_idx,
    wp::array_t<bool> var_rebuild_flags)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        bool* var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        //---------
        // forward
        // def _selective_zero_num_neighbors(                                                     <L 421>
        // tid = wp.tid()                                                                         <L 442>
        var_0 = builtin_tid1d();
        // isys = batch_idx[tid]                                                                  <L 443>
        var_1 = wp::address(var_batch_idx, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if rebuild_flags[isys]:                                                                <L 444>
        var_4 = wp::address(var_rebuild_flags, var_2);
        var_5 = wp::load(var_4);
        if (var_5) {
            // num_neighbors[tid] = 0                                                             <L 445>
            wp::array_store(var_num_neighbors, var_0, var_6);
        }
        var_7 = wp::load(var_4);
    }
}



extern "C" __global__ void _selective_zero_num_neighbors_single_8a8dd1ae_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_num_neighbors,
    wp::array_t<bool> var_rebuild_flags)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        bool* var_2;
        bool var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        //---------
        // forward
        // def _selective_zero_num_neighbors_single(                                              <L 484>
        // tid = wp.tid()                                                                         <L 502>
        var_0 = builtin_tid1d();
        // if rebuild_flags[0]:                                                                   <L 503>
        var_2 = wp::address(var_rebuild_flags, var_1);
        var_3 = wp::load(var_2);
        if (var_3) {
            // num_neighbors[tid] = 0                                                             <L 504>
            wp::array_store(var_num_neighbors, var_0, var_4);
        }
        var_5 = wp::load(var_2);
    }
}



extern "C" __global__ void _update_ref_positions_kernel_3eb745a0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<bool> var_rebuild_flag,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ref_positions)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        bool* var_2;
        bool var_3;
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<3, wp::float32> var_5;
        bool var_6;
        //---------
        // forward
        // def _update_ref_positions_kernel(                                                      <L 1>
        // i = wp.tid()                                                                           <L 22>
        var_0 = builtin_tid1d();
        // if rebuild_flag[0]:                                                                    <L 23>
        var_2 = wp::address(var_rebuild_flag, var_1);
        var_3 = wp::load(var_2);
        if (var_3) {
            // ref_positions[i] = positions[i]                                                    <L 24>
            var_4 = wp::address(var_positions, var_0);
            var_5 = wp::load(var_4);
            wp::array_store(var_ref_positions, var_0, var_5);
        }
        var_6 = wp::load(var_2);
    }
}



extern "C" __global__ void _update_ref_positions_kernel_2a1cd500_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<bool> var_rebuild_flag,
    wp::array_t<wp::vec_t<3, wp::float64>> var_ref_positions)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        bool* var_2;
        bool var_3;
        wp::vec_t<3, wp::float64>* var_4;
        wp::vec_t<3, wp::float64> var_5;
        bool var_6;
        //---------
        // forward
        // def _update_ref_positions_kernel(                                                      <L 1>
        // i = wp.tid()                                                                           <L 22>
        var_0 = builtin_tid1d();
        // if rebuild_flag[0]:                                                                    <L 23>
        var_2 = wp::address(var_rebuild_flag, var_1);
        var_3 = wp::load(var_2);
        if (var_3) {
            // ref_positions[i] = positions[i]                                                    <L 24>
            var_4 = wp::address(var_positions, var_0);
            var_5 = wp::load(var_4);
            wp::array_store(var_ref_positions, var_0, var_5);
        }
        var_6 = wp::load(var_2);
    }
}



extern "C" __global__ void _wrap_positions_batch_kernel_5c032bec_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_inv_cell,
    wp::array_t<wp::int32> var_batch_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions_wrapped,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::mat_t<3, 3, wp::float32>* var_4;
        wp::mat_t<3, 3, wp::float32> var_5;
        wp::mat_t<3, 3, wp::float32> var_6;
        wp::mat_t<3, 3, wp::float32>* var_7;
        wp::mat_t<3, 3, wp::float32> var_8;
        wp::mat_t<3, 3, wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        const wp::int32 var_14 = 0;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 2;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::int32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        //---------
        // forward
        // def _wrap_positions_batch_kernel(                                                      <L 1>
        // i = wp.tid()                                                                           <L 35>
        var_0 = builtin_tid1d();
        // isys = batch_idx[i]                                                                    <L 36>
        var_1 = wp::address(var_batch_idx, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // _cell = cell[isys]                                                                     <L 37>
        var_4 = wp::address(var_cell, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // _inv_cell = inv_cell[isys]                                                             <L 38>
        var_7 = wp::address(var_inv_cell, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // _pos = positions[i]                                                                    <L 39>
        var_10 = wp::address(var_positions, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // _frac = _pos * _inv_cell                                                               <L 40>
        var_13 = wp::mul(var_11, var_8);
        // _int = wp.vec3i(                                                                       <L 41>
        // wp.int32(wp.floor(_frac[0])),                                                          <L 42>
        var_15 = wp::extract(var_13, var_14);
        var_16 = wp::floor(var_15);
        var_17 = wp::int32(var_16);
        // wp.int32(wp.floor(_frac[1])),                                                          <L 43>
        var_19 = wp::extract(var_13, var_18);
        var_20 = wp::floor(var_19);
        var_21 = wp::int32(var_20);
        // wp.int32(wp.floor(_frac[2])),                                                          <L 44>
        var_23 = wp::extract(var_13, var_22);
        var_24 = wp::floor(var_23);
        var_25 = wp::int32(var_24);
        var_26 = wp::vec_t<3, wp::int32>(var_17, var_21, var_25);
        // positions_wrapped[i] = _pos - type(_pos)(_int) * _cell                                 <L 46>
        var_27 = wp::vec_t<3, wp::float32>(var_26);
        var_28 = wp::mul(var_27, var_5);
        var_29 = wp::sub(var_11, var_28);
        wp::array_store(var_positions_wrapped, var_0, var_29);
        // per_atom_cell_offsets[i] = _int                                                        <L 47>
        wp::array_store(var_per_atom_cell_offsets, var_0, var_26);
    }
}



extern "C" __global__ void _wrap_positions_batch_kernel_25183de6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_inv_cell,
    wp::array_t<wp::int32> var_batch_idx,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions_wrapped,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::mat_t<3, 3, wp::float64>* var_4;
        wp::mat_t<3, 3, wp::float64> var_5;
        wp::mat_t<3, 3, wp::float64> var_6;
        wp::mat_t<3, 3, wp::float64>* var_7;
        wp::mat_t<3, 3, wp::float64> var_8;
        wp::mat_t<3, 3, wp::float64> var_9;
        wp::vec_t<3, wp::float64>* var_10;
        wp::vec_t<3, wp::float64> var_11;
        wp::vec_t<3, wp::float64> var_12;
        wp::vec_t<3, wp::float64> var_13;
        const wp::int32 var_14 = 0;
        wp::float64 var_15;
        wp::float64 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::float64 var_19;
        wp::float64 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 2;
        wp::float64 var_23;
        wp::float64 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::int32> var_26;
        wp::vec_t<3, wp::float64> var_27;
        wp::vec_t<3, wp::float64> var_28;
        wp::vec_t<3, wp::float64> var_29;
        //---------
        // forward
        // def _wrap_positions_batch_kernel(                                                      <L 1>
        // i = wp.tid()                                                                           <L 35>
        var_0 = builtin_tid1d();
        // isys = batch_idx[i]                                                                    <L 36>
        var_1 = wp::address(var_batch_idx, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // _cell = cell[isys]                                                                     <L 37>
        var_4 = wp::address(var_cell, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // _inv_cell = inv_cell[isys]                                                             <L 38>
        var_7 = wp::address(var_inv_cell, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // _pos = positions[i]                                                                    <L 39>
        var_10 = wp::address(var_positions, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // _frac = _pos * _inv_cell                                                               <L 40>
        var_13 = wp::mul(var_11, var_8);
        // _int = wp.vec3i(                                                                       <L 41>
        // wp.int32(wp.floor(_frac[0])),                                                          <L 42>
        var_15 = wp::extract(var_13, var_14);
        var_16 = wp::floor(var_15);
        var_17 = wp::int32(var_16);
        // wp.int32(wp.floor(_frac[1])),                                                          <L 43>
        var_19 = wp::extract(var_13, var_18);
        var_20 = wp::floor(var_19);
        var_21 = wp::int32(var_20);
        // wp.int32(wp.floor(_frac[2])),                                                          <L 44>
        var_23 = wp::extract(var_13, var_22);
        var_24 = wp::floor(var_23);
        var_25 = wp::int32(var_24);
        var_26 = wp::vec_t<3, wp::int32>(var_17, var_21, var_25);
        // positions_wrapped[i] = _pos - type(_pos)(_int) * _cell                                 <L 46>
        var_27 = wp::vec_t<3, wp::float64>(var_26);
        var_28 = wp::mul(var_27, var_5);
        var_29 = wp::sub(var_11, var_28);
        wp::array_store(var_positions_wrapped, var_0, var_29);
        // per_atom_cell_offsets[i] = _int                                                        <L 47>
        wp::array_store(var_per_atom_cell_offsets, var_0, var_26);
    }
}



extern "C" __global__ void _wrap_positions_batch_kernel_c82cd93b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_inv_cell,
    wp::array_t<wp::int32> var_batch_idx,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions_wrapped,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::mat_t<3, 3, wp::float16>* var_4;
        wp::mat_t<3, 3, wp::float16> var_5;
        wp::mat_t<3, 3, wp::float16> var_6;
        wp::mat_t<3, 3, wp::float16>* var_7;
        wp::mat_t<3, 3, wp::float16> var_8;
        wp::mat_t<3, 3, wp::float16> var_9;
        wp::vec_t<3, wp::float16>* var_10;
        wp::vec_t<3, wp::float16> var_11;
        wp::vec_t<3, wp::float16> var_12;
        wp::vec_t<3, wp::float16> var_13;
        const wp::int32 var_14 = 0;
        wp::float16 var_15;
        wp::float16 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::float16 var_19;
        wp::float16 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 2;
        wp::float16 var_23;
        wp::float16 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::int32> var_26;
        wp::vec_t<3, wp::float16> var_27;
        wp::vec_t<3, wp::float16> var_28;
        wp::vec_t<3, wp::float16> var_29;
        //---------
        // forward
        // def _wrap_positions_batch_kernel(                                                      <L 1>
        // i = wp.tid()                                                                           <L 35>
        var_0 = builtin_tid1d();
        // isys = batch_idx[i]                                                                    <L 36>
        var_1 = wp::address(var_batch_idx, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // _cell = cell[isys]                                                                     <L 37>
        var_4 = wp::address(var_cell, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // _inv_cell = inv_cell[isys]                                                             <L 38>
        var_7 = wp::address(var_inv_cell, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // _pos = positions[i]                                                                    <L 39>
        var_10 = wp::address(var_positions, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // _frac = _pos * _inv_cell                                                               <L 40>
        var_13 = wp::mul(var_11, var_8);
        // _int = wp.vec3i(                                                                       <L 41>
        // wp.int32(wp.floor(_frac[0])),                                                          <L 42>
        var_15 = wp::extract(var_13, var_14);
        var_16 = wp::floor(var_15);
        var_17 = wp::int32(var_16);
        // wp.int32(wp.floor(_frac[1])),                                                          <L 43>
        var_19 = wp::extract(var_13, var_18);
        var_20 = wp::floor(var_19);
        var_21 = wp::int32(var_20);
        // wp.int32(wp.floor(_frac[2])),                                                          <L 44>
        var_23 = wp::extract(var_13, var_22);
        var_24 = wp::floor(var_23);
        var_25 = wp::int32(var_24);
        var_26 = wp::vec_t<3, wp::int32>(var_17, var_21, var_25);
        // positions_wrapped[i] = _pos - type(_pos)(_int) * _cell                                 <L 46>
        var_27 = wp::vec_t<3, wp::float16>(var_26);
        var_28 = wp::mul(var_27, var_5);
        var_29 = wp::sub(var_11, var_28);
        wp::array_store(var_positions_wrapped, var_0, var_29);
        // per_atom_cell_offsets[i] = _int                                                        <L 47>
        wp::array_store(var_per_atom_cell_offsets, var_0, var_26);
    }
}



extern "C" __global__ void _update_ref_positions_batch_kernel_b96eac83_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<bool> var_rebuild_flags,
    wp::array_t<wp::int32> var_batch_idx,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ref_positions)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        bool* var_2;
        wp::int32 var_3;
        bool var_4;
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        bool var_7;
        //---------
        // forward
        // def _update_ref_positions_batch_kernel(                                                <L 1>
        // i = wp.tid()                                                                           <L 25>
        var_0 = builtin_tid1d();
        // if rebuild_flags[batch_idx[i]]:                                                        <L 26>
        var_1 = wp::address(var_batch_idx, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::address(var_rebuild_flags, var_3);
        var_4 = wp::load(var_2);
        if (var_4) {
            // ref_positions[i] = positions[i]                                                    <L 27>
            var_5 = wp::address(var_positions, var_0);
            var_6 = wp::load(var_5);
            wp::array_store(var_ref_positions, var_0, var_6);
        }
        var_7 = wp::load(var_2);
    }
}



extern "C" __global__ void _update_ref_positions_batch_kernel_4cf1db34_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<bool> var_rebuild_flags,
    wp::array_t<wp::int32> var_batch_idx,
    wp::array_t<wp::vec_t<3, wp::float64>> var_ref_positions)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        bool* var_2;
        wp::int32 var_3;
        bool var_4;
        wp::vec_t<3, wp::float64>* var_5;
        wp::vec_t<3, wp::float64> var_6;
        bool var_7;
        //---------
        // forward
        // def _update_ref_positions_batch_kernel(                                                <L 1>
        // i = wp.tid()                                                                           <L 25>
        var_0 = builtin_tid1d();
        // if rebuild_flags[batch_idx[i]]:                                                        <L 26>
        var_1 = wp::address(var_batch_idx, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::address(var_rebuild_flags, var_3);
        var_4 = wp::load(var_2);
        if (var_4) {
            // ref_positions[i] = positions[i]                                                    <L 27>
            var_5 = wp::address(var_positions, var_0);
            var_6 = wp::load(var_5);
            wp::array_store(var_ref_positions, var_0, var_6);
        }
        var_7 = wp::load(var_2);
    }
}

