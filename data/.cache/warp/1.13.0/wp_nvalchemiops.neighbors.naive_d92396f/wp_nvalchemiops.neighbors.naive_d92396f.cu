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


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/neighbor_utils.py:204
static CUDA_CALLABLE void _update_neighbor_matrix_0(
    wp::int32 var_i,
    wp::int32 var_j,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    wp::int32 var_max_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    bool var_2;
    bool var_3;
    bool var_4;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    bool var_8;
    wp::int32 var_9;
    //---------
    // forward
    // def _update_neighbor_matrix(                                                           <L 205>
    // pos = wp.atomic_add(num_neighbors, i, 1)                                               <L 231>
    var_1 = wp::atomic_add(var_num_neighbors, var_i, var_0);
    // if pos < max_neighbors:                                                                <L 232>
    var_2 = (var_1 < var_max_neighbors);
    if (var_2) {
        // neighbor_matrix[i, pos] = j                                                        <L 233>
        wp::array_store(var_neighbor_matrix, var_i, var_1, var_j);
    }
    // if not half_fill and i < j:                                                            <L 234>
    var_4 = wp::unot(var_half_fill);
    var_3 = var_4;
    if (var_3) {
        var_5 = (var_i < var_j);
        var_3 = var_3 && var_5;
    }
    if (var_3) {
        // pos = wp.atomic_add(num_neighbors, j, 1)                                           <L 235>
        var_7 = wp::atomic_add(var_num_neighbors, var_j, var_6);
        // if pos < max_neighbors:                                                            <L 236>
        var_8 = (var_7 < var_max_neighbors);
        if (var_8) {
            // neighbor_matrix[j, pos] = i                                                    <L 237>
            wp::array_store(var_neighbor_matrix, var_j, var_7, var_i);
        }
    }
    var_9 = wp::where(var_3, var_7, var_1);
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_body_0(
    wp::int32 var_tid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::shape_t* var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    wp::shape_t var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    wp::range_t var_13;
    wp::int32 var_14;
    wp::vec_t<3, wp::float32>* var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::float32 var_18;
    bool var_19;
    //---------
    // forward
    // def _naive_neighbor_body(                                                              <L 1>
    // j_end = positions.shape[0]                                                             <L 9>
    var_0 = &(var_positions.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    // positions_i = positions[tid]                                                           <L 10>
    var_4 = wp::address(var_positions, var_tid);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // max_neighbors = neighbor_matrix.shape[1]                                               <L 11>
    var_7 = &(var_neighbor_matrix.shape);
    var_10 = wp::load(var_7);
    var_9 = wp::extract(var_10, var_8);
    // for j in range(tid + 1, j_end):                                                        <L 12>
    var_12 = wp::add(var_tid, var_11);
    var_13 = wp::range(var_12, var_2);
    start_for_0:;
        if (iter_cmp(var_13) == 0) goto end_for_0;
        var_14 = wp::iter_next(var_13);
        // diff = positions_i - positions[j]                                                  <L 13>
        var_15 = wp::address(var_positions, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::sub(var_5, var_17);
        // dist_sq = wp.length_sq(diff)                                                       <L 14>
        var_18 = wp::length_sq(var_16);
        // if dist_sq < cutoff_sq:                                                            <L 15>
        var_19 = (var_18 < var_cutoff_sq);
        if (var_19) {
            // _update_neighbor_matrix(                                                       <L 16>
            // tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill               <L 17>
            _update_neighbor_matrix_0(var_tid, var_14, var_neighbor_matrix, var_num_neighbors, var_9, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_body_0(
    wp::int32 var_tid,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    wp::vec_t<3, wp::float64>* var_4;
    wp::vec_t<3, wp::float64> var_5;
    wp::vec_t<3, wp::float64> var_6;
    wp::shape_t* var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    wp::shape_t var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    wp::range_t var_13;
    wp::int32 var_14;
    wp::vec_t<3, wp::float64>* var_15;
    wp::vec_t<3, wp::float64> var_16;
    wp::vec_t<3, wp::float64> var_17;
    wp::float64 var_18;
    bool var_19;
    //---------
    // forward
    // def _naive_neighbor_body(                                                              <L 1>
    // j_end = positions.shape[0]                                                             <L 9>
    var_0 = &(var_positions.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    // positions_i = positions[tid]                                                           <L 10>
    var_4 = wp::address(var_positions, var_tid);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // max_neighbors = neighbor_matrix.shape[1]                                               <L 11>
    var_7 = &(var_neighbor_matrix.shape);
    var_10 = wp::load(var_7);
    var_9 = wp::extract(var_10, var_8);
    // for j in range(tid + 1, j_end):                                                        <L 12>
    var_12 = wp::add(var_tid, var_11);
    var_13 = wp::range(var_12, var_2);
    start_for_0:;
        if (iter_cmp(var_13) == 0) goto end_for_0;
        var_14 = wp::iter_next(var_13);
        // diff = positions_i - positions[j]                                                  <L 13>
        var_15 = wp::address(var_positions, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::sub(var_5, var_17);
        // dist_sq = wp.length_sq(diff)                                                       <L 14>
        var_18 = wp::length_sq(var_16);
        // if dist_sq < cutoff_sq:                                                            <L 15>
        var_19 = (var_18 < var_cutoff_sq);
        if (var_19) {
            // _update_neighbor_matrix(                                                       <L 16>
            // tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill               <L 17>
            _update_neighbor_matrix_0(var_tid, var_14, var_neighbor_matrix, var_num_neighbors, var_9, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_body_0(
    wp::int32 var_tid,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    wp::vec_t<3, wp::float16>* var_4;
    wp::vec_t<3, wp::float16> var_5;
    wp::vec_t<3, wp::float16> var_6;
    wp::shape_t* var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    wp::shape_t var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    wp::range_t var_13;
    wp::int32 var_14;
    wp::vec_t<3, wp::float16>* var_15;
    wp::vec_t<3, wp::float16> var_16;
    wp::vec_t<3, wp::float16> var_17;
    wp::float16 var_18;
    bool var_19;
    //---------
    // forward
    // def _naive_neighbor_body(                                                              <L 1>
    // j_end = positions.shape[0]                                                             <L 9>
    var_0 = &(var_positions.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    // positions_i = positions[tid]                                                           <L 10>
    var_4 = wp::address(var_positions, var_tid);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // max_neighbors = neighbor_matrix.shape[1]                                               <L 11>
    var_7 = &(var_neighbor_matrix.shape);
    var_10 = wp::load(var_7);
    var_9 = wp::extract(var_10, var_8);
    // for j in range(tid + 1, j_end):                                                        <L 12>
    var_12 = wp::add(var_tid, var_11);
    var_13 = wp::range(var_12, var_2);
    start_for_0:;
        if (iter_cmp(var_13) == 0) goto end_for_0;
        var_14 = wp::iter_next(var_13);
        // diff = positions_i - positions[j]                                                  <L 13>
        var_15 = wp::address(var_positions, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::sub(var_5, var_17);
        // dist_sq = wp.length_sq(diff)                                                       <L 14>
        var_18 = wp::length_sq(var_16);
        // if dist_sq < cutoff_sq:                                                            <L 15>
        var_19 = (var_18 < var_cutoff_sq);
        if (var_19) {
            // _update_neighbor_matrix(                                                       <L 16>
            // tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill               <L 17>
            _update_neighbor_matrix_0(var_tid, var_14, var_neighbor_matrix, var_num_neighbors, var_9, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/neighbor_utils.py:159
static CUDA_CALLABLE wp::vec_t<3, wp::int32> _decode_shift_index_0(
    wp::int32 var_local_idx,
    wp::vec_t<3, wp::int32> var_shift_range)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    const wp::int32 var_1 = 2;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    wp::int32 var_14;
    const wp::int32 var_15 = 2;
    wp::int32 var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 1;
    wp::int32 var_19;
    const wp::int32 var_20 = 0;
    wp::int32 var_21;
    const wp::int32 var_22 = 0;
    wp::int32 var_23;
    const wp::int32 var_24 = 0;
    wp::int32 var_25;
    bool var_26;
    const wp::int32 var_27 = 2;
    wp::int32 var_28;
    bool var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    const wp::int32 var_32 = 2;
    wp::int32 var_33;
    const wp::int32 var_34 = 1;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::int32 var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 2;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    const wp::int32 var_51 = 1;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 1;
    wp::int32 var_57;
    wp::int32 var_58;
    wp::int32 var_59;
    const wp::int32 var_60 = 2;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    wp::int32 var_65;
    wp::int32 var_66;
    wp::vec_t<3, wp::int32> var_67;
    //---------
    // forward
    // def _decode_shift_index(local_idx: int, shift_range: wp.vec3i) -> wp.vec3i:            <L 160>
    // k2_size = 2 * shift_range[2] + 1                                                       <L 179>
    var_2 = wp::extract(var_shift_range, var_1);
    var_3 = wp::mul(var_0, var_2);
    var_5 = wp::add(var_3, var_4);
    // k1_size = 2 * shift_range[1] + 1                                                       <L 180>
    var_8 = wp::extract(var_shift_range, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_11 = wp::add(var_9, var_10);
    // group0_size = shift_range[1] * k2_size + shift_range[2] + 1                            <L 181>
    var_13 = wp::extract(var_shift_range, var_12);
    var_14 = wp::mul(var_13, var_5);
    var_16 = wp::extract(var_shift_range, var_15);
    var_17 = wp::add(var_14, var_16);
    var_19 = wp::add(var_17, var_18);
    // k0 = wp.int32(0)                                                                       <L 183>
    var_21 = wp::int32(var_20);
    // k1 = wp.int32(0)                                                                       <L 184>
    var_23 = wp::int32(var_22);
    // k2 = wp.int32(0)                                                                       <L 185>
    var_25 = wp::int32(var_24);
    // if local_idx < group0_size:                                                            <L 187>
    var_26 = (var_local_idx < var_19);
    if (var_26) {
        // if local_idx <= shift_range[2]:                                                    <L 188>
        var_28 = wp::extract(var_shift_range, var_27);
        var_29 = (var_local_idx <= var_28);
        if (var_29) {
            // k2 = local_idx                                                                 <L 189>
            var_30 = wp::copy(var_local_idx);
        }
        var_31 = wp::where(var_29, var_30, var_25);
        if (!var_29) {
            // rem = local_idx - (shift_range[2] + 1)                                         <L 191>
            var_33 = wp::extract(var_shift_range, var_32);
            var_35 = wp::add(var_33, var_34);
            var_36 = wp::sub(var_local_idx, var_35);
            // k1 = rem / k2_size + 1                                                         <L 192>
            var_37 = wp::div(var_36, var_5);
            var_39 = wp::add(var_37, var_38);
            // k2 = rem % k2_size - shift_range[2]                                            <L 193>
            var_40 = wp::mod(var_36, var_5);
            var_42 = wp::extract(var_shift_range, var_41);
            var_43 = wp::sub(var_40, var_42);
        }
        var_44 = wp::where(var_29, var_23, var_39);
        var_45 = wp::where(var_29, var_31, var_43);
    }
    var_46 = wp::where(var_26, var_44, var_23);
    var_47 = wp::where(var_26, var_45, var_25);
    if (!var_26) {
        // rem = local_idx - group0_size                                                      <L 195>
        var_48 = wp::sub(var_local_idx, var_19);
        // k0 = rem / (k1_size * k2_size) + 1                                                 <L 196>
        var_49 = wp::mul(var_11, var_5);
        var_50 = wp::div(var_48, var_49);
        var_52 = wp::add(var_50, var_51);
        // rem2 = rem % (k1_size * k2_size)                                                   <L 197>
        var_53 = wp::mul(var_11, var_5);
        var_54 = wp::mod(var_48, var_53);
        // k1 = rem2 / k2_size - shift_range[1]                                               <L 198>
        var_55 = wp::div(var_54, var_5);
        var_57 = wp::extract(var_shift_range, var_56);
        var_58 = wp::sub(var_55, var_57);
        // k2 = rem2 % k2_size - shift_range[2]                                               <L 199>
        var_59 = wp::mod(var_54, var_5);
        var_61 = wp::extract(var_shift_range, var_60);
        var_62 = wp::sub(var_59, var_61);
    }
    var_63 = wp::where(var_26, var_21, var_52);
    var_64 = wp::where(var_26, var_46, var_58);
    var_65 = wp::where(var_26, var_47, var_62);
    var_66 = wp::where(var_26, var_36, var_48);
    // return wp.vec3i(k0, k1, k2)                                                            <L 201>
    var_67 = wp::vec_t<3, wp::int32>(var_63, var_64, var_65);
    return var_67;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/neighbor_utils.py:240
static CUDA_CALLABLE void _update_neighbor_matrix_pbc_0(
    wp::int32 var_i,
    wp::int32 var_j,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    wp::vec_t<3, wp::int32> var_unit_shift,
    wp::int32 var_max_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    bool var_2;
    bool var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    bool var_6;
    wp::vec_t<3, wp::int32> var_7;
    wp::int32 var_8;
    //---------
    // forward
    // def _update_neighbor_matrix_pbc(                                                       <L 241>
    // pos = wp.atomic_add(num_neighbors, i, 1)                                               <L 273>
    var_1 = wp::atomic_add(var_num_neighbors, var_i, var_0);
    // if pos < max_neighbors:                                                                <L 274>
    var_2 = (var_1 < var_max_neighbors);
    if (var_2) {
        // neighbor_matrix[i, pos] = j                                                        <L 275>
        wp::array_store(var_neighbor_matrix, var_i, var_1, var_j);
        // neighbor_matrix_shifts[i, pos] = unit_shift                                        <L 276>
        wp::array_store(var_neighbor_matrix_shifts, var_i, var_1, var_unit_shift);
    }
    // if not half_fill:                                                                      <L 277>
    var_3 = wp::unot(var_half_fill);
    if (var_3) {
        // pos = wp.atomic_add(num_neighbors, j, 1)                                           <L 278>
        var_5 = wp::atomic_add(var_num_neighbors, var_j, var_4);
        // if pos < max_neighbors:                                                            <L 279>
        var_6 = (var_5 < var_max_neighbors);
        if (var_6) {
            // neighbor_matrix[j, pos] = i                                                    <L 280>
            wp::array_store(var_neighbor_matrix, var_j, var_5, var_i);
            // neighbor_matrix_shifts[j, pos] = -unit_shift                                   <L 281>
            var_7 = wp::neg(var_unit_shift);
            wp::array_store(var_neighbor_matrix_shifts, var_j, var_5, var_7);
        }
    }
    var_8 = wp::where(var_3, var_5, var_1);
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_pbc_body_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::shape_t* var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::shape_t var_9;
    const wp::int32 var_10 = 0;
    wp::mat_t<3, 3, wp::float32>* var_11;
    wp::mat_t<3, 3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::vec_t<3, wp::float32>* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::int32>* var_17;
    wp::vec_t<3, wp::int32> var_18;
    wp::vec_t<3, wp::int32> var_19;
    const wp::int32 var_20 = 0;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    bool var_25;
    const wp::int32 var_26 = 0;
    wp::int32 var_27;
    const wp::int32 var_28 = 0;
    bool var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    const wp::int32 var_32 = 0;
    bool var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    const wp::int32 var_36 = 0;
    bool var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::range_t var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::float32 var_46;
    bool var_47;
    wp::vec_t<3, wp::int32>* var_48;
    wp::vec_t<3, wp::int32> var_49;
    wp::vec_t<3, wp::int32> var_50;
    const wp::int32 var_51 = 0;
    wp::int32 var_52;
    const wp::int32 var_53 = 0;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 0;
    wp::int32 var_57;
    wp::int32 var_58;
    const wp::int32 var_59 = 1;
    wp::int32 var_60;
    const wp::int32 var_61 = 1;
    wp::int32 var_62;
    wp::int32 var_63;
    const wp::int32 var_64 = 1;
    wp::int32 var_65;
    wp::int32 var_66;
    const wp::int32 var_67 = 2;
    wp::int32 var_68;
    const wp::int32 var_69 = 2;
    wp::int32 var_70;
    wp::int32 var_71;
    const wp::int32 var_72 = 2;
    wp::int32 var_73;
    wp::int32 var_74;
    wp::vec_t<3, wp::int32> var_75;
    //---------
    // forward
    // def _naive_neighbor_pbc_body(                                                          <L 1>
    // jatom_start = wp.int32(0)                                                              <L 13>
    var_1 = wp::int32(var_0);
    // jatom_end = positions.shape[0]                                                         <L 14>
    var_2 = &(var_positions.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    // maxnb = neighbor_matrix.shape[1]                                                       <L 15>
    var_6 = &(var_neighbor_matrix.shape);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    // _cell = cell[0]                                                                        <L 16>
    var_11 = wp::address(var_cell, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // _pos_i = positions[iatom]                                                              <L 17>
    var_14 = wp::address(var_positions, var_iatom);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // _int_i = per_atom_cell_offsets[iatom]                                                  <L 18>
    var_17 = wp::address(var_per_atom_cell_offsets, var_iatom);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // positions_shifted = type(_cell[0])(shift) * _cell + _pos_i                             <L 19>
    var_21 = wp::extract(var_12, var_20);
    var_22 = wp::vec_t<3, wp::float32>(var_shift);
    var_23 = wp::mul(var_22, var_12);
    var_24 = wp::add(var_23, var_15);
    // _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0                        <L 20>
    var_27 = wp::extract(var_shift, var_26);
    var_29 = (var_27 == var_28);
    var_25 = var_29;
    if (var_25) {
        var_31 = wp::extract(var_shift, var_30);
        var_33 = (var_31 == var_32);
        var_25 = var_25 && var_33;
    }
    if (var_25) {
        var_35 = wp::extract(var_shift, var_34);
        var_37 = (var_35 == var_36);
        var_25 = var_25 && var_37;
    }
    // if _zero_shift:                                                                        <L 21>
    if (var_25) {
        // jatom_end = iatom                                                                  <L 22>
        var_38 = wp::copy(var_iatom);
    }
    var_39 = wp::where(var_25, var_38, var_4);
    // for jatom in range(jatom_start, jatom_end):                                            <L 23>
    var_40 = wp::range(var_1, var_39);
    start_for_0:;
        if (iter_cmp(var_40) == 0) goto end_for_0;
        var_41 = wp::iter_next(var_40);
        // _pos_j = positions[jatom]                                                          <L 24>
        var_42 = wp::address(var_positions, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // diff = positions_shifted - _pos_j                                                  <L 25>
        var_45 = wp::sub(var_24, var_43);
        // dist_sq = wp.length_sq(diff)                                                       <L 26>
        var_46 = wp::length_sq(var_45);
        // if dist_sq < cutoff_sq:                                                            <L 27>
        var_47 = (var_46 < var_cutoff_sq);
        if (var_47) {
            // _int_j = per_atom_cell_offsets[jatom]                                          <L 28>
            var_48 = wp::address(var_per_atom_cell_offsets, var_41);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // _corrected_shift = wp.vec3i(                                                   <L 29>
            // shift[0] - _int_i[0] + _int_j[0],                                              <L 30>
            var_52 = wp::extract(var_shift, var_51);
            var_54 = wp::extract(var_18, var_53);
            var_55 = wp::sub(var_52, var_54);
            var_57 = wp::extract(var_49, var_56);
            var_58 = wp::add(var_55, var_57);
            // shift[1] - _int_i[1] + _int_j[1],                                              <L 31>
            var_60 = wp::extract(var_shift, var_59);
            var_62 = wp::extract(var_18, var_61);
            var_63 = wp::sub(var_60, var_62);
            var_65 = wp::extract(var_49, var_64);
            var_66 = wp::add(var_63, var_65);
            // shift[2] - _int_i[2] + _int_j[2],                                              <L 32>
            var_68 = wp::extract(var_shift, var_67);
            var_70 = wp::extract(var_18, var_69);
            var_71 = wp::sub(var_68, var_70);
            var_73 = wp::extract(var_49, var_72);
            var_74 = wp::add(var_71, var_73);
            var_75 = wp::vec_t<3, wp::int32>(var_58, var_66, var_74);
            // _update_neighbor_matrix_pbc(                                                   <L 34>
            // jatom,                                                                         <L 35>
            // iatom,                                                                         <L 36>
            // neighbor_matrix,                                                               <L 37>
            // neighbor_matrix_shifts,                                                        <L 38>
            // num_neighbors,                                                                 <L 39>
            // _corrected_shift,                                                              <L 40>
            // maxnb,                                                                         <L 41>
            // half_fill,                                                                     <L 42>
            _update_neighbor_matrix_pbc_0(var_41, var_iatom, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_75, var_8, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_pbc_body_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::shape_t* var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::shape_t var_9;
    const wp::int32 var_10 = 0;
    wp::mat_t<3, 3, wp::float64>* var_11;
    wp::mat_t<3, 3, wp::float64> var_12;
    wp::mat_t<3, 3, wp::float64> var_13;
    wp::vec_t<3, wp::float64>* var_14;
    wp::vec_t<3, wp::float64> var_15;
    wp::vec_t<3, wp::float64> var_16;
    wp::vec_t<3, wp::int32>* var_17;
    wp::vec_t<3, wp::int32> var_18;
    wp::vec_t<3, wp::int32> var_19;
    const wp::int32 var_20 = 0;
    wp::vec_t<3, wp::float64> var_21;
    wp::vec_t<3, wp::float64> var_22;
    wp::vec_t<3, wp::float64> var_23;
    wp::vec_t<3, wp::float64> var_24;
    bool var_25;
    const wp::int32 var_26 = 0;
    wp::int32 var_27;
    const wp::int32 var_28 = 0;
    bool var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    const wp::int32 var_32 = 0;
    bool var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    const wp::int32 var_36 = 0;
    bool var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::range_t var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float64>* var_42;
    wp::vec_t<3, wp::float64> var_43;
    wp::vec_t<3, wp::float64> var_44;
    wp::vec_t<3, wp::float64> var_45;
    wp::float64 var_46;
    bool var_47;
    wp::vec_t<3, wp::int32>* var_48;
    wp::vec_t<3, wp::int32> var_49;
    wp::vec_t<3, wp::int32> var_50;
    const wp::int32 var_51 = 0;
    wp::int32 var_52;
    const wp::int32 var_53 = 0;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 0;
    wp::int32 var_57;
    wp::int32 var_58;
    const wp::int32 var_59 = 1;
    wp::int32 var_60;
    const wp::int32 var_61 = 1;
    wp::int32 var_62;
    wp::int32 var_63;
    const wp::int32 var_64 = 1;
    wp::int32 var_65;
    wp::int32 var_66;
    const wp::int32 var_67 = 2;
    wp::int32 var_68;
    const wp::int32 var_69 = 2;
    wp::int32 var_70;
    wp::int32 var_71;
    const wp::int32 var_72 = 2;
    wp::int32 var_73;
    wp::int32 var_74;
    wp::vec_t<3, wp::int32> var_75;
    //---------
    // forward
    // def _naive_neighbor_pbc_body(                                                          <L 1>
    // jatom_start = wp.int32(0)                                                              <L 13>
    var_1 = wp::int32(var_0);
    // jatom_end = positions.shape[0]                                                         <L 14>
    var_2 = &(var_positions.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    // maxnb = neighbor_matrix.shape[1]                                                       <L 15>
    var_6 = &(var_neighbor_matrix.shape);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    // _cell = cell[0]                                                                        <L 16>
    var_11 = wp::address(var_cell, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // _pos_i = positions[iatom]                                                              <L 17>
    var_14 = wp::address(var_positions, var_iatom);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // _int_i = per_atom_cell_offsets[iatom]                                                  <L 18>
    var_17 = wp::address(var_per_atom_cell_offsets, var_iatom);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // positions_shifted = type(_cell[0])(shift) * _cell + _pos_i                             <L 19>
    var_21 = wp::extract(var_12, var_20);
    var_22 = wp::vec_t<3, wp::float64>(var_shift);
    var_23 = wp::mul(var_22, var_12);
    var_24 = wp::add(var_23, var_15);
    // _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0                        <L 20>
    var_27 = wp::extract(var_shift, var_26);
    var_29 = (var_27 == var_28);
    var_25 = var_29;
    if (var_25) {
        var_31 = wp::extract(var_shift, var_30);
        var_33 = (var_31 == var_32);
        var_25 = var_25 && var_33;
    }
    if (var_25) {
        var_35 = wp::extract(var_shift, var_34);
        var_37 = (var_35 == var_36);
        var_25 = var_25 && var_37;
    }
    // if _zero_shift:                                                                        <L 21>
    if (var_25) {
        // jatom_end = iatom                                                                  <L 22>
        var_38 = wp::copy(var_iatom);
    }
    var_39 = wp::where(var_25, var_38, var_4);
    // for jatom in range(jatom_start, jatom_end):                                            <L 23>
    var_40 = wp::range(var_1, var_39);
    start_for_0:;
        if (iter_cmp(var_40) == 0) goto end_for_0;
        var_41 = wp::iter_next(var_40);
        // _pos_j = positions[jatom]                                                          <L 24>
        var_42 = wp::address(var_positions, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // diff = positions_shifted - _pos_j                                                  <L 25>
        var_45 = wp::sub(var_24, var_43);
        // dist_sq = wp.length_sq(diff)                                                       <L 26>
        var_46 = wp::length_sq(var_45);
        // if dist_sq < cutoff_sq:                                                            <L 27>
        var_47 = (var_46 < var_cutoff_sq);
        if (var_47) {
            // _int_j = per_atom_cell_offsets[jatom]                                          <L 28>
            var_48 = wp::address(var_per_atom_cell_offsets, var_41);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // _corrected_shift = wp.vec3i(                                                   <L 29>
            // shift[0] - _int_i[0] + _int_j[0],                                              <L 30>
            var_52 = wp::extract(var_shift, var_51);
            var_54 = wp::extract(var_18, var_53);
            var_55 = wp::sub(var_52, var_54);
            var_57 = wp::extract(var_49, var_56);
            var_58 = wp::add(var_55, var_57);
            // shift[1] - _int_i[1] + _int_j[1],                                              <L 31>
            var_60 = wp::extract(var_shift, var_59);
            var_62 = wp::extract(var_18, var_61);
            var_63 = wp::sub(var_60, var_62);
            var_65 = wp::extract(var_49, var_64);
            var_66 = wp::add(var_63, var_65);
            // shift[2] - _int_i[2] + _int_j[2],                                              <L 32>
            var_68 = wp::extract(var_shift, var_67);
            var_70 = wp::extract(var_18, var_69);
            var_71 = wp::sub(var_68, var_70);
            var_73 = wp::extract(var_49, var_72);
            var_74 = wp::add(var_71, var_73);
            var_75 = wp::vec_t<3, wp::int32>(var_58, var_66, var_74);
            // _update_neighbor_matrix_pbc(                                                   <L 34>
            // jatom,                                                                         <L 35>
            // iatom,                                                                         <L 36>
            // neighbor_matrix,                                                               <L 37>
            // neighbor_matrix_shifts,                                                        <L 38>
            // num_neighbors,                                                                 <L 39>
            // _corrected_shift,                                                              <L 40>
            // maxnb,                                                                         <L 41>
            // half_fill,                                                                     <L 42>
            _update_neighbor_matrix_pbc_0(var_41, var_iatom, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_75, var_8, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_pbc_body_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::shape_t* var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::shape_t var_9;
    const wp::int32 var_10 = 0;
    wp::mat_t<3, 3, wp::float16>* var_11;
    wp::mat_t<3, 3, wp::float16> var_12;
    wp::mat_t<3, 3, wp::float16> var_13;
    wp::vec_t<3, wp::float16>* var_14;
    wp::vec_t<3, wp::float16> var_15;
    wp::vec_t<3, wp::float16> var_16;
    wp::vec_t<3, wp::int32>* var_17;
    wp::vec_t<3, wp::int32> var_18;
    wp::vec_t<3, wp::int32> var_19;
    const wp::int32 var_20 = 0;
    wp::vec_t<3, wp::float16> var_21;
    wp::vec_t<3, wp::float16> var_22;
    wp::vec_t<3, wp::float16> var_23;
    wp::vec_t<3, wp::float16> var_24;
    bool var_25;
    const wp::int32 var_26 = 0;
    wp::int32 var_27;
    const wp::int32 var_28 = 0;
    bool var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    const wp::int32 var_32 = 0;
    bool var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    const wp::int32 var_36 = 0;
    bool var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::range_t var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float16>* var_42;
    wp::vec_t<3, wp::float16> var_43;
    wp::vec_t<3, wp::float16> var_44;
    wp::vec_t<3, wp::float16> var_45;
    wp::float16 var_46;
    bool var_47;
    wp::vec_t<3, wp::int32>* var_48;
    wp::vec_t<3, wp::int32> var_49;
    wp::vec_t<3, wp::int32> var_50;
    const wp::int32 var_51 = 0;
    wp::int32 var_52;
    const wp::int32 var_53 = 0;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 0;
    wp::int32 var_57;
    wp::int32 var_58;
    const wp::int32 var_59 = 1;
    wp::int32 var_60;
    const wp::int32 var_61 = 1;
    wp::int32 var_62;
    wp::int32 var_63;
    const wp::int32 var_64 = 1;
    wp::int32 var_65;
    wp::int32 var_66;
    const wp::int32 var_67 = 2;
    wp::int32 var_68;
    const wp::int32 var_69 = 2;
    wp::int32 var_70;
    wp::int32 var_71;
    const wp::int32 var_72 = 2;
    wp::int32 var_73;
    wp::int32 var_74;
    wp::vec_t<3, wp::int32> var_75;
    //---------
    // forward
    // def _naive_neighbor_pbc_body(                                                          <L 1>
    // jatom_start = wp.int32(0)                                                              <L 13>
    var_1 = wp::int32(var_0);
    // jatom_end = positions.shape[0]                                                         <L 14>
    var_2 = &(var_positions.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    // maxnb = neighbor_matrix.shape[1]                                                       <L 15>
    var_6 = &(var_neighbor_matrix.shape);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    // _cell = cell[0]                                                                        <L 16>
    var_11 = wp::address(var_cell, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // _pos_i = positions[iatom]                                                              <L 17>
    var_14 = wp::address(var_positions, var_iatom);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // _int_i = per_atom_cell_offsets[iatom]                                                  <L 18>
    var_17 = wp::address(var_per_atom_cell_offsets, var_iatom);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // positions_shifted = type(_cell[0])(shift) * _cell + _pos_i                             <L 19>
    var_21 = wp::extract(var_12, var_20);
    var_22 = wp::vec_t<3, wp::float16>(var_shift);
    var_23 = wp::mul(var_22, var_12);
    var_24 = wp::add(var_23, var_15);
    // _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0                        <L 20>
    var_27 = wp::extract(var_shift, var_26);
    var_29 = (var_27 == var_28);
    var_25 = var_29;
    if (var_25) {
        var_31 = wp::extract(var_shift, var_30);
        var_33 = (var_31 == var_32);
        var_25 = var_25 && var_33;
    }
    if (var_25) {
        var_35 = wp::extract(var_shift, var_34);
        var_37 = (var_35 == var_36);
        var_25 = var_25 && var_37;
    }
    // if _zero_shift:                                                                        <L 21>
    if (var_25) {
        // jatom_end = iatom                                                                  <L 22>
        var_38 = wp::copy(var_iatom);
    }
    var_39 = wp::where(var_25, var_38, var_4);
    // for jatom in range(jatom_start, jatom_end):                                            <L 23>
    var_40 = wp::range(var_1, var_39);
    start_for_0:;
        if (iter_cmp(var_40) == 0) goto end_for_0;
        var_41 = wp::iter_next(var_40);
        // _pos_j = positions[jatom]                                                          <L 24>
        var_42 = wp::address(var_positions, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // diff = positions_shifted - _pos_j                                                  <L 25>
        var_45 = wp::sub(var_24, var_43);
        // dist_sq = wp.length_sq(diff)                                                       <L 26>
        var_46 = wp::length_sq(var_45);
        // if dist_sq < cutoff_sq:                                                            <L 27>
        var_47 = (var_46 < var_cutoff_sq);
        if (var_47) {
            // _int_j = per_atom_cell_offsets[jatom]                                          <L 28>
            var_48 = wp::address(var_per_atom_cell_offsets, var_41);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // _corrected_shift = wp.vec3i(                                                   <L 29>
            // shift[0] - _int_i[0] + _int_j[0],                                              <L 30>
            var_52 = wp::extract(var_shift, var_51);
            var_54 = wp::extract(var_18, var_53);
            var_55 = wp::sub(var_52, var_54);
            var_57 = wp::extract(var_49, var_56);
            var_58 = wp::add(var_55, var_57);
            // shift[1] - _int_i[1] + _int_j[1],                                              <L 31>
            var_60 = wp::extract(var_shift, var_59);
            var_62 = wp::extract(var_18, var_61);
            var_63 = wp::sub(var_60, var_62);
            var_65 = wp::extract(var_49, var_64);
            var_66 = wp::add(var_63, var_65);
            // shift[2] - _int_i[2] + _int_j[2],                                              <L 32>
            var_68 = wp::extract(var_shift, var_67);
            var_70 = wp::extract(var_18, var_69);
            var_71 = wp::sub(var_68, var_70);
            var_73 = wp::extract(var_49, var_72);
            var_74 = wp::add(var_71, var_73);
            var_75 = wp::vec_t<3, wp::int32>(var_58, var_66, var_74);
            // _update_neighbor_matrix_pbc(                                                   <L 34>
            // jatom,                                                                         <L 35>
            // iatom,                                                                         <L 36>
            // neighbor_matrix,                                                               <L 37>
            // neighbor_matrix_shifts,                                                        <L 38>
            // num_neighbors,                                                                 <L 39>
            // _corrected_shift,                                                              <L 40>
            // maxnb,                                                                         <L 41>
            // half_fill,                                                                     <L 42>
            _update_neighbor_matrix_pbc_0(var_41, var_iatom, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_75, var_8, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_pbc_body_prewrapped_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::shape_t* var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::shape_t var_9;
    const wp::int32 var_10 = 0;
    wp::mat_t<3, 3, wp::float32>* var_11;
    wp::mat_t<3, 3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::vec_t<3, wp::float32>* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    bool var_22;
    const wp::int32 var_23 = 0;
    wp::int32 var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    const wp::int32 var_29 = 0;
    bool var_30;
    const wp::int32 var_31 = 2;
    wp::int32 var_32;
    const wp::int32 var_33 = 0;
    bool var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::range_t var_37;
    wp::int32 var_38;
    wp::vec_t<3, wp::float32>* var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    bool var_44;
    //---------
    // forward
    // def _naive_neighbor_pbc_body_prewrapped(                                               <L 1>
    // jatom_start = wp.int32(0)                                                              <L 12>
    var_1 = wp::int32(var_0);
    // jatom_end = positions.shape[0]                                                         <L 13>
    var_2 = &(var_positions.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    // maxnb = neighbor_matrix.shape[1]                                                       <L 14>
    var_6 = &(var_neighbor_matrix.shape);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    // _cell = cell[0]                                                                        <L 15>
    var_11 = wp::address(var_cell, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // _pos_i = positions[iatom]                                                              <L 16>
    var_14 = wp::address(var_positions, var_iatom);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // positions_shifted = type(_cell[0])(shift) * _cell + _pos_i                             <L 17>
    var_18 = wp::extract(var_12, var_17);
    var_19 = wp::vec_t<3, wp::float32>(var_shift);
    var_20 = wp::mul(var_19, var_12);
    var_21 = wp::add(var_20, var_15);
    // _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0                        <L 18>
    var_24 = wp::extract(var_shift, var_23);
    var_26 = (var_24 == var_25);
    var_22 = var_26;
    if (var_22) {
        var_28 = wp::extract(var_shift, var_27);
        var_30 = (var_28 == var_29);
        var_22 = var_22 && var_30;
    }
    if (var_22) {
        var_32 = wp::extract(var_shift, var_31);
        var_34 = (var_32 == var_33);
        var_22 = var_22 && var_34;
    }
    // if _zero_shift:                                                                        <L 19>
    if (var_22) {
        // jatom_end = iatom                                                                  <L 20>
        var_35 = wp::copy(var_iatom);
    }
    var_36 = wp::where(var_22, var_35, var_4);
    // for jatom in range(jatom_start, jatom_end):                                            <L 21>
    var_37 = wp::range(var_1, var_36);
    start_for_0:;
        if (iter_cmp(var_37) == 0) goto end_for_0;
        var_38 = wp::iter_next(var_37);
        // _pos_j = positions[jatom]                                                          <L 22>
        var_39 = wp::address(var_positions, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // diff = positions_shifted - _pos_j                                                  <L 23>
        var_42 = wp::sub(var_21, var_40);
        // dist_sq = wp.length_sq(diff)                                                       <L 24>
        var_43 = wp::length_sq(var_42);
        // if dist_sq < cutoff_sq:                                                            <L 25>
        var_44 = (var_43 < var_cutoff_sq);
        if (var_44) {
            // _update_neighbor_matrix_pbc(                                                   <L 26>
            // jatom,                                                                         <L 27>
            // iatom,                                                                         <L 28>
            // neighbor_matrix,                                                               <L 29>
            // neighbor_matrix_shifts,                                                        <L 30>
            // num_neighbors,                                                                 <L 31>
            // shift,                                                                         <L 32>
            // maxnb,                                                                         <L 33>
            // half_fill,                                                                     <L 34>
            _update_neighbor_matrix_pbc_0(var_38, var_iatom, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_shift, var_8, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_pbc_body_prewrapped_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::shape_t* var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::shape_t var_9;
    const wp::int32 var_10 = 0;
    wp::mat_t<3, 3, wp::float64>* var_11;
    wp::mat_t<3, 3, wp::float64> var_12;
    wp::mat_t<3, 3, wp::float64> var_13;
    wp::vec_t<3, wp::float64>* var_14;
    wp::vec_t<3, wp::float64> var_15;
    wp::vec_t<3, wp::float64> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float64> var_18;
    wp::vec_t<3, wp::float64> var_19;
    wp::vec_t<3, wp::float64> var_20;
    wp::vec_t<3, wp::float64> var_21;
    bool var_22;
    const wp::int32 var_23 = 0;
    wp::int32 var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    const wp::int32 var_29 = 0;
    bool var_30;
    const wp::int32 var_31 = 2;
    wp::int32 var_32;
    const wp::int32 var_33 = 0;
    bool var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::range_t var_37;
    wp::int32 var_38;
    wp::vec_t<3, wp::float64>* var_39;
    wp::vec_t<3, wp::float64> var_40;
    wp::vec_t<3, wp::float64> var_41;
    wp::vec_t<3, wp::float64> var_42;
    wp::float64 var_43;
    bool var_44;
    //---------
    // forward
    // def _naive_neighbor_pbc_body_prewrapped(                                               <L 1>
    // jatom_start = wp.int32(0)                                                              <L 12>
    var_1 = wp::int32(var_0);
    // jatom_end = positions.shape[0]                                                         <L 13>
    var_2 = &(var_positions.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    // maxnb = neighbor_matrix.shape[1]                                                       <L 14>
    var_6 = &(var_neighbor_matrix.shape);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    // _cell = cell[0]                                                                        <L 15>
    var_11 = wp::address(var_cell, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // _pos_i = positions[iatom]                                                              <L 16>
    var_14 = wp::address(var_positions, var_iatom);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // positions_shifted = type(_cell[0])(shift) * _cell + _pos_i                             <L 17>
    var_18 = wp::extract(var_12, var_17);
    var_19 = wp::vec_t<3, wp::float64>(var_shift);
    var_20 = wp::mul(var_19, var_12);
    var_21 = wp::add(var_20, var_15);
    // _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0                        <L 18>
    var_24 = wp::extract(var_shift, var_23);
    var_26 = (var_24 == var_25);
    var_22 = var_26;
    if (var_22) {
        var_28 = wp::extract(var_shift, var_27);
        var_30 = (var_28 == var_29);
        var_22 = var_22 && var_30;
    }
    if (var_22) {
        var_32 = wp::extract(var_shift, var_31);
        var_34 = (var_32 == var_33);
        var_22 = var_22 && var_34;
    }
    // if _zero_shift:                                                                        <L 19>
    if (var_22) {
        // jatom_end = iatom                                                                  <L 20>
        var_35 = wp::copy(var_iatom);
    }
    var_36 = wp::where(var_22, var_35, var_4);
    // for jatom in range(jatom_start, jatom_end):                                            <L 21>
    var_37 = wp::range(var_1, var_36);
    start_for_0:;
        if (iter_cmp(var_37) == 0) goto end_for_0;
        var_38 = wp::iter_next(var_37);
        // _pos_j = positions[jatom]                                                          <L 22>
        var_39 = wp::address(var_positions, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // diff = positions_shifted - _pos_j                                                  <L 23>
        var_42 = wp::sub(var_21, var_40);
        // dist_sq = wp.length_sq(diff)                                                       <L 24>
        var_43 = wp::length_sq(var_42);
        // if dist_sq < cutoff_sq:                                                            <L 25>
        var_44 = (var_43 < var_cutoff_sq);
        if (var_44) {
            // _update_neighbor_matrix_pbc(                                                   <L 26>
            // jatom,                                                                         <L 27>
            // iatom,                                                                         <L 28>
            // neighbor_matrix,                                                               <L 29>
            // neighbor_matrix_shifts,                                                        <L 30>
            // num_neighbors,                                                                 <L 31>
            // shift,                                                                         <L 32>
            // maxnb,                                                                         <L 33>
            // half_fill,                                                                     <L 34>
            _update_neighbor_matrix_pbc_0(var_38, var_iatom, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_shift, var_8, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void _naive_neighbor_pbc_body_prewrapped_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::shape_t* var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::shape_t var_9;
    const wp::int32 var_10 = 0;
    wp::mat_t<3, 3, wp::float16>* var_11;
    wp::mat_t<3, 3, wp::float16> var_12;
    wp::mat_t<3, 3, wp::float16> var_13;
    wp::vec_t<3, wp::float16>* var_14;
    wp::vec_t<3, wp::float16> var_15;
    wp::vec_t<3, wp::float16> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float16> var_18;
    wp::vec_t<3, wp::float16> var_19;
    wp::vec_t<3, wp::float16> var_20;
    wp::vec_t<3, wp::float16> var_21;
    bool var_22;
    const wp::int32 var_23 = 0;
    wp::int32 var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    const wp::int32 var_29 = 0;
    bool var_30;
    const wp::int32 var_31 = 2;
    wp::int32 var_32;
    const wp::int32 var_33 = 0;
    bool var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::range_t var_37;
    wp::int32 var_38;
    wp::vec_t<3, wp::float16>* var_39;
    wp::vec_t<3, wp::float16> var_40;
    wp::vec_t<3, wp::float16> var_41;
    wp::vec_t<3, wp::float16> var_42;
    wp::float16 var_43;
    bool var_44;
    //---------
    // forward
    // def _naive_neighbor_pbc_body_prewrapped(                                               <L 1>
    // jatom_start = wp.int32(0)                                                              <L 12>
    var_1 = wp::int32(var_0);
    // jatom_end = positions.shape[0]                                                         <L 13>
    var_2 = &(var_positions.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    // maxnb = neighbor_matrix.shape[1]                                                       <L 14>
    var_6 = &(var_neighbor_matrix.shape);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    // _cell = cell[0]                                                                        <L 15>
    var_11 = wp::address(var_cell, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // _pos_i = positions[iatom]                                                              <L 16>
    var_14 = wp::address(var_positions, var_iatom);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // positions_shifted = type(_cell[0])(shift) * _cell + _pos_i                             <L 17>
    var_18 = wp::extract(var_12, var_17);
    var_19 = wp::vec_t<3, wp::float16>(var_shift);
    var_20 = wp::mul(var_19, var_12);
    var_21 = wp::add(var_20, var_15);
    // _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0                        <L 18>
    var_24 = wp::extract(var_shift, var_23);
    var_26 = (var_24 == var_25);
    var_22 = var_26;
    if (var_22) {
        var_28 = wp::extract(var_shift, var_27);
        var_30 = (var_28 == var_29);
        var_22 = var_22 && var_30;
    }
    if (var_22) {
        var_32 = wp::extract(var_shift, var_31);
        var_34 = (var_32 == var_33);
        var_22 = var_22 && var_34;
    }
    // if _zero_shift:                                                                        <L 19>
    if (var_22) {
        // jatom_end = iatom                                                                  <L 20>
        var_35 = wp::copy(var_iatom);
    }
    var_36 = wp::where(var_22, var_35, var_4);
    // for jatom in range(jatom_start, jatom_end):                                            <L 21>
    var_37 = wp::range(var_1, var_36);
    start_for_0:;
        if (iter_cmp(var_37) == 0) goto end_for_0;
        var_38 = wp::iter_next(var_37);
        // _pos_j = positions[jatom]                                                          <L 22>
        var_39 = wp::address(var_positions, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // diff = positions_shifted - _pos_j                                                  <L 23>
        var_42 = wp::sub(var_21, var_40);
        // dist_sq = wp.length_sq(diff)                                                       <L 24>
        var_43 = wp::length_sq(var_42);
        // if dist_sq < cutoff_sq:                                                            <L 25>
        var_44 = (var_43 < var_cutoff_sq);
        if (var_44) {
            // _update_neighbor_matrix_pbc(                                                   <L 26>
            // jatom,                                                                         <L 27>
            // iatom,                                                                         <L 28>
            // neighbor_matrix,                                                               <L 29>
            // neighbor_matrix_shifts,                                                        <L 30>
            // num_neighbors,                                                                 <L 31>
            // shift,                                                                         <L 32>
            // maxnb,                                                                         <L 33>
            // half_fill,                                                                     <L 34>
            _update_neighbor_matrix_pbc_0(var_38, var_iatom, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_shift, var_8, var_half_fill);
        }
        goto start_for_0;
    end_for_0:;
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/neighbor_utils.py:204
static CUDA_CALLABLE void adj__update_neighbor_matrix_0(
    wp::int32 var_i,
    wp::int32 var_j,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    wp::int32 var_max_neighbors,
    bool var_half_fill,
    wp::int32 & adj_i,
    wp::int32 & adj_j,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::int32> & adj_num_neighbors,
    wp::int32 & adj_max_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_body_0(
    wp::int32 var_tid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::int32 & adj_tid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_positions,
    wp::float32 & adj_cutoff_sq,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_body_0(
    wp::int32 var_tid,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::int32 & adj_tid,
    wp::array_t<wp::vec_t<3, wp::float64>> & adj_positions,
    wp::float64 & adj_cutoff_sq,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_body_0(
    wp::int32 var_tid,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::int32 & adj_tid,
    wp::array_t<wp::vec_t<3, wp::float16>> & adj_positions,
    wp::float16 & adj_cutoff_sq,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/neighbor_utils.py:159
static CUDA_CALLABLE void adj__decode_shift_index_0(
    wp::int32 var_local_idx,
    wp::vec_t<3, wp::int32> var_shift_range,
    wp::int32 & adj_local_idx,
    wp::vec_t<3, wp::int32> & adj_shift_range,
    wp::vec_t<3, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/neighbor_utils.py:240
static CUDA_CALLABLE void adj__update_neighbor_matrix_pbc_0(
    wp::int32 var_i,
    wp::int32 var_j,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    wp::vec_t<3, wp::int32> var_unit_shift,
    wp::int32 var_max_neighbors,
    bool var_half_fill,
    wp::int32 & adj_i,
    wp::int32 & adj_j,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_neighbor_matrix_shifts,
    wp::array_t<wp::int32> & adj_num_neighbors,
    wp::vec_t<3, wp::int32> & adj_unit_shift,
    wp::int32 & adj_max_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_pbc_body_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::vec_t<3, wp::int32> & adj_shift,
    wp::int32 & adj_iatom,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_per_atom_cell_offsets,
    wp::float32 & adj_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cell,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_neighbor_matrix_shifts,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_pbc_body_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::vec_t<3, wp::int32> & adj_shift,
    wp::int32 & adj_iatom,
    wp::array_t<wp::vec_t<3, wp::float64>> & adj_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_per_atom_cell_offsets,
    wp::float64 & adj_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> & adj_cell,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_neighbor_matrix_shifts,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_pbc_body_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::vec_t<3, wp::int32> & adj_shift,
    wp::int32 & adj_iatom,
    wp::array_t<wp::vec_t<3, wp::float16>> & adj_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_per_atom_cell_offsets,
    wp::float16 & adj_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> & adj_cell,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_neighbor_matrix_shifts,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_pbc_body_prewrapped_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::vec_t<3, wp::int32> & adj_shift,
    wp::int32 & adj_iatom,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_positions,
    wp::float32 & adj_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cell,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_neighbor_matrix_shifts,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_pbc_body_prewrapped_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::vec_t<3, wp::int32> & adj_shift,
    wp::int32 & adj_iatom,
    wp::array_t<wp::vec_t<3, wp::float64>> & adj_positions,
    wp::float64 & adj_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> & adj_cell,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_neighbor_matrix_shifts,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/bitnami/python/lib/python3.12/site-packages/nvalchemiops/neighbors/naive.py:0
static CUDA_CALLABLE void adj__naive_neighbor_pbc_body_prewrapped_0(
    wp::vec_t<3, wp::int32> var_shift,
    wp::int32 var_iatom,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
    wp::vec_t<3, wp::int32> & adj_shift,
    wp::int32 & adj_iatom,
    wp::array_t<wp::vec_t<3, wp::float16>> & adj_positions,
    wp::float16 & adj_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> & adj_cell,
    wp::array_t<wp::int32> & adj_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_neighbor_matrix_shifts,
    wp::array_t<wp::int32> & adj_num_neighbors,
    bool & adj_half_fill)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _fill_naive_neighbor_matrix_cbdb0640_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        //---------
        // forward
        // def _fill_naive_neighbor_matrix(                                                       <L 1>
        // tid = wp.tid()                                                                         <L 45>
        var_0 = builtin_tid1d();
        // _naive_neighbor_body(                                                                  <L 46>
        // tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill                   <L 47>
        _naive_neighbor_body_0(var_0, var_positions, var_cutoff_sq, var_neighbor_matrix, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_85ec8d93_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        //---------
        // forward
        // def _fill_naive_neighbor_matrix(                                                       <L 1>
        // tid = wp.tid()                                                                         <L 45>
        var_0 = builtin_tid1d();
        // _naive_neighbor_body(                                                                  <L 46>
        // tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill                   <L 47>
        _naive_neighbor_body_0(var_0, var_positions, var_cutoff_sq, var_neighbor_matrix, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_afa73c0c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        //---------
        // forward
        // def _fill_naive_neighbor_matrix(                                                       <L 1>
        // tid = wp.tid()                                                                         <L 45>
        var_0 = builtin_tid1d();
        // _naive_neighbor_body(                                                                  <L 46>
        // tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill                   <L 47>
        _naive_neighbor_body_0(var_0, var_positions, var_cutoff_sq, var_neighbor_matrix, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_selective_a41c59d9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        bool var_4;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_selective(                                             <L 1>
        // tid = wp.tid()                                                                         <L 31>
        var_0 = builtin_tid1d();
        // if not rebuild_flags[0]:                                                               <L 32>
        var_2 = wp::address(var_rebuild_flags, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::unot(var_4);
        if (var_3) {
            // return                                                                             <L 33>
            continue;
        }
        // _naive_neighbor_body(                                                                  <L 34>
        // tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill                   <L 35>
        _naive_neighbor_body_0(var_0, var_positions, var_cutoff_sq, var_neighbor_matrix, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_selective_0603a96f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        bool var_4;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_selective(                                             <L 1>
        // tid = wp.tid()                                                                         <L 31>
        var_0 = builtin_tid1d();
        // if not rebuild_flags[0]:                                                               <L 32>
        var_2 = wp::address(var_rebuild_flags, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::unot(var_4);
        if (var_3) {
            // return                                                                             <L 33>
            continue;
        }
        // _naive_neighbor_body(                                                                  <L 34>
        // tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill                   <L 35>
        _naive_neighbor_body_0(var_0, var_positions, var_cutoff_sq, var_neighbor_matrix, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_selective_84120697_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        bool var_4;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_selective(                                             <L 1>
        // tid = wp.tid()                                                                         <L 31>
        var_0 = builtin_tid1d();
        // if not rebuild_flags[0]:                                                               <L 32>
        var_2 = wp::address(var_rebuild_flags, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::unot(var_4);
        if (var_3) {
            // return                                                                             <L 33>
            continue;
        }
        // _naive_neighbor_body(                                                                  <L 34>
        // tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill                   <L 35>
        _naive_neighbor_body_0(var_0, var_positions, var_cutoff_sq, var_neighbor_matrix, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_655f0a4b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::int32>* var_3;
        wp::vec_t<3, wp::int32> var_4;
        wp::vec_t<3, wp::int32> var_5;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc(                                                   <L 1>
        // ishift, iatom = wp.tid()                                                               <L 61>
        builtin_tid2d(var_0, var_1);
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 62>
        var_3 = wp::address(var_shift_range, var_2);
        var_5 = wp::load(var_3);
        var_4 = _decode_shift_index_0(var_0, var_5);
        // _naive_neighbor_pbc_body(                                                              <L 63>
        // shift,                                                                                 <L 64>
        // iatom,                                                                                 <L 65>
        // positions,                                                                             <L 66>
        // per_atom_cell_offsets,                                                                 <L 67>
        // cutoff_sq,                                                                             <L 68>
        // cell,                                                                                  <L 69>
        // neighbor_matrix,                                                                       <L 70>
        // neighbor_matrix_shifts,                                                                <L 71>
        // num_neighbors,                                                                         <L 72>
        // half_fill,                                                                             <L 73>
        _naive_neighbor_pbc_body_0(var_4, var_1, var_positions, var_per_atom_cell_offsets, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_ba578788_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::int32>* var_3;
        wp::vec_t<3, wp::int32> var_4;
        wp::vec_t<3, wp::int32> var_5;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc(                                                   <L 1>
        // ishift, iatom = wp.tid()                                                               <L 61>
        builtin_tid2d(var_0, var_1);
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 62>
        var_3 = wp::address(var_shift_range, var_2);
        var_5 = wp::load(var_3);
        var_4 = _decode_shift_index_0(var_0, var_5);
        // _naive_neighbor_pbc_body(                                                              <L 63>
        // shift,                                                                                 <L 64>
        // iatom,                                                                                 <L 65>
        // positions,                                                                             <L 66>
        // per_atom_cell_offsets,                                                                 <L 67>
        // cutoff_sq,                                                                             <L 68>
        // cell,                                                                                  <L 69>
        // neighbor_matrix,                                                                       <L 70>
        // neighbor_matrix_shifts,                                                                <L 71>
        // num_neighbors,                                                                         <L 72>
        // half_fill,                                                                             <L 73>
        _naive_neighbor_pbc_body_0(var_4, var_1, var_positions, var_per_atom_cell_offsets, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_8ee96967_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::int32>* var_3;
        wp::vec_t<3, wp::int32> var_4;
        wp::vec_t<3, wp::int32> var_5;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc(                                                   <L 1>
        // ishift, iatom = wp.tid()                                                               <L 61>
        builtin_tid2d(var_0, var_1);
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 62>
        var_3 = wp::address(var_shift_range, var_2);
        var_5 = wp::load(var_3);
        var_4 = _decode_shift_index_0(var_0, var_5);
        // _naive_neighbor_pbc_body(                                                              <L 63>
        // shift,                                                                                 <L 64>
        // iatom,                                                                                 <L 65>
        // positions,                                                                             <L 66>
        // per_atom_cell_offsets,                                                                 <L 67>
        // cutoff_sq,                                                                             <L 68>
        // cell,                                                                                  <L 69>
        // neighbor_matrix,                                                                       <L 70>
        // neighbor_matrix_shifts,                                                                <L 71>
        // num_neighbors,                                                                         <L 72>
        // half_fill,                                                                             <L 73>
        _naive_neighbor_pbc_body_0(var_4, var_1, var_positions, var_per_atom_cell_offsets, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_prewrapped_selective_63f409b8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::int32>* var_7;
        wp::vec_t<3, wp::int32> var_8;
        wp::vec_t<3, wp::int32> var_9;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_prewrapped_selective(                              <L 1>
        // ishift, iatom = wp.tid()                                                               <L 18>
        builtin_tid2d(var_0, var_1);
        // if not rebuild_flags[0]:                                                               <L 19>
        var_3 = wp::address(var_rebuild_flags, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 20>
            continue;
        }
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 21>
        var_7 = wp::address(var_shift_range, var_6);
        var_9 = wp::load(var_7);
        var_8 = _decode_shift_index_0(var_0, var_9);
        // _naive_neighbor_pbc_body_prewrapped(                                                   <L 22>
        // shift,                                                                                 <L 23>
        // iatom,                                                                                 <L 24>
        // positions,                                                                             <L 25>
        // cutoff_sq,                                                                             <L 26>
        // cell,                                                                                  <L 27>
        // neighbor_matrix,                                                                       <L 28>
        // neighbor_matrix_shifts,                                                                <L 29>
        // num_neighbors,                                                                         <L 30>
        // half_fill,                                                                             <L 31>
        _naive_neighbor_pbc_body_prewrapped_0(var_8, var_1, var_positions, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_prewrapped_selective_88f8d646_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::int32>* var_7;
        wp::vec_t<3, wp::int32> var_8;
        wp::vec_t<3, wp::int32> var_9;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_prewrapped_selective(                              <L 1>
        // ishift, iatom = wp.tid()                                                               <L 18>
        builtin_tid2d(var_0, var_1);
        // if not rebuild_flags[0]:                                                               <L 19>
        var_3 = wp::address(var_rebuild_flags, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 20>
            continue;
        }
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 21>
        var_7 = wp::address(var_shift_range, var_6);
        var_9 = wp::load(var_7);
        var_8 = _decode_shift_index_0(var_0, var_9);
        // _naive_neighbor_pbc_body_prewrapped(                                                   <L 22>
        // shift,                                                                                 <L 23>
        // iatom,                                                                                 <L 24>
        // positions,                                                                             <L 25>
        // cutoff_sq,                                                                             <L 26>
        // cell,                                                                                  <L 27>
        // neighbor_matrix,                                                                       <L 28>
        // neighbor_matrix_shifts,                                                                <L 29>
        // num_neighbors,                                                                         <L 30>
        // half_fill,                                                                             <L 31>
        _naive_neighbor_pbc_body_prewrapped_0(var_8, var_1, var_positions, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_prewrapped_selective_17b99259_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::int32>* var_7;
        wp::vec_t<3, wp::int32> var_8;
        wp::vec_t<3, wp::int32> var_9;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_prewrapped_selective(                              <L 1>
        // ishift, iatom = wp.tid()                                                               <L 18>
        builtin_tid2d(var_0, var_1);
        // if not rebuild_flags[0]:                                                               <L 19>
        var_3 = wp::address(var_rebuild_flags, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 20>
            continue;
        }
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 21>
        var_7 = wp::address(var_shift_range, var_6);
        var_9 = wp::load(var_7);
        var_8 = _decode_shift_index_0(var_0, var_9);
        // _naive_neighbor_pbc_body_prewrapped(                                                   <L 22>
        // shift,                                                                                 <L 23>
        // iatom,                                                                                 <L 24>
        // positions,                                                                             <L 25>
        // cutoff_sq,                                                                             <L 26>
        // cell,                                                                                  <L 27>
        // neighbor_matrix,                                                                       <L 28>
        // neighbor_matrix_shifts,                                                                <L 29>
        // num_neighbors,                                                                         <L 30>
        // half_fill,                                                                             <L 31>
        _naive_neighbor_pbc_body_prewrapped_0(var_8, var_1, var_positions, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_prewrapped_0bbe9bdb_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::int32>* var_3;
        wp::vec_t<3, wp::int32> var_4;
        wp::vec_t<3, wp::int32> var_5;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_prewrapped(                                        <L 1>
        // ishift, iatom = wp.tid()                                                               <L 17>
        builtin_tid2d(var_0, var_1);
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 18>
        var_3 = wp::address(var_shift_range, var_2);
        var_5 = wp::load(var_3);
        var_4 = _decode_shift_index_0(var_0, var_5);
        // _naive_neighbor_pbc_body_prewrapped(                                                   <L 19>
        // shift,                                                                                 <L 20>
        // iatom,                                                                                 <L 21>
        // positions,                                                                             <L 22>
        // cutoff_sq,                                                                             <L 23>
        // cell,                                                                                  <L 24>
        // neighbor_matrix,                                                                       <L 25>
        // neighbor_matrix_shifts,                                                                <L 26>
        // num_neighbors,                                                                         <L 27>
        // half_fill,                                                                             <L 28>
        _naive_neighbor_pbc_body_prewrapped_0(var_4, var_1, var_positions, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_prewrapped_9092bf0f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::int32>* var_3;
        wp::vec_t<3, wp::int32> var_4;
        wp::vec_t<3, wp::int32> var_5;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_prewrapped(                                        <L 1>
        // ishift, iatom = wp.tid()                                                               <L 17>
        builtin_tid2d(var_0, var_1);
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 18>
        var_3 = wp::address(var_shift_range, var_2);
        var_5 = wp::load(var_3);
        var_4 = _decode_shift_index_0(var_0, var_5);
        // _naive_neighbor_pbc_body_prewrapped(                                                   <L 19>
        // shift,                                                                                 <L 20>
        // iatom,                                                                                 <L 21>
        // positions,                                                                             <L 22>
        // cutoff_sq,                                                                             <L 23>
        // cell,                                                                                  <L 24>
        // neighbor_matrix,                                                                       <L 25>
        // neighbor_matrix_shifts,                                                                <L 26>
        // num_neighbors,                                                                         <L 27>
        // half_fill,                                                                             <L 28>
        _naive_neighbor_pbc_body_prewrapped_0(var_4, var_1, var_positions, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_prewrapped_e41a2353_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill)
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        wp::vec_t<3, wp::int32>* var_3;
        wp::vec_t<3, wp::int32> var_4;
        wp::vec_t<3, wp::int32> var_5;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_prewrapped(                                        <L 1>
        // ishift, iatom = wp.tid()                                                               <L 17>
        builtin_tid2d(var_0, var_1);
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 18>
        var_3 = wp::address(var_shift_range, var_2);
        var_5 = wp::load(var_3);
        var_4 = _decode_shift_index_0(var_0, var_5);
        // _naive_neighbor_pbc_body_prewrapped(                                                   <L 19>
        // shift,                                                                                 <L 20>
        // iatom,                                                                                 <L 21>
        // positions,                                                                             <L 22>
        // cutoff_sq,                                                                             <L 23>
        // cell,                                                                                  <L 24>
        // neighbor_matrix,                                                                       <L 25>
        // neighbor_matrix_shifts,                                                                <L 26>
        // num_neighbors,                                                                         <L 27>
        // half_fill,                                                                             <L 28>
        _naive_neighbor_pbc_body_prewrapped_0(var_4, var_1, var_positions, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_selective_99aedecd_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float32 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::int32>* var_7;
        wp::vec_t<3, wp::int32> var_8;
        wp::vec_t<3, wp::int32> var_9;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_selective(                                         <L 1>
        // ishift, iatom = wp.tid()                                                               <L 46>
        builtin_tid2d(var_0, var_1);
        // if not rebuild_flags[0]:                                                               <L 47>
        var_3 = wp::address(var_rebuild_flags, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 48>
            continue;
        }
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 49>
        var_7 = wp::address(var_shift_range, var_6);
        var_9 = wp::load(var_7);
        var_8 = _decode_shift_index_0(var_0, var_9);
        // _naive_neighbor_pbc_body(                                                              <L 50>
        // shift,                                                                                 <L 51>
        // iatom,                                                                                 <L 52>
        // positions,                                                                             <L 53>
        // per_atom_cell_offsets,                                                                 <L 54>
        // cutoff_sq,                                                                             <L 55>
        // cell,                                                                                  <L 56>
        // neighbor_matrix,                                                                       <L 57>
        // neighbor_matrix_shifts,                                                                <L 58>
        // num_neighbors,                                                                         <L 59>
        // half_fill,                                                                             <L 60>
        _naive_neighbor_pbc_body_0(var_8, var_1, var_positions, var_per_atom_cell_offsets, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_selective_217fa007_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float64>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float64 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float64>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::int32>* var_7;
        wp::vec_t<3, wp::int32> var_8;
        wp::vec_t<3, wp::int32> var_9;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_selective(                                         <L 1>
        // ishift, iatom = wp.tid()                                                               <L 46>
        builtin_tid2d(var_0, var_1);
        // if not rebuild_flags[0]:                                                               <L 47>
        var_3 = wp::address(var_rebuild_flags, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 48>
            continue;
        }
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 49>
        var_7 = wp::address(var_shift_range, var_6);
        var_9 = wp::load(var_7);
        var_8 = _decode_shift_index_0(var_0, var_9);
        // _naive_neighbor_pbc_body(                                                              <L 50>
        // shift,                                                                                 <L 51>
        // iatom,                                                                                 <L 52>
        // positions,                                                                             <L 53>
        // per_atom_cell_offsets,                                                                 <L 54>
        // cutoff_sq,                                                                             <L 55>
        // cell,                                                                                  <L 56>
        // neighbor_matrix,                                                                       <L 57>
        // neighbor_matrix_shifts,                                                                <L 58>
        // num_neighbors,                                                                         <L 59>
        // half_fill,                                                                             <L 60>
        _naive_neighbor_pbc_body_0(var_8, var_1, var_positions, var_per_atom_cell_offsets, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}



extern "C" __global__ void _fill_naive_neighbor_matrix_pbc_selective_375ae5a6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float16>> var_positions,
    wp::array_t<wp::vec_t<3, wp::int32>> var_per_atom_cell_offsets,
    wp::float16 var_cutoff_sq,
    wp::array_t<wp::mat_t<3, 3, wp::float16>> var_cell,
    wp::array_t<wp::vec_t<3, wp::int32>> var_shift_range,
    wp::array_t<wp::int32> var_neighbor_matrix,
    wp::array_t<wp::vec_t<3, wp::int32>> var_neighbor_matrix_shifts,
    wp::array_t<wp::int32> var_num_neighbors,
    bool var_half_fill,
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
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        wp::vec_t<3, wp::int32>* var_7;
        wp::vec_t<3, wp::int32> var_8;
        wp::vec_t<3, wp::int32> var_9;
        //---------
        // forward
        // def _fill_naive_neighbor_matrix_pbc_selective(                                         <L 1>
        // ishift, iatom = wp.tid()                                                               <L 46>
        builtin_tid2d(var_0, var_1);
        // if not rebuild_flags[0]:                                                               <L 47>
        var_3 = wp::address(var_rebuild_flags, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 48>
            continue;
        }
        // shift = _decode_shift_index(ishift, shift_range[0])                                    <L 49>
        var_7 = wp::address(var_shift_range, var_6);
        var_9 = wp::load(var_7);
        var_8 = _decode_shift_index_0(var_0, var_9);
        // _naive_neighbor_pbc_body(                                                              <L 50>
        // shift,                                                                                 <L 51>
        // iatom,                                                                                 <L 52>
        // positions,                                                                             <L 53>
        // per_atom_cell_offsets,                                                                 <L 54>
        // cutoff_sq,                                                                             <L 55>
        // cell,                                                                                  <L 56>
        // neighbor_matrix,                                                                       <L 57>
        // neighbor_matrix_shifts,                                                                <L 58>
        // num_neighbors,                                                                         <L 59>
        // half_fill,                                                                             <L 60>
        _naive_neighbor_pbc_body_0(var_8, var_1, var_positions, var_per_atom_cell_offsets, var_cutoff_sq, var_cell, var_neighbor_matrix, var_neighbor_matrix_shifts, var_num_neighbors, var_half_fill);
    }
}

