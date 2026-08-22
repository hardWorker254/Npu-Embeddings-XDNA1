module {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %Y_row1_buff_0 = aie.buffer(%tile_0_5) {address = 8192 : i32, sym_name = "Y_row1_buff_0"} : memref<3072xf32> 
    %Y_row1_buff_1 = aie.buffer(%tile_0_5) {address = 20480 : i32, sym_name = "Y_row1_buff_1"} : memref<3072xf32> 
    %Y_row1_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "Y_row1_prod_lock_0"}
    %Y_row1_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "Y_row1_cons_lock_0"}
    %B_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, sym_name = "B_L3L2_cons_buff_0"} : memref<3072xbf16> 
    %B_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 137216 : i32, sym_name = "B_L3L2_cons_buff_1"} : memref<3072xbf16> 
    %B_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 12) {init = 2 : i32, sym_name = "B_L3L2_cons_prod_lock_0"}
    %B_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 13) {init = 0 : i32, sym_name = "B_L3L2_cons_cons_lock_0"}
    %Y_row0_buff_0 = aie.buffer(%tile_0_4) {address = 8192 : i32, sym_name = "Y_row0_buff_0"} : memref<3072xf32> 
    %Y_row0_buff_1 = aie.buffer(%tile_0_4) {address = 20480 : i32, sym_name = "Y_row0_buff_1"} : memref<3072xf32> 
    %Y_row0_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "Y_row0_prod_lock_0"}
    %Y_row0_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "Y_row0_cons_lock_0"}
    %Y_mem_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "Y_mem_cons_prod_lock_0"}
    %Y_mem_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "Y_mem_cons_cons_lock_0"}
    %C_out_1_buff_0 = aie.buffer(%tile_0_3) {address = 3328 : i32, sym_name = "C_out_1_buff_0"} : memref<64x48xf32> 
    %C_out_1_buff_1 = aie.buffer(%tile_0_3) {address = 15616 : i32, sym_name = "C_out_1_buff_1"} : memref<64x48xf32> 
    %C_out_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "C_out_1_prod_lock_0"}
    %C_out_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "C_out_1_cons_lock_0"}
    %C_in_1_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, sym_name = "C_in_1_cons_buff_0"} : memref<64x48xf32> 
    %C_in_1_cons_buff_1 = aie.buffer(%tile_0_5) {address = 45056 : i32, sym_name = "C_in_1_cons_buff_1"} : memref<64x48xf32> 
    %C_in_1_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "C_in_1_cons_prod_lock_0"}
    %C_in_1_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "C_in_1_cons_cons_lock_0"}
    %C_out_0_buff_0 = aie.buffer(%tile_0_2) {address = 3328 : i32, sym_name = "C_out_0_buff_0"} : memref<64x48xf32> 
    %C_out_0_buff_1 = aie.buffer(%tile_0_2) {address = 15616 : i32, sym_name = "C_out_0_buff_1"} : memref<64x48xf32> 
    %C_out_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "C_out_0_prod_lock_0"}
    %C_out_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_out_0_cons_lock_0"}
    %C_in_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, sym_name = "C_in_0_cons_buff_0"} : memref<64x48xf32> 
    %C_in_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 45056 : i32, sym_name = "C_in_0_cons_buff_1"} : memref<64x48xf32> 
    %C_in_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "C_in_0_cons_prod_lock_0"}
    %C_in_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "C_in_0_cons_cons_lock_0"}
    %C_out_1_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 81920 : i32, sym_name = "C_out_1_cons_buff_0"} : memref<64x48xf32> 
    %C_out_1_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 94208 : i32, sym_name = "C_out_1_cons_buff_1"} : memref<64x48xf32> 
    %C_out_1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "C_out_1_cons_prod_lock_0"}
    %C_out_1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "C_out_1_cons_cons_lock_0"}
    %B_fwd_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 44288 : i32, sym_name = "B_fwd_0_cons_buff_0"} : memref<64x48xbf16> 
    %B_fwd_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 50432 : i32, sym_name = "B_fwd_0_cons_buff_1"} : memref<64x48xbf16> 
    %B_fwd_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_fwd_0_cons_prod_lock_0"}
    %B_fwd_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_fwd_0_cons_cons_lock_0"}
    %B_fwd_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 44288 : i32, sym_name = "B_fwd_1_cons_buff_0"} : memref<64x48xbf16> 
    %B_fwd_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 50432 : i32, sym_name = "B_fwd_1_cons_buff_1"} : memref<64x48xbf16> 
    %B_fwd_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "B_fwd_1_cons_prod_lock_0"}
    %B_fwd_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "B_fwd_1_cons_cons_lock_0"}
    %C_out_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 106496 : i32, sym_name = "C_out_0_cons_buff_0"} : memref<64x48xf32> 
    %C_out_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 118784 : i32, sym_name = "C_out_0_cons_buff_1"} : memref<64x48xf32> 
    %C_out_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "C_out_0_cons_prod_lock_0"}
    %C_out_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "C_out_0_cons_cons_lock_0"}
    %B_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_prod_lock_0"}
    %B_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_cons_lock_0"}
    %A_row1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 27904 : i32, sym_name = "A_row1_cons_buff_0"} : memref<64x64xbf16> 
    %A_row1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 36096 : i32, sym_name = "A_row1_cons_buff_1"} : memref<64x64xbf16> 
    %A_row1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "A_row1_cons_prod_lock_0"}
    %A_row1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "A_row1_cons_cons_lock_0"}
    %A_row0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 27904 : i32, sym_name = "A_row0_cons_buff_0"} : memref<64x64xbf16> 
    %A_row0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 36096 : i32, sym_name = "A_row0_cons_buff_1"} : memref<64x64xbf16> 
    %A_row0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_row0_cons_prod_lock_0"}
    %A_row0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_row0_cons_cons_lock_0"}
    %A_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 49152 : i32, sym_name = "A_L3L2_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, sym_name = "A_L3L2_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "A_L3L2_cons_prod_lock_0"}
    %A_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "A_L3L2_cons_cons_lock_0"}
    %A_L3L2_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "A_L3L2_cons_prod_lock_1"}
    %A_L3L2_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "A_L3L2_cons_cons_lock_1"}
    %Y_mem_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "Y_mem_buff_0"} : memref<6144xf32> 
    %Y_mem_buff_1 = aie.buffer(%mem_tile_0_1) {address = 24576 : i32, sym_name = "Y_mem_buff_1"} : memref<6144xf32> 
    %Y_mem_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "Y_mem_prod_lock_0"}
    %Y_mem_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "Y_mem_cons_lock_0"}
    %Y_mem_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "Y_mem_prod_lock_1"}
    %Y_mem_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "Y_mem_cons_lock_1"}
    %A_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_prod_lock_0"}
    %A_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 4, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%mem_tile_0_1, DMA : 5, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_0_1, DMA : 5)
    func.func private @zero_f32(memref<3072xf32>) attributes {link_with = "matmul_bf16_f32_c894e098.o"}
    func.func private @c894e098_matmul_bf16_f32(memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) attributes {link_with = "matmul_bf16_f32_c894e098.o"}
    func.func private @gelu_epilogue_3072_f32_io(memref<3072xf32>, memref<3072xf32>) attributes {link_with = "gelu_epilogue_3072_f32_io.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 56576 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_out_0_prod_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i64
      cf.switch %4 : i64, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%C_out_0_buff_0 : memref<64x48xf32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%C_out_0_buff_1 : memref<64x48xf32>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%C_out_0_buff_0 : memref<64x48xf32>)
    ^bb6(%5: memref<64x48xf32>):  // 3 preds: ^bb3, ^bb4, ^bb5
      %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%A_row0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i64
      cf.switch %8 : i64, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_row0_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_row0_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_row0_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%9: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_fwd_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_fwd_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_fwd_0_cons_buff_1 : memref<64x48xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_fwd_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb14(%13: memref<64x48xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      %collapse_shape_0 = memref.collapse_shape %9 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %13 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_row0_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_fwd_0_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%C_out_0_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous0[%c0] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o"], stack_size = 3328 : i32}
    %_anonymous1 = aie.buffer(%tile_0_3) {address = 56576 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_out_1_prod_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i64
      cf.switch %4 : i64, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%C_out_1_buff_0 : memref<64x48xf32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%C_out_1_buff_1 : memref<64x48xf32>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%C_out_1_buff_0 : memref<64x48xf32>)
    ^bb6(%5: memref<64x48xf32>):  // 3 preds: ^bb3, ^bb4, ^bb5
      %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%A_row1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i64
      cf.switch %8 : i64, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_row1_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_row1_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_row1_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%9: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_fwd_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_fwd_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_fwd_1_cons_buff_1 : memref<64x48xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_fwd_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb14(%13: memref<64x48xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      %collapse_shape_0 = memref.collapse_shape %9 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %13 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_row1_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%B_fwd_1_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%C_out_1_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous1[%c0] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o"], stack_size = 3328 : i32}
    %_anonymous2 = aie.buffer(%tile_0_4) {address = 57344 : i32, sym_name = "_anonymous2"} : memref<2xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<2xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<2xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb10
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb11
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_in_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous2[%c0] : memref<2xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i64
      cf.switch %4 : i64, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%C_in_0_cons_buff_0 : memref<64x48xf32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%C_in_0_cons_buff_1 : memref<64x48xf32>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%C_in_0_cons_buff_0 : memref<64x48xf32>)
    ^bb6(%5: memref<64x48xf32>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%Y_row0_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous2[%c1] : memref<2xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i64
      cf.switch %8 : i64, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Y_row0_buff_0 : memref<3072xf32>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Y_row0_buff_1 : memref<3072xf32>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Y_row0_buff_0 : memref<3072xf32>)
    ^bb10(%9: memref<3072xf32>):  // 3 preds: ^bb7, ^bb8, ^bb9
      %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @gelu_epilogue_3072_f32_io(%collapse_shape, %9) : (memref<3072xf32>, memref<3072xf32>) -> ()
      aie.use_lock(%C_in_0_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous2[%c0] : memref<2xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c2_i32 : i32
      %13 = arith.subi %11, %c2_i32 : i32
      %14 = arith.select %12, %13, %11 : i32
      memref.store %14, %_anonymous2[%c0] : memref<2xi32>
      aie.use_lock(%Y_row0_cons_lock_0, Release, 1)
      %15 = memref.load %_anonymous2[%c1] : memref<2xi32>
      %16 = arith.addi %15, %c1_i32 : i32
      %17 = arith.cmpi sge, %16, %c2_i32 : i32
      %18 = arith.subi %16, %c2_i32 : i32
      %19 = arith.select %17, %18, %16 : i32
      memref.store %19, %_anonymous2[%c1] : memref<2xi32>
      %20 = arith.addi %0, %c1 : index
      cf.br ^bb1(%20 : index)
    ^bb11:  // pred: ^bb1
      aie.end
    } {link_files = ["gelu_epilogue_3072_f32_io.o"], stack_size = 8192 : i32}
    %_anonymous3 = aie.buffer(%tile_0_5) {address = 57344 : i32, sym_name = "_anonymous3"} : memref<2xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<2xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<2xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb10
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb11
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_in_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous3[%c0] : memref<2xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i64
      cf.switch %4 : i64, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%C_in_1_cons_buff_0 : memref<64x48xf32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%C_in_1_cons_buff_1 : memref<64x48xf32>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%C_in_1_cons_buff_0 : memref<64x48xf32>)
    ^bb6(%5: memref<64x48xf32>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%Y_row1_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous3[%c1] : memref<2xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i64
      cf.switch %8 : i64, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Y_row1_buff_0 : memref<3072xf32>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Y_row1_buff_1 : memref<3072xf32>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Y_row1_buff_0 : memref<3072xf32>)
    ^bb10(%9: memref<3072xf32>):  // 3 preds: ^bb7, ^bb8, ^bb9
      %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @gelu_epilogue_3072_f32_io(%collapse_shape, %9) : (memref<3072xf32>, memref<3072xf32>) -> ()
      aie.use_lock(%C_in_1_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous3[%c0] : memref<2xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c2_i32 : i32
      %13 = arith.subi %11, %c2_i32 : i32
      %14 = arith.select %12, %13, %11 : i32
      memref.store %14, %_anonymous3[%c0] : memref<2xi32>
      aie.use_lock(%Y_row1_cons_lock_0, Release, 1)
      %15 = memref.load %_anonymous3[%c1] : memref<2xi32>
      %16 = arith.addi %15, %c1_i32 : i32
      %17 = arith.cmpi sge, %16, %c2_i32 : i32
      %18 = arith.subi %16, %c2_i32 : i32
      %19 = arith.select %17, %18, %16 : i32
      memref.store %19, %_anonymous3[%c1] : memref<2xi32>
      %20 = arith.addi %0, %c1 : index
      cf.br ^bb1(%20 : index)
    ^bb11:  // pred: ^bb1
      aie.end
    } {link_files = ["gelu_epilogue_3072_f32_io.o"], stack_size = 8192 : i32}
    aie.trace.config @trace_core_1_config(%tile_0_4) packet_type = core {
      aie.trace.reg register = "Trace_Control0" value = 2038038528 mask = 2139029507 comment = "trace mode + start event + stop event"
      aie.trace.reg register = "Trace_Control1" value = 1 mask = 28703 comment = "packet ID + packet type"
      aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" value = 289 mask = 16191 comment = "port 0 ID + port 0 master/slave + port 1 ID + port 1 master/slave"
      aie.trace.reg register = "Trace_Event0" value = 388309537 mask = 2139062143 comment = "INSTR_EVENT_0 + INSTR_EVENT_1 + INSTR_VECTOR + MEMORY_STALL"
      aie.trace.reg register = "Trace_Event1" value = 1330321944 mask = 2139062143 comment = "STREAM_STALL + LOCK_STALL + PORT_RUNNING_0 + PORT_RUNNING_1"
    }
    aie.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<3072xbf16>, %arg2: memref<6144xf32>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 4 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 4 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 4 : i32, value = 289 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 4 : i32, value = 388309537 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 4 : i32, value = 1330321944 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 4 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 65280 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      %0 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 64>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<3072xbf16>, 0, 3072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @Y_mem_shim_alloc {
        aie.dma_bd(%arg2 : memref<6144xf32>, 0, 6144, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_4, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.shim_dma_allocation @A_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<8192xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<8192xbf16>, 4096, 4096) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<8192xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%A_L3L2_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<8192xbf16>, 4096, 4096) {bd_id = 3 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<8192xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<8192xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      %2 = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
    ^bb9:  // 2 preds: ^bb8, ^bb10
      aie.use_lock(%A_L3L2_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<8192xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L3L2_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<8192xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb11:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb12, ^bb14)
    ^bb12:  // 2 preds: ^bb11, ^bb13
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb14:  // pred: ^bb11
      %4 = aie.dma_start(MM2S, 2, ^bb15, ^bb17)
    ^bb15:  // 2 preds: ^bb14, ^bb16
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb17:  // pred: ^bb14
      %5 = aie.dma_start(MM2S, 3, ^bb18, ^bb20)
    ^bb18:  // 2 preds: ^bb17, ^bb19
      aie.use_lock(%C_out_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_0_cons_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_out_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%C_out_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_0_cons_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_out_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb18
    ^bb20:  // pred: ^bb17
      %6 = aie.dma_start(S2MM, 2, ^bb21, ^bb23)
    ^bb21:  // 2 preds: ^bb20, ^bb22
      aie.use_lock(%C_out_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_0_cons_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_out_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%C_out_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_0_cons_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_out_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb21
    ^bb23:  // pred: ^bb20
      %7 = aie.dma_start(MM2S, 4, ^bb24, ^bb26)
    ^bb24:  // 2 preds: ^bb23, ^bb25
      aie.use_lock(%C_out_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_1_cons_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_out_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%C_out_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_1_cons_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_out_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb26:  // pred: ^bb23
      %8 = aie.dma_start(S2MM, 3, ^bb27, ^bb29)
    ^bb27:  // 2 preds: ^bb26, ^bb28
      aie.use_lock(%C_out_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_1_cons_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_out_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_out_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_1_cons_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_out_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb27
    ^bb29:  // pred: ^bb26
      %9 = aie.dma_start(MM2S, 5, ^bb30, ^bb34)
    ^bb30:  // 2 preds: ^bb29, ^bb33
      aie.use_lock(%Y_mem_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_0 : memref<6144xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%Y_mem_prod_lock_0, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%Y_mem_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_0 : memref<6144xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%Y_mem_prod_lock_1, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%Y_mem_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_1 : memref<6144xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%Y_mem_prod_lock_0, Release, 1)
      aie.next_bd ^bb33
    ^bb33:  // pred: ^bb32
      aie.use_lock(%Y_mem_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_1 : memref<6144xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%Y_mem_prod_lock_1, Release, 1)
      aie.next_bd ^bb30
    ^bb34:  // pred: ^bb29
      %10 = aie.dma_start(S2MM, 4, ^bb35, ^bb37)
    ^bb35:  // 2 preds: ^bb34, ^bb36
      aie.use_lock(%Y_mem_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_0 : memref<6144xf32>, 0, 3072) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%Y_mem_cons_lock_0, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%Y_mem_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_1 : memref<6144xf32>, 0, 3072) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%Y_mem_cons_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb37:  // pred: ^bb34
      %11 = aie.dma_start(S2MM, 5, ^bb38, ^bb40)
    ^bb38:  // 2 preds: ^bb37, ^bb39
      aie.use_lock(%Y_mem_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_0 : memref<6144xf32>, 3072, 3072) {bd_id = 36 : i32, next_bd_id = 37 : i32}
      aie.use_lock(%Y_mem_cons_lock_1, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%Y_mem_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_mem_buff_1 : memref<6144xf32>, 3072, 3072) {bd_id = 37 : i32, next_bd_id = 36 : i32}
      aie.use_lock(%Y_mem_cons_lock_1, Release, 1)
      aie.next_bd ^bb38
    ^bb40:  // pred: ^bb37
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_row0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_row0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_row0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_row0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_row0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_row0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_fwd_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_fwd_0_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_fwd_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_fwd_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_fwd_0_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_fwd_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_0_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_0_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_row1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_row1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_row1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_row1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_row1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_row1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_fwd_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_fwd_1_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_fwd_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_fwd_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_fwd_1_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_fwd_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_1_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_out_1_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%C_in_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_in_0_cons_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%C_in_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_in_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_in_0_cons_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%C_in_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Y_row0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_row0_buff_0 : memref<3072xf32>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Y_row0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Y_row0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_row0_buff_1 : memref<3072xf32>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Y_row0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%C_in_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_in_1_cons_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%C_in_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_in_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_in_1_cons_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%C_in_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Y_row1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_row1_buff_0 : memref<3072xf32>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Y_row1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Y_row1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Y_row1_buff_1 : memref<3072xf32>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Y_row1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @Y_mem_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
