module {
  aie.device(npu2) {
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %C_L2L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "C_L2L3_3_cons_prod_lock_0"}
    %C_L2L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "C_L2L3_3_cons_cons_lock_0"}
    %B_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 114688 : i32, sym_name = "B_L3L2_3_cons_buff_0"} : memref<3072xbf16> 
    %B_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 120832 : i32, sym_name = "B_L3L2_3_cons_buff_1"} : memref<3072xbf16> 
    %B_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 10) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_0"}
    %B_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 11) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_0"}
    %C_L1L2_3_3_buff_0 = aie.buffer(%tile_3_5) {address = 8192 : i32, sym_name = "C_L1L2_3_3_buff_0"} : memref<64x48xf32> 
    %C_L1L2_3_3_buff_1 = aie.buffer(%tile_3_5) {address = 20480 : i32, sym_name = "C_L1L2_3_3_buff_1"} : memref<64x48xf32> 
    %C_L1L2_3_3_prod_lock_0 = aie.lock(%tile_3_5, 4) {init = 2 : i32, sym_name = "C_L1L2_3_3_prod_lock_0"}
    %C_L1L2_3_3_cons_lock_0 = aie.lock(%tile_3_5, 5) {init = 0 : i32, sym_name = "C_L1L2_3_3_cons_lock_0"}
    %B_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 114688 : i32, sym_name = "B_L3L2_2_cons_buff_0"} : memref<3072xbf16> 
    %B_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 120832 : i32, sym_name = "B_L3L2_2_cons_buff_1"} : memref<3072xbf16> 
    %B_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 10) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_0"}
    %B_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 11) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_0"}
    %C_L1L2_3_2_buff_0 = aie.buffer(%tile_2_5) {address = 8192 : i32, sym_name = "C_L1L2_3_2_buff_0"} : memref<64x48xf32> 
    %C_L1L2_3_2_buff_1 = aie.buffer(%tile_2_5) {address = 20480 : i32, sym_name = "C_L1L2_3_2_buff_1"} : memref<64x48xf32> 
    %C_L1L2_3_2_prod_lock_0 = aie.lock(%tile_2_5, 4) {init = 2 : i32, sym_name = "C_L1L2_3_2_prod_lock_0"}
    %C_L1L2_3_2_cons_lock_0 = aie.lock(%tile_2_5, 5) {init = 0 : i32, sym_name = "C_L1L2_3_2_cons_lock_0"}
    %C_L1L2_3_1_buff_0 = aie.buffer(%tile_1_5) {address = 8192 : i32, sym_name = "C_L1L2_3_1_buff_0"} : memref<64x48xf32> 
    %C_L1L2_3_1_buff_1 = aie.buffer(%tile_1_5) {address = 20480 : i32, sym_name = "C_L1L2_3_1_buff_1"} : memref<64x48xf32> 
    %C_L1L2_3_1_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 2 : i32, sym_name = "C_L1L2_3_1_prod_lock_0"}
    %C_L1L2_3_1_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "C_L1L2_3_1_cons_lock_0"}
    %B_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 114688 : i32, sym_name = "B_L3L2_1_cons_buff_0"} : memref<3072xbf16> 
    %B_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 120832 : i32, sym_name = "B_L3L2_1_cons_buff_1"} : memref<3072xbf16> 
    %B_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_0"}
    %B_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_0"}
    %C_L1L2_3_0_buff_0 = aie.buffer(%tile_0_5) {address = 8192 : i32, sym_name = "C_L1L2_3_0_buff_0"} : memref<64x48xf32> 
    %C_L1L2_3_0_buff_1 = aie.buffer(%tile_0_5) {address = 20480 : i32, sym_name = "C_L1L2_3_0_buff_1"} : memref<64x48xf32> 
    %C_L1L2_3_0_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "C_L1L2_3_0_prod_lock_0"}
    %C_L1L2_3_0_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_lock_0"}
    %C_L2L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_2_cons_prod_lock_0"}
    %C_L2L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_2_cons_cons_lock_0"}
    %B_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 114688 : i32, sym_name = "B_L3L2_0_cons_buff_0"} : memref<3072xbf16> 
    %B_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 120832 : i32, sym_name = "B_L3L2_0_cons_buff_1"} : memref<3072xbf16> 
    %B_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 10) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_0"}
    %B_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 11) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_0"}
    %C_L1L2_2_3_buff_0 = aie.buffer(%tile_3_4) {address = 8192 : i32, sym_name = "C_L1L2_2_3_buff_0"} : memref<64x48xf32> 
    %C_L1L2_2_3_buff_1 = aie.buffer(%tile_3_4) {address = 20480 : i32, sym_name = "C_L1L2_2_3_buff_1"} : memref<64x48xf32> 
    %C_L1L2_2_3_prod_lock_0 = aie.lock(%tile_3_4, 4) {init = 2 : i32, sym_name = "C_L1L2_2_3_prod_lock_0"}
    %C_L1L2_2_3_cons_lock_0 = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "C_L1L2_2_3_cons_lock_0"}
    %C_L1L2_2_2_buff_0 = aie.buffer(%tile_2_4) {address = 8192 : i32, sym_name = "C_L1L2_2_2_buff_0"} : memref<64x48xf32> 
    %C_L1L2_2_2_buff_1 = aie.buffer(%tile_2_4) {address = 20480 : i32, sym_name = "C_L1L2_2_2_buff_1"} : memref<64x48xf32> 
    %C_L1L2_2_2_prod_lock_0 = aie.lock(%tile_2_4, 4) {init = 2 : i32, sym_name = "C_L1L2_2_2_prod_lock_0"}
    %C_L1L2_2_2_cons_lock_0 = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "C_L1L2_2_2_cons_lock_0"}
    %A_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 98304 : i32, sym_name = "A_L3L2_3_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 106496 : i32, sym_name = "A_L3L2_3_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 8) {init = 2 : i32, sym_name = "A_L3L2_3_cons_prod_lock_0"}
    %A_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 9) {init = 0 : i32, sym_name = "A_L3L2_3_cons_cons_lock_0"}
    %C_L1L2_2_1_buff_0 = aie.buffer(%tile_1_4) {address = 8192 : i32, sym_name = "C_L1L2_2_1_buff_0"} : memref<64x48xf32> 
    %C_L1L2_2_1_buff_1 = aie.buffer(%tile_1_4) {address = 20480 : i32, sym_name = "C_L1L2_2_1_buff_1"} : memref<64x48xf32> 
    %C_L1L2_2_1_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 2 : i32, sym_name = "C_L1L2_2_1_prod_lock_0"}
    %C_L1L2_2_1_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "C_L1L2_2_1_cons_lock_0"}
    %C_L1L2_2_0_buff_0 = aie.buffer(%tile_0_4) {address = 8192 : i32, sym_name = "C_L1L2_2_0_buff_0"} : memref<64x48xf32> 
    %C_L1L2_2_0_buff_1 = aie.buffer(%tile_0_4) {address = 20480 : i32, sym_name = "C_L1L2_2_0_buff_1"} : memref<64x48xf32> 
    %C_L1L2_2_0_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "C_L1L2_2_0_prod_lock_0"}
    %C_L1L2_2_0_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_lock_0"}
    %C_L2L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "C_L2L3_1_cons_prod_lock_0"}
    %C_L2L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "C_L2L3_1_cons_cons_lock_0"}
    %A_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 98304 : i32, sym_name = "A_L3L2_2_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 106496 : i32, sym_name = "A_L3L2_2_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 8) {init = 2 : i32, sym_name = "A_L3L2_2_cons_prod_lock_0"}
    %A_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 9) {init = 0 : i32, sym_name = "A_L3L2_2_cons_cons_lock_0"}
    %C_L1L2_1_3_buff_0 = aie.buffer(%tile_3_3) {address = 8192 : i32, sym_name = "C_L1L2_1_3_buff_0"} : memref<64x48xf32> 
    %C_L1L2_1_3_buff_1 = aie.buffer(%tile_3_3) {address = 20480 : i32, sym_name = "C_L1L2_1_3_buff_1"} : memref<64x48xf32> 
    %C_L1L2_1_3_prod_lock_0 = aie.lock(%tile_3_3, 4) {init = 2 : i32, sym_name = "C_L1L2_1_3_prod_lock_0"}
    %C_L1L2_1_3_cons_lock_0 = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "C_L1L2_1_3_cons_lock_0"}
    %A_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 98304 : i32, sym_name = "A_L3L2_1_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 106496 : i32, sym_name = "A_L3L2_1_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 8) {init = 2 : i32, sym_name = "A_L3L2_1_cons_prod_lock_0"}
    %A_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 9) {init = 0 : i32, sym_name = "A_L3L2_1_cons_cons_lock_0"}
    %C_L1L2_1_2_buff_0 = aie.buffer(%tile_2_3) {address = 8192 : i32, sym_name = "C_L1L2_1_2_buff_0"} : memref<64x48xf32> 
    %C_L1L2_1_2_buff_1 = aie.buffer(%tile_2_3) {address = 20480 : i32, sym_name = "C_L1L2_1_2_buff_1"} : memref<64x48xf32> 
    %C_L1L2_1_2_prod_lock_0 = aie.lock(%tile_2_3, 4) {init = 2 : i32, sym_name = "C_L1L2_1_2_prod_lock_0"}
    %C_L1L2_1_2_cons_lock_0 = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "C_L1L2_1_2_cons_lock_0"}
    %C_L1L2_1_1_buff_0 = aie.buffer(%tile_1_3) {address = 8192 : i32, sym_name = "C_L1L2_1_1_buff_0"} : memref<64x48xf32> 
    %C_L1L2_1_1_buff_1 = aie.buffer(%tile_1_3) {address = 20480 : i32, sym_name = "C_L1L2_1_1_buff_1"} : memref<64x48xf32> 
    %C_L1L2_1_1_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 2 : i32, sym_name = "C_L1L2_1_1_prod_lock_0"}
    %C_L1L2_1_1_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "C_L1L2_1_1_cons_lock_0"}
    %A_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 98304 : i32, sym_name = "A_L3L2_0_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 106496 : i32, sym_name = "A_L3L2_0_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "A_L3L2_0_cons_prod_lock_0"}
    %A_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "A_L3L2_0_cons_cons_lock_0"}
    %C_L1L2_1_0_buff_0 = aie.buffer(%tile_0_3) {address = 8192 : i32, sym_name = "C_L1L2_1_0_buff_0"} : memref<64x48xf32> 
    %C_L1L2_1_0_buff_1 = aie.buffer(%tile_0_3) {address = 20480 : i32, sym_name = "C_L1L2_1_0_buff_1"} : memref<64x48xf32> 
    %C_L1L2_1_0_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "C_L1L2_1_0_prod_lock_0"}
    %C_L1L2_1_0_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_lock_0"}
    %C_L2L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "C_L2L3_0_cons_prod_lock_0"}
    %C_L2L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_cons_lock_0"}
    %C_L1L2_0_3_buff_0 = aie.buffer(%tile_3_2) {address = 8192 : i32, sym_name = "C_L1L2_0_3_buff_0"} : memref<64x48xf32> 
    %C_L1L2_0_3_buff_1 = aie.buffer(%tile_3_2) {address = 20480 : i32, sym_name = "C_L1L2_0_3_buff_1"} : memref<64x48xf32> 
    %C_L1L2_0_3_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 2 : i32, sym_name = "C_L1L2_0_3_prod_lock_0"}
    %C_L1L2_0_3_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "C_L1L2_0_3_cons_lock_0"}
    %C_L1L2_0_2_buff_0 = aie.buffer(%tile_2_2) {address = 8192 : i32, sym_name = "C_L1L2_0_2_buff_0"} : memref<64x48xf32> 
    %C_L1L2_0_2_buff_1 = aie.buffer(%tile_2_2) {address = 20480 : i32, sym_name = "C_L1L2_0_2_buff_1"} : memref<64x48xf32> 
    %C_L1L2_0_2_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 2 : i32, sym_name = "C_L1L2_0_2_prod_lock_0"}
    %C_L1L2_0_2_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "C_L1L2_0_2_cons_lock_0"}
    %C_L1L2_0_1_buff_0 = aie.buffer(%tile_1_2) {address = 8192 : i32, sym_name = "C_L1L2_0_1_buff_0"} : memref<64x48xf32> 
    %C_L1L2_0_1_buff_1 = aie.buffer(%tile_1_2) {address = 20480 : i32, sym_name = "C_L1L2_0_1_buff_1"} : memref<64x48xf32> 
    %C_L1L2_0_1_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "C_L1L2_0_1_prod_lock_0"}
    %C_L1L2_0_1_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "C_L1L2_0_1_cons_lock_0"}
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 8192 : i32, sym_name = "C_L1L2_0_0_buff_0"} : memref<64x48xf32> 
    %C_L1L2_0_0_buff_1 = aie.buffer(%tile_0_2) {address = 20480 : i32, sym_name = "C_L1L2_0_0_buff_1"} : memref<64x48xf32> 
    %C_L1L2_0_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "C_L1L2_0_0_prod_lock_0"}
    %C_L1L2_0_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_lock_0"}
    %B_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "B_L3L2_3_prod_lock_0"}
    %B_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "B_L3L2_3_cons_lock_0"}
    %B_L2L1_3_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 49152 : i32, sym_name = "B_L2L1_3_0_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_3_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 55296 : i32, sym_name = "B_L2L1_3_0_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_3_0_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "B_L2L1_3_0_cons_prod_lock_0"}
    %B_L2L1_3_0_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "B_L2L1_3_0_cons_cons_lock_0"}
    %B_L2L1_3_1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 49152 : i32, sym_name = "B_L2L1_3_1_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_3_1_cons_buff_1 = aie.buffer(%tile_1_5) {address = 55296 : i32, sym_name = "B_L2L1_3_1_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_3_1_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "B_L2L1_3_1_cons_prod_lock_0"}
    %B_L2L1_3_1_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "B_L2L1_3_1_cons_cons_lock_0"}
    %B_L2L1_3_2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 49152 : i32, sym_name = "B_L2L1_3_2_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_3_2_cons_buff_1 = aie.buffer(%tile_2_5) {address = 55296 : i32, sym_name = "B_L2L1_3_2_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_3_2_cons_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 2 : i32, sym_name = "B_L2L1_3_2_cons_prod_lock_0"}
    %B_L2L1_3_2_cons_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "B_L2L1_3_2_cons_cons_lock_0"}
    %B_L2L1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 49152 : i32, sym_name = "B_L2L1_3_3_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 55296 : i32, sym_name = "B_L2L1_3_3_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 2) {init = 2 : i32, sym_name = "B_L2L1_3_3_cons_prod_lock_0"}
    %B_L2L1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "B_L2L1_3_3_cons_cons_lock_0"}
    %B_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "B_L3L2_2_prod_lock_0"}
    %B_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "B_L3L2_2_cons_lock_0"}
    %B_L2L1_2_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 49152 : i32, sym_name = "B_L2L1_2_0_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_2_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 55296 : i32, sym_name = "B_L2L1_2_0_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_2_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "B_L2L1_2_0_cons_prod_lock_0"}
    %B_L2L1_2_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "B_L2L1_2_0_cons_cons_lock_0"}
    %B_L2L1_2_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 49152 : i32, sym_name = "B_L2L1_2_1_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_2_1_cons_buff_1 = aie.buffer(%tile_1_4) {address = 55296 : i32, sym_name = "B_L2L1_2_1_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_2_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "B_L2L1_2_1_cons_prod_lock_0"}
    %B_L2L1_2_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "B_L2L1_2_1_cons_cons_lock_0"}
    %B_L2L1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 49152 : i32, sym_name = "B_L2L1_2_2_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 55296 : i32, sym_name = "B_L2L1_2_2_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 2 : i32, sym_name = "B_L2L1_2_2_cons_prod_lock_0"}
    %B_L2L1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "B_L2L1_2_2_cons_cons_lock_0"}
    %B_L2L1_2_3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 49152 : i32, sym_name = "B_L2L1_2_3_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_2_3_cons_buff_1 = aie.buffer(%tile_3_4) {address = 55296 : i32, sym_name = "B_L2L1_2_3_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_2_3_cons_prod_lock_0 = aie.lock(%tile_3_4, 2) {init = 2 : i32, sym_name = "B_L2L1_2_3_cons_prod_lock_0"}
    %B_L2L1_2_3_cons_cons_lock_0 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "B_L2L1_2_3_cons_cons_lock_0"}
    %B_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_1_prod_lock_0"}
    %B_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_1_cons_lock_0"}
    %B_L2L1_1_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 49152 : i32, sym_name = "B_L2L1_1_0_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_1_0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 55296 : i32, sym_name = "B_L2L1_1_0_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_1_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "B_L2L1_1_0_cons_prod_lock_0"}
    %B_L2L1_1_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "B_L2L1_1_0_cons_cons_lock_0"}
    %B_L2L1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 49152 : i32, sym_name = "B_L2L1_1_1_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 55296 : i32, sym_name = "B_L2L1_1_1_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "B_L2L1_1_1_cons_prod_lock_0"}
    %B_L2L1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "B_L2L1_1_1_cons_cons_lock_0"}
    %B_L2L1_1_2_cons_buff_0 = aie.buffer(%tile_2_3) {address = 49152 : i32, sym_name = "B_L2L1_1_2_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_1_2_cons_buff_1 = aie.buffer(%tile_2_3) {address = 55296 : i32, sym_name = "B_L2L1_1_2_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_1_2_cons_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "B_L2L1_1_2_cons_prod_lock_0"}
    %B_L2L1_1_2_cons_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "B_L2L1_1_2_cons_cons_lock_0"}
    %B_L2L1_1_3_cons_buff_0 = aie.buffer(%tile_3_3) {address = 49152 : i32, sym_name = "B_L2L1_1_3_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_1_3_cons_buff_1 = aie.buffer(%tile_3_3) {address = 55296 : i32, sym_name = "B_L2L1_1_3_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_1_3_cons_prod_lock_0 = aie.lock(%tile_3_3, 2) {init = 2 : i32, sym_name = "B_L2L1_1_3_cons_prod_lock_0"}
    %B_L2L1_1_3_cons_cons_lock_0 = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "B_L2L1_1_3_cons_cons_lock_0"}
    %B_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "B_L3L2_0_prod_lock_0"}
    %B_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_lock_0"}
    %B_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 49152 : i32, sym_name = "B_L2L1_0_0_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 55296 : i32, sym_name = "B_L2L1_0_0_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_L2L1_0_0_cons_prod_lock_0"}
    %B_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L2L1_0_0_cons_cons_lock_0"}
    %B_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 49152 : i32, sym_name = "B_L2L1_0_1_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 55296 : i32, sym_name = "B_L2L1_0_1_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "B_L2L1_0_1_cons_prod_lock_0"}
    %B_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "B_L2L1_0_1_cons_cons_lock_0"}
    %B_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 49152 : i32, sym_name = "B_L2L1_0_2_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 55296 : i32, sym_name = "B_L2L1_0_2_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "B_L2L1_0_2_cons_prod_lock_0"}
    %B_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "B_L2L1_0_2_cons_cons_lock_0"}
    %B_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 49152 : i32, sym_name = "B_L2L1_0_3_cons_buff_0"} : memref<64x48xbf16> 
    %B_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 55296 : i32, sym_name = "B_L2L1_0_3_cons_buff_1"} : memref<64x48xbf16> 
    %B_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "B_L2L1_0_3_cons_prod_lock_0"}
    %B_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "B_L2L1_0_3_cons_cons_lock_0"}
    %A_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "A_L3L2_3_prod_lock_0"}
    %A_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "A_L3L2_3_cons_lock_0"}
    %A_L2L1_3_0_cons_buff_0 = aie.buffer(%tile_3_2) {address = 32768 : i32, sym_name = "A_L2L1_3_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_0_cons_buff_1 = aie.buffer(%tile_3_2) {address = 40960 : i32, sym_name = "A_L2L1_3_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_0_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "A_L2L1_3_0_cons_prod_lock_0"}
    %A_L2L1_3_0_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "A_L2L1_3_0_cons_cons_lock_0"}
    %A_L2L1_3_1_cons_buff_0 = aie.buffer(%tile_3_3) {address = 32768 : i32, sym_name = "A_L2L1_3_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_1_cons_buff_1 = aie.buffer(%tile_3_3) {address = 40960 : i32, sym_name = "A_L2L1_3_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_1_cons_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "A_L2L1_3_1_cons_prod_lock_0"}
    %A_L2L1_3_1_cons_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "A_L2L1_3_1_cons_cons_lock_0"}
    %A_L2L1_3_2_cons_buff_0 = aie.buffer(%tile_3_4) {address = 32768 : i32, sym_name = "A_L2L1_3_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_2_cons_buff_1 = aie.buffer(%tile_3_4) {address = 40960 : i32, sym_name = "A_L2L1_3_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_2_cons_prod_lock_0 = aie.lock(%tile_3_4, 0) {init = 2 : i32, sym_name = "A_L2L1_3_2_cons_prod_lock_0"}
    %A_L2L1_3_2_cons_cons_lock_0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "A_L2L1_3_2_cons_cons_lock_0"}
    %A_L2L1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 32768 : i32, sym_name = "A_L2L1_3_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 40960 : i32, sym_name = "A_L2L1_3_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_3_cons_prod_lock_0"}
    %A_L2L1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_3_cons_cons_lock_0"}
    %A_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "A_L3L2_2_prod_lock_0"}
    %A_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "A_L3L2_2_cons_lock_0"}
    %A_L2L1_2_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 32768 : i32, sym_name = "A_L2L1_2_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_0_cons_buff_1 = aie.buffer(%tile_2_2) {address = 40960 : i32, sym_name = "A_L2L1_2_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "A_L2L1_2_0_cons_prod_lock_0"}
    %A_L2L1_2_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "A_L2L1_2_0_cons_cons_lock_0"}
    %A_L2L1_2_1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 32768 : i32, sym_name = "A_L2L1_2_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_1_cons_buff_1 = aie.buffer(%tile_2_3) {address = 40960 : i32, sym_name = "A_L2L1_2_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_1_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "A_L2L1_2_1_cons_prod_lock_0"}
    %A_L2L1_2_1_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "A_L2L1_2_1_cons_cons_lock_0"}
    %A_L2L1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 32768 : i32, sym_name = "A_L2L1_2_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 40960 : i32, sym_name = "A_L2L1_2_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_2_cons_prod_lock_0"}
    %A_L2L1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_2_cons_cons_lock_0"}
    %A_L2L1_2_3_cons_buff_0 = aie.buffer(%tile_2_5) {address = 32768 : i32, sym_name = "A_L2L1_2_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_3_cons_buff_1 = aie.buffer(%tile_2_5) {address = 40960 : i32, sym_name = "A_L2L1_2_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_3_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 2 : i32, sym_name = "A_L2L1_2_3_cons_prod_lock_0"}
    %A_L2L1_2_3_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "A_L2L1_2_3_cons_cons_lock_0"}
    %C_L2L3_3_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "C_L2L3_3_buff_0"} : memref<12288xf32> 
    %C_L2L3_3_buff_1 = aie.buffer(%mem_tile_3_1) {address = 49152 : i32, sym_name = "C_L2L3_3_buff_1"} : memref<12288xf32> 
    %C_L2L3_3_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_0"}
    %C_L2L3_3_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_0"}
    %C_L2L3_3_prod_lock_1 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_1"}
    %C_L2L3_3_cons_lock_1 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_1"}
    %C_L2L3_3_prod_lock_2 = aie.lock(%mem_tile_3_1, 4) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_2"}
    %C_L2L3_3_cons_lock_2 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_2"}
    %C_L2L3_3_prod_lock_3 = aie.lock(%mem_tile_3_1, 6) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_3"}
    %C_L2L3_3_cons_lock_3 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_3"}
    %A_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "A_L3L2_1_prod_lock_0"}
    %A_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "A_L3L2_1_cons_lock_0"}
    %A_L2L1_1_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, sym_name = "A_L2L1_1_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_0_cons_buff_1 = aie.buffer(%tile_1_2) {address = 40960 : i32, sym_name = "A_L2L1_1_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "A_L2L1_1_0_cons_prod_lock_0"}
    %A_L2L1_1_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "A_L2L1_1_0_cons_cons_lock_0"}
    %A_L2L1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, sym_name = "A_L2L1_1_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 40960 : i32, sym_name = "A_L2L1_1_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_1_cons_prod_lock_0"}
    %A_L2L1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_1_cons_cons_lock_0"}
    %A_L2L1_1_2_cons_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, sym_name = "A_L2L1_1_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_2_cons_buff_1 = aie.buffer(%tile_1_4) {address = 40960 : i32, sym_name = "A_L2L1_1_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_2_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "A_L2L1_1_2_cons_prod_lock_0"}
    %A_L2L1_1_2_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "A_L2L1_1_2_cons_cons_lock_0"}
    %A_L2L1_1_3_cons_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, sym_name = "A_L2L1_1_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_3_cons_buff_1 = aie.buffer(%tile_1_5) {address = 40960 : i32, sym_name = "A_L2L1_1_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_3_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "A_L2L1_1_3_cons_prod_lock_0"}
    %A_L2L1_1_3_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "A_L2L1_1_3_cons_cons_lock_0"}
    %C_L2L3_2_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "C_L2L3_2_buff_0"} : memref<12288xf32> 
    %C_L2L3_2_buff_1 = aie.buffer(%mem_tile_2_1) {address = 49152 : i32, sym_name = "C_L2L3_2_buff_1"} : memref<12288xf32> 
    %C_L2L3_2_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_0"}
    %C_L2L3_2_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_0"}
    %C_L2L3_2_prod_lock_1 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_1"}
    %C_L2L3_2_cons_lock_1 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_1"}
    %C_L2L3_2_prod_lock_2 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_2"}
    %C_L2L3_2_cons_lock_2 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_2"}
    %C_L2L3_2_prod_lock_3 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_3"}
    %C_L2L3_2_cons_lock_3 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_3"}
    %C_L2L3_1_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "C_L2L3_1_buff_0"} : memref<12288xf32> 
    %C_L2L3_1_buff_1 = aie.buffer(%mem_tile_0_1) {address = 49152 : i32, sym_name = "C_L2L3_1_buff_1"} : memref<12288xf32> 
    %C_L2L3_1_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_0"}
    %C_L2L3_1_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_0"}
    %C_L2L3_1_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_1"}
    %C_L2L3_1_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_1"}
    %C_L2L3_1_prod_lock_2 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_2"}
    %C_L2L3_1_cons_lock_2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_2"}
    %C_L2L3_1_prod_lock_3 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_3"}
    %C_L2L3_1_cons_lock_3 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_3"}
    %A_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_0_prod_lock_0"}
    %A_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_0_cons_lock_0"}
    %A_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, sym_name = "A_L2L1_0_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 40960 : i32, sym_name = "A_L2L1_0_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_0_cons_prod_lock_0"}
    %A_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_0_cons_cons_lock_0"}
    %A_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, sym_name = "A_L2L1_0_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 40960 : i32, sym_name = "A_L2L1_0_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "A_L2L1_0_1_cons_prod_lock_0"}
    %A_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "A_L2L1_0_1_cons_cons_lock_0"}
    %A_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, sym_name = "A_L2L1_0_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 40960 : i32, sym_name = "A_L2L1_0_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "A_L2L1_0_2_cons_prod_lock_0"}
    %A_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "A_L2L1_0_2_cons_cons_lock_0"}
    %A_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, sym_name = "A_L2L1_0_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 40960 : i32, sym_name = "A_L2L1_0_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "A_L2L1_0_3_cons_prod_lock_0"}
    %A_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "A_L2L1_0_3_cons_cons_lock_0"}
    %C_L2L3_0_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "C_L2L3_0_buff_0"} : memref<12288xf32> 
    %C_L2L3_0_buff_1 = aie.buffer(%mem_tile_1_1) {address = 49152 : i32, sym_name = "C_L2L3_0_buff_1"} : memref<12288xf32> 
    %C_L2L3_0_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_0"}
    %C_L2L3_0_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_0"}
    %C_L2L3_0_prod_lock_1 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_1"}
    %C_L2L3_0_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_1"}
    %C_L2L3_0_prod_lock_2 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_2"}
    %C_L2L3_0_cons_lock_2 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_2"}
    %C_L2L3_0_prod_lock_3 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_3"}
    %C_L2L3_0_cons_lock_3 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_3"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 3)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_1_1, DMA : 4)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_1_1, DMA : 5)
    aie.flow(%mem_tile_1_1, DMA : 2, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_1_3, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%tile_2_3, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%tile_3_3, DMA : 0, %mem_tile_0_1, DMA : 5)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_1_0, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_1_4, DMA : 0, %mem_tile_2_1, DMA : 3)
    aie.flow(%tile_2_4, DMA : 0, %mem_tile_2_1, DMA : 4)
    aie.flow(%tile_3_4, DMA : 0, %mem_tile_2_1, DMA : 5)
    aie.flow(%mem_tile_2_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.flow(%tile_1_5, DMA : 0, %mem_tile_3_1, DMA : 3)
    aie.flow(%tile_2_5, DMA : 0, %mem_tile_3_1, DMA : 4)
    aie.flow(%tile_3_5, DMA : 0, %mem_tile_3_1, DMA : 5)
    aie.flow(%mem_tile_3_1, DMA : 2, %shim_noc_tile_2_0, DMA : 0)
    func.func private @zero_f32(memref<3072xf32>) attributes {link_with = "matmul_bf16_f32_c894e098.o"}
    func.func private @c894e098_matmul_bf16_f32(memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) attributes {link_with = "matmul_bf16_f32_c894e098.o"}
    func.func private @gelu_epilogue_3072_f32(memref<3072xf32>) attributes {link_with = "gelu_epilogue_3072_f32.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 61440 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_0_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_0_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_0_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_0_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous0[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous0[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous1 = aie.buffer(%tile_0_3) {address = 61440 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_0_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_0_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_0_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_0_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous1[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous1[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous2 = aie.buffer(%tile_0_4) {address = 61440 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_0_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_0_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_0_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_0_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous2[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous2[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous3 = aie.buffer(%tile_0_5) {address = 61440 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_0_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_0_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_0_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_0_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_0_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous3[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous3[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous4 = aie.buffer(%tile_1_2) {address = 61440 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous4[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_1_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_1_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_1_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_1_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous4[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous4[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous5 = aie.buffer(%tile_1_3) {address = 61440 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_1_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_1_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_1_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_1_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous5[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous5[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous6 = aie.buffer(%tile_1_4) {address = 61440 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_1_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_1_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_1_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_1_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous6[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous6[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous7 = aie.buffer(%tile_1_5) {address = 61440 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_1_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_1_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_1_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_1_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_1_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous7[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous7[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous8 = aie.buffer(%tile_2_2) {address = 61440 : i32, sym_name = "_anonymous8"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous8[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_2_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_2_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_2_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_2_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous8[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous8[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous8[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous9 = aie.buffer(%tile_2_3) {address = 61440 : i32, sym_name = "_anonymous9"} : memref<3xi32> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous9[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_2_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_2_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_2_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_2_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous9[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous9[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous9[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous10 = aie.buffer(%tile_2_4) {address = 61440 : i32, sym_name = "_anonymous10"} : memref<3xi32> 
    %core_2_4 = aie.core(%tile_2_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous10[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_2_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_2_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_2_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_2_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous10[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous10[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous10[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous11 = aie.buffer(%tile_2_5) {address = 61440 : i32, sym_name = "_anonymous11"} : memref<3xi32> 
    %core_2_5 = aie.core(%tile_2_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous11[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_2_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_2_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_2_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_2_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_2_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous11[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous11[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous11[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous12 = aie.buffer(%tile_3_2) {address = 61440 : i32, sym_name = "_anonymous12"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous12[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_3_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_3_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_0_3_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_3_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_0_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous12[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous12[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous12[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous13 = aie.buffer(%tile_3_3) {address = 61440 : i32, sym_name = "_anonymous13"} : memref<3xi32> 
    %core_3_3 = aie.core(%tile_3_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous13[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_3_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_3_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_1_3_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_3_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_1_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous13[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous13[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous13[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous14 = aie.buffer(%tile_3_4) {address = 61440 : i32, sym_name = "_anonymous14"} : memref<3xi32> 
    %core_3_4 = aie.core(%tile_3_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous14[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_3_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_3_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_2_3_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_3_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_2_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous14[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous14[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous14[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    %_anonymous15 = aie.buffer(%tile_3_5) {address = 61440 : i32, sym_name = "_anonymous15"} : memref<3xi32> 
    %core_3_5 = aie.core(%tile_3_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous15[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_3_buff_0 : memref<64x48xf32>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_3_buff_1 : memref<64x48xf32>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_3_3_buff_0 : memref<64x48xf32>)
    ^bb8(%7: memref<64x48xf32>):  // 3 preds: ^bb5, ^bb6, ^bb7
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x48xf32> into memref<3072xf32>
      func.call @zero_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c7 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%13: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i64
      cf.switch %16 : i64, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_3_cons_buff_1 : memref<64x48xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%B_L2L1_3_3_cons_buff_0 : memref<64x48xbf16>)
    ^bb18(%17: memref<64x48xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      %collapse_shape_0 = memref.collapse_shape %13 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<64x48xbf16> into memref<3072xbf16>
      func.call @c894e098_matmul_bf16_f32(%collapse_shape_0, %collapse_shape_1, %collapse_shape) : (memref<4096xbf16>, memref<3072xbf16>, memref<3072xf32>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous15[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous15[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      func.call @gelu_epilogue_3072_f32(%collapse_shape) : (memref<3072xf32>) -> ()
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous15[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_f32_c894e098.o", "gelu_epilogue_3072_f32.o"], stack_size = 8192 : i32}
    aie.trace.config @trace_core_1_config(%tile_0_2) packet_type = core {
      aie.trace.reg register = "Trace_Control0" value = 2038038528 mask = 2139029507 comment = "trace mode + start event + stop event"
      aie.trace.reg register = "Trace_Control1" value = 1 mask = 28703 comment = "packet ID + packet type"
      aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" value = 289 mask = 16191 comment = "port 0 ID + port 0 master/slave + port 1 ID + port 1 master/slave"
      aie.trace.reg register = "Trace_Event0" value = 388309537 mask = 2139062143 comment = "INSTR_EVENT_0 + INSTR_EVENT_1 + INSTR_VECTOR + MEMORY_STALL"
      aie.trace.reg register = "Trace_Event1" value = 1330321944 mask = 2139062143 comment = "STREAM_STALL + LOCK_STALL + PORT_RUNNING_0 + PORT_RUNNING_1"
    }
    aie.runtime_sequence(%arg0: memref<458752xbf16>, %arg1: memref<688128xbf16>, %arg2: memref<1572864xf32>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 388309537 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 1330321944 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 65280 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      %0 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 0, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 0, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 0, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 114688, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 0, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 48, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_L3L2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 28672, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 3072, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_L3L2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 143360, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 3072, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 96, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @A_L3L2_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 57344, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 6144, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @A_L3L2_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 172032, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 6144, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 144, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @A_L3L2_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 86016, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 9216, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @A_L3L2_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 200704, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 9216, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 786432, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 229376, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 0, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 344064, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 0, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 786480, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @A_L3L2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 258048, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 3072, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @A_L3L2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 372736, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 3072, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 786528, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @A_L3L2_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 286720, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 6144, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @A_L3L2_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 401408, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 6144, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<1572864xf32>, 786576, 98304, [<size = 2, stride = 393216>, <size = 8, stride = 192>, <size = 256, stride = 1536>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @A_L3L2_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 315392, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 9216, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @A_L3L2_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<458752xbf16>, 430080, 28672, [<size = 8, stride = 0>, <size = 7, stride = 64>, <size = 64, stride = 448>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<688128xbf16>, 9216, 21504, [<size = 8, stride = 12288>, <size = 7, stride = 98304>, <size = 64, stride = 48>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%39)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%35)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      aiex.dma_free_task(%29)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 3072, 3072) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 3072, 3072) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 6144, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 6144, 3072) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 9216, 3072) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 9216, 3072) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 3072, 3072) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 3072, 3072) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 6144, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 6144, 3072) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 9216, 3072) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 9216, 3072) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_1_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_1_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_1_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_1_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_1_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_1_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_1_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_1_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 3072, 3072) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 3072, 3072) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 6144, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 6144, 3072) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 9216, 3072) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 9216, 3072) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_2_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_2_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_2_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_2_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_2_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_2_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_2_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_2_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<3072xbf16>, 0, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<3072xbf16>, 0, 3072) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 3072, 3072) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 3072, 3072) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 6144, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 6144, 3072) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 9216, 3072) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 9216, 3072) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 0, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 3072, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 6144, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<12288xf32>, 9216, 3072, [<size = 16, stride = 192>, <size = 4, stride = 8>, <size = 6, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_3_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_3_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_3_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_3_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_3_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_3_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_0 : memref<64x48xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_1 : memref<64x48xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_3_buff_0 : memref<64x48xf32>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_3_buff_1 : memref<64x48xf32>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L2_0_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @C_L2L3_0_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @C_L2L3_2_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_3_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
