/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file top.cpp
 *
 * HLS Description of the CNV BNN with axi-lite based parameter loading (DoMemInit) 
 * and  dataflow architecture of the image inference (DoCompute).
 * The network uses 1 bit weights and 1 bit activation.
 *
 *****************************************************************************/
#include "config.h"

#include "bnn-library.h"

#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"

int lyr=0;

static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>  weights0;
static BinaryWeights<L1_SIMD, L1_PE, L1_WMEM>  weights1;
static BinaryWeights<L2_SIMD, L2_PE, L2_WMEM>  weights2;
static BinaryWeights<L3_SIMD, L3_PE, L3_WMEM>  weights3;
static BinaryWeights<L4_SIMD, L4_PE, L4_WMEM>  weights4;
static BinaryWeights<L5_SIMD, L5_PE, L5_WMEM>  weights5;
static BinaryWeights<L6_SIMD, L6_PE, L6_WMEM>  weights6;
static BinaryWeights<L7_SIMD, L7_PE, L7_WMEM>  weights7;
static BinaryWeights<L8_SIMD, L8_PE, L8_WMEM>  weights8;

static ThresholdsActivation<L0_TMEM, L0_PE, 15, ap_fixed<24, 16, AP_RND, AP_SAT>, ap_uint<L0_API>, 0>   threshs0;
static ThresholdsActivation<L1_TMEM, L1_PE, 15, ap_int<16>, ap_uint<L1_API>, 0>     threshs1;
static ThresholdsActivation<L2_TMEM, L2_PE, 15, ap_int<16>, ap_uint<L2_API>, 0>     threshs2;
static ThresholdsActivation<L3_TMEM, L3_PE, 15, ap_int<16>, ap_uint<L3_API>, 0>     threshs3;
static ThresholdsActivation<L4_TMEM, L4_PE, 15, ap_int<16>, ap_uint<L4_API>, 0>     threshs4;
static ThresholdsActivation<L5_TMEM, L5_PE, 15, ap_int<16>, ap_uint<L5_API>, 0>     threshs5;
static ThresholdsActivation<L6_TMEM, L6_PE, 15, ap_int<16>, ap_uint<L6_API>, 0>     threshs6;
static ThresholdsActivation<L7_TMEM, L7_PE, 15, ap_int<16>, ap_uint<L7_API>, 0>     threshs7;

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val) {
  switch (targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      break;
    case 1:
      threshs0.m_thresholds[targetMem][targetInd][targetThresh] = *reinterpret_cast<ap_fixed<64, 56> *>(&val);
      break;
    case 2:
      weights1.m_weights[targetMem][targetInd] = val;
      break;
    case 3:
      threshs1.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 4:
      weights2.m_weights[targetMem][targetInd] = val;
      break;
    case 5:
      threshs2.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 6:
      weights3.m_weights[targetMem][targetInd] = val;
      break;
    case 7:
      threshs3.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 8:
      weights4.m_weights[targetMem][targetInd] = val;
      break;
    case 9:
      threshs4.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 10:
      weights5.m_weights[targetMem][targetInd] = val;
      break;
    case 11:
      threshs5.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 12:
      weights6.m_weights[targetMem][targetInd] = val;
      break;
    case 13:
      threshs6.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 14:
      weights7.m_weights[targetMem][targetInd] = val;
      break;
    case 15:
      threshs7.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 16:
      weights8.m_weights[targetMem][targetInd] = val;
      break;
    case 17:
      // do nothing, no thres mem for layer 8 as PassThrough activation is used
      break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64>* out, const unsigned int numReps) {
#pragma HLS DATAFLOW
  stream<ap_uint<64>> inter0("DoCompute.inter0");
  stream<ap_uint<192>> inter0_1("DoCompute.inter0_1");
  stream<ap_uint<24>> inter0_2("DoCompute.inter0_2");
#pragma HLS STREAM variable=inter0_2 depth=128  

// c0 mx0
  stream<ap_uint<L0_OFM_CH>> inter1_0("DoCompute.inter1_0");
#pragma HLS STREAM variable=inter1_0 depth=128
  stream<ap_uint<L0_OFM_CH>> inter1_1("DoCompute.inter1_1");
#pragma HLS STREAM variable=inter1_1 depth=128

// c1 mx1
  stream<ap_uint<L1_OFM_CH>> inter2_0("DoCompute.inter2_0");
#pragma HLS STREAM variable=inter2_0 depth=128
  stream<ap_uint<L1_OFM_CH>> inter2_1("DoCompute.inter2_1");
#pragma HLS STREAM variable=inter2_1 depth=128

// c2 mx2
  stream<ap_uint<L2_OFM_CH>> inter3_0("DoCompute.inter3_0");
#pragma HLS STREAM variable=inter3_0 depth=64
  stream<ap_uint<L2_OFM_CH>> inter3_1("DoCompute.inter3_1");
#pragma HLS STREAM variable=inter3_1 depth=64

// c3 mx3
  stream<ap_uint<L3_OFM_CH>> inter4_0("DoCompute.inter4_0");
#pragma HLS STREAM variable=inter4_0 depth=128
  stream<ap_uint<L3_OFM_CH>> inter4_1("DoCompute.inter4_1");
#pragma HLS STREAM variable=inter4_1 depth=128

// c4 mx4
  stream<ap_uint<L4_OFM_CH>> inter5_0("DoCompute.inter5_0");
#pragma HLS STREAM variable=inter5_0 depth=128
  stream<ap_uint<L4_OFM_CH>> inter5_1("DoCompute.inter5_1");
#pragma HLS STREAM variable=inter5_1 depth=128

// c5 c6 c7 c8
  stream<ap_uint<L5_OFM_CH>> inter6("DoCompute.inter6");
#pragma HLS STREAM variable=inter6 depth=128
  stream<ap_uint<L6_OFM_CH>> inter7("DoCompute.inter7");
#pragma HLS STREAM variable=inter7 depth=128
  stream<ap_uint<L7_OFM_CH>> inter8("DoCompute.inter8");
#pragma HLS STREAM variable=inter8 depth=128
  // stream<ap_uint<L8_OFM_CH*64>> inter9("DoCompute.inter9");
  stream<ap_uint<5*64>> inter9("DoCompute.inter9");
#pragma HLS STREAM variable=inter9 depth=64  

 // out
  stream<ap_uint<64>> memOutStrm("DoCompute.memOutStrm");




  const unsigned int inBits = 416 * 416 * 3 * 8;
  // const unsigned int inBitsPadded = paddedSize(inBits, 64);
  const unsigned int outbytes = 13 * 13 * 125;
  const unsigned int outBits = outbytes*64;

  Mem2Stream_Batch<64, inBits / 8>(in, inter0, numReps);
  StreamingDataWidthConverter_Batch<64, 192, (416 * 416 * 3 * 8) / 64>(inter0, inter0_1, numReps);
  StreamingDataWidthConverter_Batch<192, 24, (416 * 416 * 3 * 8) / 192>(inter0_1, inter0_2, numReps);



  // convolutional layers
  // std::cout << "Conv0" << std::endl;
  ConvLayerFxdSame_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, L0_SIMD, L0_PE, Slice<ap_fixed<8, 1, AP_RND, AP_SAT>>, Slice<ap_uint<4>>, Recast<Binary>>(inter0_2, inter1_0, weights0, threshs0, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter1_0.size() << std::endl; 

  StreamingMaxPool_Precision_Batch<L0_OFM_DIM, 2, L0_OFM_CH, ap_uint<4>, 0>(inter1_0, inter1_1, numReps);
  // std::cout << "output depth: " << inter1_1.size() << std::endl;
 
  // std::cout << "Conv1" << std::endl;
  ConvLayerSame_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, Slice<ap_uint<4> >, Slice<ap_uint<4>>, Recast<Binary> >(inter1_1, inter2_0, weights1, threshs1, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter2_0.size() << std::endl;
    StreamingMaxPool_Precision_Batch<L1_OFM_DIM, 2, L1_OFM_CH, ap_uint<4>, 0>(inter2_0, inter2_1, numReps);
  // std::cout << "output depth: " << inter2_1.size() << std::endl;

  // std::cout << "Conv2" << std::endl;
  ConvLayerSame_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, Slice<ap_uint<4> >, Slice<ap_uint<4>>, Recast<Binary>>(inter2_1, inter3_0, weights2, threshs2, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter3_0.size() << std::endl;
  StreamingMaxPool_Precision_Batch<L2_OFM_DIM, 2, L2_OFM_CH, ap_uint<4>, 0>(inter3_0, inter3_1, numReps);
  // std::cout << "output depth: " << inter3_1.size() << std::endl;

  // std::cout << "Conv3" << std::endl;
  ConvLayerSame_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_SIMD, L3_PE, Slice<ap_uint<4> >, Slice<ap_uint<4>>, Recast<Binary>>(inter3_1, inter4_0, weights3, threshs3, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter4_0.size() << std::endl;
  StreamingMaxPool_Precision_Batch<L3_OFM_DIM, 2, L3_OFM_CH, ap_uint<4>, 0>(inter4_0, inter4_1, numReps);
  // std::cout << "output depth: " << inter4_1.size() << std::endl;

  // std::cout << "Conv4" << std::endl;
  ConvLayerSame_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM, L4_SIMD, L4_PE, Slice<ap_uint<4> >, Slice<ap_uint<4>>, Recast<Binary>>(inter4_1, inter5_0, weights4, threshs4, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter5_0.size() << std::endl;
  StreamingMaxPool_Precision_Batch<L4_OFM_DIM, 2, L4_OFM_CH, ap_uint<4>, 0>(inter5_0, inter5_1, numReps);
  // std::cout << "output depth: " << inter5_1.size() << std::endl;

  // std::cout << "Conv5" << std::endl;
  ConvLayerSame_Batch<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_CH, L5_OFM_DIM, L5_SIMD, L5_PE, Slice<ap_uint<4> >, Slice<ap_uint<4>>, Recast<Binary>>(inter5_1, inter6, weights5, threshs5, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter6.size() << std::endl;

  // std::cout << "Conv6" << std::endl;
  ConvLayerSame_Batch<L6_K, L6_IFM_CH, L6_IFM_DIM, L6_OFM_CH, L6_OFM_DIM, L6_SIMD, L6_PE, Slice<ap_uint<4> >, Slice<ap_uint<4>>, Recast<Binary>>(inter6, inter7, weights6, threshs6, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter7.size() << std::endl;

  // std::cout << "Conv7" << std::endl;
  ConvLayerSame_Batch<L7_K, L7_IFM_CH, L7_IFM_DIM, L7_OFM_CH, L7_OFM_DIM, L7_SIMD, L7_PE, Slice<ap_uint<4> >, Slice<ap_uint<4>>, Recast<Binary>>(inter7, inter8, weights7, threshs7, numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter8.size() << std::endl;
 

  // std::cout << "Conv8" << std::endl;
  ConvLayerSame_Batch<L8_K, L8_IFM_CH, L8_IFM_DIM, L8_OFM_CH, L8_OFM_DIM, L8_SIMD, L8_PE, Slice<ap_uint<4> >, Slice<ap_uint<64>>,Recast<Binary> >(inter8, inter9, weights8, PassThroughActivation<ap_uint<64>>(), numReps, ap_resource_lut());
  // std::cout << "output depth: " << inter9.size() << std::endl;


  // stream<ap_uint<L8_OFM_CH*32*2>> inter9x("DoCompute.inter9x"); 

  // std::cout << "Convert" << std::endl;
  // StreamingDataWidthConverter_Batch<125*32, 125*32*2, outBits / (125*32)>(inter9, inter9x, numReps); 
  StreamingDataWidthConverter_Batch<5*64, 64, outBits / (5*64)>(inter9, memOutStrm, numReps); 
  // std::cout << "output depth: " << memOutStrm.size() << std::endl;

  Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);
  // std::cout << "done" << std::endl;
}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=targetThresh bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=512
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=16
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weights0.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights1.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights2.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights3.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs3.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs3.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights4.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs4.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs4.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights5.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs5.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs5.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights6.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs6.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs6.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights7.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs7.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs7.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights8.m_weights complete dim=1

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }
}
