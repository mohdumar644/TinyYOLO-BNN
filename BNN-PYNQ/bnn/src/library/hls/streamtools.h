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
 ******************************************************************************/
 
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file stream-tools.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of convenience funtions used to adapt stream size, 
 *  remove unnecessary streams (padding) and casting
 *
 ******************************************************************************/

#ifndef STREAMTOOLS_H
#define STREAMTOOLS_H

// only let the first X elements of a stream to pass through, the remainder
// are consumed from input but not re-emitted from the output
// useful for getting rid of e.g. padding words
template<unsigned int DataWidth,    // stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal       // total number of words (NumTotal-NumAllowed swallowed)
>
void StreamLimiter(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out) {
  CASSERT_DATAFLOW(NumTotal >= NumAllowed);
  unsigned int numLeft = NumAllowed;
  for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in.read();
    if (numLeft > 0) {
      out.write(e);
      numLeft--;
    }
  }
}

template<unsigned int DataWidth,	// stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal       // total number of words (NumTotal-NumAllowed swallowed)
>
void StreamLimiter_Batch(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out, unsigned int numReps) {
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamLimiter<DataWidth, NumAllowed, NumTotal>(in, out);
  }
}

template<typename InT, typename OutT>
void StreamingCast(hls::stream<InT> & in, hls::stream<OutT> & out, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write((OutT) in.read());
  }
}

template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords		// number of input words to process
>
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth> > & in,
		hls::stream<ap_uint<OutWidth> > & out, const unsigned int numReps) {
  if (InWidth > OutWidth) {
    // emit multiple output words per input word read
    CASSERT_DATAFLOW(InWidth % OutWidth == 0);
    const unsigned int outPerIn = InWidth / OutWidth;
    const unsigned int totalIters = NumInWords * outPerIn * numReps;
    unsigned int o = 0;
    ap_uint<InWidth> ei = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
      // read new input word if current out count is zero
      if (o == 0) {
        ei = in.read();
	  }
      // pick output word from the rightmost position
      ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
      out.write(eo);
      // shift input to get new output word for next iteration
      ei = ei >> OutWidth;
      // increment written output count
      o++;
      // wraparound indices to recreate the nested loop structure
      if (o == outPerIn) {
        o = 0;
      }
    }
  } else if (InWidth == OutWidth) {
    // straight-through copy
    for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II=1
      ap_uint<InWidth> e = in.read();
      out.write(e);
    }
  } else { // InWidth < OutWidth
    // read multiple input words per output word emitted
    CASSERT_DATAFLOW(OutWidth % InWidth == 0);
    const unsigned int inPerOut = OutWidth / InWidth;
    const unsigned int totalIters = NumInWords * numReps;
    unsigned int i = 0;
    ap_uint<OutWidth> eo = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
      // read input and shift into output buffer
      ap_uint<InWidth> ei = in.read();
      eo = eo >> InWidth;
      eo(OutWidth - 1, OutWidth - InWidth) = ei;
      // increment read input count
      i++;
      // wraparound logic to recreate nested loop functionality
      if (i == inPerOut) {
        i = 0;
        out.write(eo);
      }
    }
  }
}

template<unsigned IW, unsigned OW, unsigned N>
 class WidthAdjustedInputStream {
  hls::stream<ap_uint<OW>>  m_target;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<IW> >&  source, unsigned const  reps) {
    StreamingDataWidthConverter_Batch<IW, OW, N>(source, m_target, reps);
  }
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<OW> >&() {
    return  m_target;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedInputStream<W, W, N> {

  hls::stream<ap_uint<W>> &m_source;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<W> >&  source, unsigned const  reps) : m_source(source) {}
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_source;
  }
};


template<unsigned IW, unsigned OW, unsigned N>
class WidthAdjustedOutputStream {
  hls::stream<ap_uint<IW>>  m_buffer;
  hls::stream<ap_uint<OW>> &m_target;
  unsigned const  m_reps;
  
 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<OW> >&  target, unsigned const  reps) : m_target(target), m_reps(reps) {}
  ~WidthAdjustedOutputStream() {
    StreamingDataWidthConverter_Batch<IW, OW, N>(m_buffer, m_target, m_reps);
  }

 public:
  operator hls::stream<ap_uint<IW> >&() {
    return  m_buffer;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedOutputStream<W, W, N> {
  hls::stream<ap_uint<W>> &m_target;

 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<W> >&  target, unsigned const  reps)
    : m_target(target) {}
  ~WidthAdjustedOutputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_target;
  }
};





#include <fstream>


// Reshape input stream to output only useful data when padding is same:
// Might add 0s at left, right, upper, lower side of the input
// Pad with 0
template<unsigned int NumChannels>
void StreamPadZero(hls::stream<ap_uint<NumChannels> > &in, hls::stream<ap_uint<NumChannels> > &out, const unsigned int ImgDim, const unsigned int PaddedDim)
{

    // Padding
    const unsigned int Padding = PaddedDim - ImgDim;
    // Padding Up and Left
    const unsigned int PaddingUp = Padding / 2;
    const unsigned int PaddingLeft = Padding / 2;
    // Padding Down and Right (might be 1 element more than up and left in case of odd padding)
    const unsigned int PaddingDown = Padding - PaddingUp;
    const unsigned int PaddingRight = Padding - PaddingLeft;

    ap_uint<NumChannels> outData, inData;

 // std::ofstream outcsv; 
 //      outcsv.open("testinput.txt"); 



    for(unsigned int y = 0; y < PaddedDim; y++)
    {
        for(unsigned int x = 0; x < PaddedDim; x++)
        {
            #pragma HLS PIPELINE II=1

            // Padding Rows
            if(y < PaddingUp || y >= (PaddedDim - PaddingDown))
            {
                outData = 0;
            }
            // Padding Cols
            else if(x < PaddingLeft || x >= (PaddedDim - PaddingRight))
            {
                outData = 0;
            }
            // No Padding
            else
            {
                inData = in.read();
 //                std::cout << std::setprecision(14) << x << " " << y << " = " << inData << std::endl;
 
 // ap_fixed<8, 1, AP_RND, AP_SAT>   tmp;
 // tmp.V = inData(15,8);
 //        outcsv << std::setprecision(14) << x << " " << y << " = " << tmp << std::endl;


                outData = inData;
            }

            out.write(outData);
        }
    }
    // outcsv.close();
}

template<unsigned int NumChannels>
void StreamPadZero_Batch(hls::stream<ap_uint<NumChannels> > &in, hls::stream<ap_uint<NumChannels> > &out,
    const unsigned int ImgDim, const unsigned int PaddedDim, unsigned int numReps)
{
  for(unsigned int rep = 0; rep < numReps; rep++)
  {
    StreamPadZero<NumChannels>(in, out, ImgDim, PaddedDim);
  }
}




// lookup table defined for alternate padding in StreamPad for 256 channels image

// Reshape input stream to output only useful data when padding is same:
// Might add 0s at left, right, upper, lower side of the input
// Pad with 0
template<unsigned int NumChannels>
void StreamPad(hls::stream<ap_uint<NumChannels> > &in, hls::stream<ap_uint<NumChannels> > &out, const unsigned int ImgDim, const unsigned int PaddedDim)
{
    if (NumChannels > 512)
    {
      cout << "Error!! If you are using NumChannels > 256 for padding, Please update the lookuptable in the file streamtools.h" << endl;
      exit(-1);
    }
    // Padding
    const unsigned int Padding = PaddedDim - ImgDim;
    // Padding Up and Left
    const unsigned int PaddingUp = Padding / 2;
    const unsigned int PaddingLeft = Padding / 2;
    // Padding Down and Right (might be 1 element more than up and left in case of odd padding)
    const unsigned int PaddingDown = Padding - PaddingUp;
    const unsigned int PaddingRight = Padding - PaddingLeft;

    ap_uint<NumChannels> outData,outData1,outData2,outData3, inData;

    // Using lookup table
    // Using equation 
    //ap_uint<NumChannels> val = (1/3)*((2^(NumChannels-1))*(3+(-1)^NumChannels)-2);
    for(unsigned int y = 0; y < PaddedDim; y++)
    {
        for(unsigned int x = 0; x < PaddedDim; x++)
        {
            #pragma HLS PIPELINE II=1

            // Padding Rows
            if(y < PaddingUp || y >= (PaddedDim - PaddingDown))
            {
                outData = 0;
                outData1 = 0; 
                outData2 = 0; 
                outData3 = 0; 
            }
            // Padding Cols
            else if(x < PaddingLeft || x >= (PaddedDim - PaddingRight))
            {
                outData = 0;
                outData1 = 0; 
                outData2 = 0; 
                outData3 = 0; 
            }
            // No Padding
            else
            {
                inData = in.read();
                outData = inData;
                inData = in.read();
                outData1 = inData;
                inData = in.read();
                outData2 = inData;
                inData = in.read();
                outData3 = inData;
            }

            out.write(outData);
            out.write(outData1);
            out.write(outData2);
            out.write(outData3);
        }
    }
}

template<unsigned int NumChannels>
void StreamPad_Batch(hls::stream<ap_uint<NumChannels> > &in, hls::stream<ap_uint<NumChannels> > &out,
    const unsigned int ImgDim, const unsigned int PaddedDim, unsigned int numReps)
{
  for(unsigned int rep = 0; rep < numReps; rep++)
  {
    StreamPad<NumChannels>(in, out, ImgDim, PaddedDim);
  }
}





#endif



