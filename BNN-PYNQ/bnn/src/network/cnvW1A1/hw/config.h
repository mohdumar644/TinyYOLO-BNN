/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =   416  IFM_CH =     3
 *      OFM  =   416  OFM_CH =    16
 *     SIMD  =     3    PE   =    16
 *     WMEM  =     9   TMEM  =     1
 *     #Ops  = 149520384   Ext Latency  = 1557504
**/

#define L0_K 3
#define L0_IFM_CH 3
#define L0_IFM_DIM 416
#define L0_OFM_CH 16
#define L0_OFM_DIM 416
#define L0_SIMD 3
#define L0_PE 16
#define L0_WMEM 9
#define L0_TMEM 1
#define L0_WPI 1
#define L0_API 4
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =   208  IFM_CH =    16
 *      OFM  =   208  OFM_CH =    64
 *     SIMD  =    16    PE   =    16
 *     WMEM  =    36   TMEM  =     4
 *     #Ops  = 797442048   Ext Latency  = 1557504
**/

#define L1_K 3
#define L1_IFM_CH 16
#define L1_IFM_DIM 208
#define L1_OFM_CH 64
#define L1_OFM_DIM 208
#define L1_SIMD 16
#define L1_PE 16
#define L1_WMEM 36
#define L1_TMEM 4
#define L1_WPI 1
#define L1_API 4
#define L1_WPF 0
#define L1_APF 0

/**
 * Convolutional Layer L2:
 *      IFM  =   104  IFM_CH =    64
 *      OFM  =   104  OFM_CH =    64
 *     SIMD  =    32    PE   =    16
 *     WMEM  =    72   TMEM  =     4
 *     #Ops  = 797442048   Ext Latency  = 778752
**/

#define L2_K 3
#define L2_IFM_CH 64
#define L2_IFM_DIM 104
#define L2_OFM_CH 64
#define L2_OFM_DIM 104
#define L2_SIMD 32
#define L2_PE 16
#define L2_WMEM 72
#define L2_TMEM 4
#define L2_WPI 1
#define L2_API 4
#define L2_WPF 0
#define L2_APF 0

/**
 * Convolutional Layer L3:
 *      IFM  =    52  IFM_CH =    64
 *      OFM  =    52  OFM_CH =   128
 *     SIMD  =    32    PE   =    16
 *     WMEM  =   144   TMEM  =     8
 *     #Ops  = 398721024   Ext Latency  = 389376
**/

#define L3_K 3
#define L3_IFM_CH 64
#define L3_IFM_DIM 52
#define L3_OFM_CH 128
#define L3_OFM_DIM 52
#define L3_SIMD 32
#define L3_PE 16
#define L3_WMEM 144
#define L3_TMEM 8
#define L3_WPI 1
#define L3_API 4
#define L3_WPF 0
#define L3_APF 0

/**
 * Convolutional Layer L4:
 *      IFM  =    26  IFM_CH =   128
 *      OFM  =    26  OFM_CH =   256
 *     SIMD  =    32    PE   =    16
 *     WMEM  =   576   TMEM  =    16
 *     #Ops  = 398721024   Ext Latency  = 389376
**/

#define L4_K 3
#define L4_IFM_CH 128
#define L4_IFM_DIM 26
#define L4_OFM_CH 256
#define L4_OFM_DIM 26
#define L4_SIMD 32
#define L4_PE 16
#define L4_WMEM 576
#define L4_TMEM 16
#define L4_WPI 1
#define L4_API 4
#define L4_WPF 0
#define L4_APF 0

/**
 * Convolutional Layer L5:
 *      IFM  =    13  IFM_CH =   256
 *      OFM  =    13  OFM_CH =   512
 *     SIMD  =    32    PE   =    16
 *     WMEM  =  2304   TMEM  =    32
 *     #Ops  = 398721024   Ext Latency  = 389376
**/

#define L5_K 3
#define L5_IFM_CH 256
#define L5_IFM_DIM 13
#define L5_OFM_CH 512
#define L5_OFM_DIM 13
#define L5_SIMD 32
#define L5_PE 16
#define L5_WMEM 2304
#define L5_TMEM 32
#define L5_WPI 1
#define L5_API 4
#define L5_WPF 0
#define L5_APF 0

/**
 * Convolutional Layer L6:
 *      IFM  =    13  IFM_CH =   512
 *      OFM  =    13  OFM_CH =   512
 *     SIMD  =    32    PE   =    16
 *     WMEM  =  4608   TMEM  =    32
 *     #Ops  = 797442048   Ext Latency  = 778752
**/

#define L6_K 3
#define L6_IFM_CH 512
#define L6_IFM_DIM 13
#define L6_OFM_CH 512
#define L6_OFM_DIM 13
#define L6_SIMD 32
#define L6_PE 16
#define L6_WMEM 4608
#define L6_TMEM 32
#define L6_WPI 1
#define L6_API 4
#define L6_WPF 0
#define L6_APF 0

/**
 * Convolutional Layer L7:
 *      IFM  =    13  IFM_CH =   512
 *      OFM  =    13  OFM_CH =   512
 *     SIMD  =    32    PE   =    16
 *     WMEM  =  4608   TMEM  =    32
 *     #Ops  = 797442048   Ext Latency  = 778752
**/

#define L7_K 3
#define L7_IFM_CH 512
#define L7_IFM_DIM 13
#define L7_OFM_CH 512
#define L7_OFM_DIM 13
#define L7_SIMD 32
#define L7_PE 16
#define L7_WMEM 4608
#define L7_TMEM 32
#define L7_WPI 1
#define L7_API 4
#define L7_WPF 0
#define L7_APF 0

/**
 * Convolutional Layer L8:
 *      IFM  =    13  IFM_CH =   512
 *      OFM  =    13  OFM_CH =   125
 *     SIMD  =    32    PE   =     5
 *     WMEM  =   400   TMEM  =    25
 *     #Ops  = 21632000   Ext Latency  = 67600
**/

#define L8_K 1
#define L8_IFM_CH 512
#define L8_IFM_DIM 13
#define L8_OFM_CH 125
#define L8_OFM_DIM 13
#define L8_SIMD 32
#define L8_PE 5
#define L8_WMEM 400
#define L8_TMEM 25
#define L8_WPI 1
#define L8_API 1
#define L8_WPF 0
#define L8_APF 0

#endif //__LAYER_CONFIG_H_
