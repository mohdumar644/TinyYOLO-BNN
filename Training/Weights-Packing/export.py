import torch 
import numpy as np

   

net = torch.load('pretrained.pt', map_location = 'cpu')['network']
 
dic = {
		'arr_0':net['layers.1_convbatch.layers.0.weight'],
		'arr_1':np.zeros(net['layers.1_convbatch.layers.0.weight'].shape[0],dtype=np.float32),
		'arr_2':net['layers.1_convbatch.layers.1.bias'],
		'arr_3':net['layers.1_convbatch.layers.1.weight'],
		'arr_4':net['layers.1_convbatch.layers.1.running_mean'],
		'arr_5':1./(np.sqrt(net['layers.1_convbatch.layers.1.running_var'])),

		'arr_6':net['layers.3_convbatch.layers.0.weight'],
		'arr_7':np.zeros(net['layers.3_convbatch.layers.0.weight'].shape[0],dtype=np.float32), 
		'arr_8':net['layers.3_convbatch.layers.1.bias'],
		'arr_9':net['layers.3_convbatch.layers.1.weight'],
		'arr_10':net['layers.3_convbatch.layers.1.running_mean'], 
		'arr_11':1./(np.sqrt(net['layers.3_convbatch.layers.1.running_var'])),

		'arr_12':net['layers.5_convbatch.layers.0.weight'],
		'arr_13':np.zeros(net['layers.5_convbatch.layers.0.weight'].shape[0],dtype=np.float32),  
		'arr_14':net['layers.5_convbatch.layers.1.bias'],
		'arr_15':net['layers.5_convbatch.layers.1.weight'],
		'arr_16':net['layers.5_convbatch.layers.1.running_mean'], 
		'arr_17':1./(np.sqrt(net['layers.5_convbatch.layers.1.running_var'])),
 
		'arr_18':net['layers.7_convbatch.layers.0.weight'],
		'arr_19':np.zeros(net['layers.7_convbatch.layers.0.weight'].shape[0],dtype=np.float32),   
		'arr_20':net['layers.7_convbatch.layers.1.bias'],
		'arr_21':net['layers.7_convbatch.layers.1.weight'],
		'arr_22':net['layers.7_convbatch.layers.1.running_mean'], 
		'arr_23':1./(np.sqrt(net['layers.7_convbatch.layers.1.running_var'])),
 
		'arr_24':net['layers.9_convbatch.layers.0.weight'],
		'arr_25':np.zeros(net['layers.9_convbatch.layers.0.weight'].shape[0],dtype=np.float32),   
		'arr_26':net['layers.9_convbatch.layers.1.bias'],
		'arr_27':net['layers.9_convbatch.layers.1.weight'],
		'arr_28':net['layers.9_convbatch.layers.1.running_mean'], 
		'arr_29':1./(np.sqrt(net['layers.9_convbatch.layers.1.running_var'])),
 
		'arr_30':net['layers.11_convbatch.layers.0.weight'],
		'arr_31':np.zeros(net['layers.11_convbatch.layers.0.weight'].shape[0],dtype=np.float32),   
		'arr_32':net['layers.11_convbatch.layers.1.bias'],
		'arr_33':net['layers.11_convbatch.layers.1.weight'],
		'arr_34':net['layers.11_convbatch.layers.1.running_mean'], 
		'arr_35':1./(np.sqrt(net['layers.11_convbatch.layers.1.running_var'])),
 
		'arr_36':net['layers.13_convbatch.layers.0.weight'],
		'arr_37':np.zeros(net['layers.13_convbatch.layers.0.weight'].shape[0],dtype=np.float32),   
		'arr_38':net['layers.13_convbatch.layers.1.bias'],
		'arr_39':net['layers.13_convbatch.layers.1.weight'],
		'arr_40':net['layers.13_convbatch.layers.1.running_mean'], 
		'arr_41':1./(np.sqrt(net['layers.13_convbatch.layers.1.running_var'])),
 
		'arr_42':net['layers.14_convbatch.layers.0.weight'],
		'arr_43':np.zeros(net['layers.14_convbatch.layers.0.weight'].shape[0],dtype=np.float32),   
		'arr_44':net['layers.14_convbatch.layers.1.bias'],
		'arr_45':net['layers.14_convbatch.layers.1.weight'],
		'arr_46':net['layers.14_convbatch.layers.1.running_mean'], 
		'arr_47':1./(np.sqrt(net['layers.14_convbatch.layers.1.running_var'])),
  
		'arr_48':net['layers.15_conv.weight'],    
		'arr_49':np.transpose(net['layers.15_conv.bias']),  

		'arr_50':np.zeros(net['layers.15_conv.weight'].shape[0],dtype=np.float32)+1e-7, 
		'arr_51':np.zeros(net['layers.15_conv.weight'].shape[0],dtype=np.float32)+1e-7, 
		'arr_52':np.zeros(net['layers.15_conv.weight'].shape[0],dtype=np.float32)+1e-7, 
		'arr_53':np.zeros(net['layers.15_conv.weight'].shape[0],dtype=np.float32)+1e-7, 
	}
  
for x in dic:
	dic[x]  = np.asarray(dic[x],dtype= np.float64)

np.savez('tiny-yolo.npz', **dic)
