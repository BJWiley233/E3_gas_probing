### python to VMD script for the visulazation of Numpy files from [inference](https://github.com/BJWiley233/E3_gas_probing/blob/main/inference.py#L1614)

```
python python2vmd.py <folder_suffix>

# example
folder_suffix=1
python python2vmd.py Sim1_predict_events$folder_suffix filtered1.$folder_suffix.tcl
python python2vmd.py Sim7_predict_events$folder_suffix filtered7.$folder_suffix.tcl

folder_suffix=3
python python2vmd.py Sim7_predict_events$folder_suffix filtered7.$folder_suffix.tcl
python python2vmd.py Sim7_predict_events$folder_suffix filtered7.$folder_suffix.tcl



## defualt is a bulk of Sim1-Sim10
folder_suffix=1
python python2vmd.py $folder_suffix filtered1.$folder_suffix.tcl

