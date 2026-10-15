### python to VMD script for the visulazation of Numpy files from [inference](https://github.com/BJWiley233/E3_gas_probing/blob/main/inference.py#L1614)

```
python python2vmd.py <folder_suffix> <tcl_file_prefix>

# example
folder_suffix=1
python python2vmd.py Sim1_predict_events$folder_suffix filtered
python python2vmd.py Sim7_predict_events$folder_suffix filtered

folder_suffix=3
python python2vmd.py Sim7_predict_events$folder_suffix filtered
python python2vmd.py Sim7_predict_events$folder_suffix filtered



## defualt is a bulk of Sim1-Sim10
folder_suffix=9
python python2vmd.py $folder_suffix filtered

