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
```

Then to run in VMD on (centered) protein loaded:
```tcl
source filtered10.1.tcl
# output
START 3200, dist 18.64767074584961, len 53
frame 3201: score 0.985 <Residue O2IF, 26792>, OI: 0.582870
frame 3202: score 0.992 <Residue O2IF, 26792>, OI: 0.295197
frame 3203: score 0.992 <Residue O2IF, 26792>, OI: 0.626866
frame 3204: score 0.959 <Residue O2IF, 26792>, OI: 0.293751
frame 3205: score 0.985 <Residue O2IF, 26792>, OI: 0.411988
frame 3206: score 0.809 <Residue O2IF, 26792>, OI: 1.264871
frame 3207: score 0.668 <Residue LEU, 324>, HG: 2.155060
...
frame 3250: score 0.999 <Residue O2IF, 26792>, OI: 0.452186
frame 3251: score 0.984 <Residue O2IF, 26792>, OI: 0.832414
frame 3252: score 0.946 <Residue O2IF, 26792>, OI: 0.867487
frame 3253: score 0.996 <Residue TIP3, 6068>, H1: 2.333936

# to pause animation
pause_animation

# to resume animation
resume_animation

# to draw an event number
draw_event 18
# output
START 4200, dist 19.89888572692871, len 10
frame 4201: score 0.966 <Residue TIP3, 4525>, H1: 2.643651
frame 4202: score 0.989 <Residue O2IF, 26770>, OI: 0.484474
frame 4203: score 0.884 <Residue O2IF, 26752>, OI: 1.502760
frame 4204: score 0.84 <Residue O2IF, 26752>, OI: 0.906842
frame 4205: score 0.986 <Residue VAL, 217>, HB: 2.548701
frame 4206: score 0.998 <Residue O2IF, 26752>, OI: 0.365187
frame 4207: score 0.846 <Residue CYMX, 30001>, HNZ1: 2.193465
frame 4208: score 0.963 <Residue TIP3, 7816>, H1: 2.130443
frame 4209: score 0.909 <Residue CYMX, 30001>, HY1: 2.122949
frame 4210: score 0.985 <Residue CYMX, 30001>, HB2: 2.407697
```
<img width="826" height="884" alt="image" src="https://github.com/user-attachments/assets/0133fa83-8869-42ea-ae37-e5ff79426f51" />


