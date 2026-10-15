import re
import sys
from itertools import cycle
import numpy as np
import glob

# python python2vmd.py $folder_suffix filtered

folder_suffix=sys.argv[1]
tcl_file_prefix=sys.argv[2]

def tcl_out(filtered_events, OUTPUT_TCL):
    
    color_cycle = cycle(COLORS)
    output = []

    output.append("""
    set animation_after_id ""

    proc draw_one_event {idx} {

        global events

        set event [lindex $events $idx]

        set start   [lindex $event 0]
        set color   [lindex $event 1]
        set points  [lindex $event 2]
        set label   [lindex $event 3]
        set nearest [lindex $event 4]

        puts $label

        draw color blue
        draw sphere $start radius 1.0

        draw color $color

        set prev $start

        for {set i 0} {$i < [llength $points]} {incr i} {

            set p [lindex $points $i]

            draw sphere $p radius 0.5
            vmd_draw_arrow top $prev $p

            puts [lindex $nearest $i]

            set prev $p
        }
    }

    proc draw_event {idx} {

        global events
        global animation_after_id
        global current_frame

        set current_frame $idx

        draw delete all

        draw_one_event $idx

        display update ui

        set next [expr {$idx + 1}]
        if {$next < [llength $events]} {
            set animation_after_id [after %d [list draw_event $next]]
        }
    }

    proc draw_all_events {} {

        global events

        draw delete all

        for {set i 0} {$i < [llength $events]} {incr i} {
            draw_one_event $i
        }

        display update ui
    }

    proc stop_animation {} {

        global animation_after_id

        after cancel $animation_after_id


    }

    proc pause_animation {} {

        global animation_after_id
        global animation_paused

        set animation_paused 1

        if {$animation_after_id ne ""} {
            after cancel $animation_after_id
        }

        puts "Animation paused"
    }

    proc resume_animation {} {

        global current_frame
        global animation_paused

        set animation_paused 0

        puts "Resuming animation"

        # continue from NEXT frame
        draw_event [expr {$current_frame + 1}]
    }

    proc restart_animation {} {

        global animation_paused

        set animation_paused 0

        draw_event 0
    }
    """ % FRAME_DELAY_MS)

    # ---- delay variable ----
    # output.append(f"set FRAME_DELAY_MS {FRAME_DELAY_MS}")

    # ---- event list ----
    output.append("set events {")

    for event in filtered_events:

        start = event["start"]
        points = event["points"]
        scores = event["scores"]
        nearest_atoms = event["closet_points"]

        color = next(color_cycle)

        start_idx = event["start_idx"]
        tensor_dist = event["tensor_dist"]

        label = ""

        if start_idx is not None and tensor_dist is not None:
            label = f"START {start_idx}, dist {tensor_dist}, len {len(points)}"

        start_tcl = f"{{ {start[0]} {start[1]} {start[2]} }}"

        point_strs = [
            f"{{ {p[0]} {p[1]} {p[2]} }}"
            for p in points
        ]

        points_tcl = "{ " + " ".join(point_strs) + " }"

        nearest_strs = [
            f"\"frame {start_idx+i+1}: score {round(scores[i+1],3)} {p}\""
            for i,p in enumerate(nearest_atoms)
        ]

        nearest_tcl = "{ " + " ".join(nearest_strs) + " }"

        output.append(
            f"    {{ {start_tcl} {color} {points_tcl} \"{label}\" {nearest_tcl} }}"
        )

    output.append("}")

    # ---- start animation ----
    output.append("")
    output.append("draw_event 0")
    with open(OUTPUT_TCL, "w") as f:
        f.write("\n".join(output))


dl = []
dens = []
for Sim in range(1,11):
    INPUT_DIR  = "Sim%d_predict_events%s/"%(Sim, folder_suffix)
    OUTPUT_TCL = "%s%d.%s.tcl"%(tcl_file_prefix, Sim, folder_suffix)
    COLORS = [
        'blue', 'green', 'black', 'orange', 'yellow',
        'red', 'purple', 'cyan', 'silver', 'pink',
        'green2', 'violet'
    ]
    color_cycle = cycle(COLORS)
    # COLORS=["green","green"]
    DISTANCE_THRESHOLD = 6.0
    FRAME_DELAY_MS = 5000



    filtered_events = []

    for i, file in enumerate(glob.glob(INPUT_DIR+"*.npy")):
        # print(file)
        e = []
        event = np.load(file, allow_pickle=True).item()
        dens.append(event['init_dens'])
        current_event = {
                    "start": None,
                    "points": [],
                    "start_idx": None,
                    "tensor_dist": None
                }
        if len(event['path']) > 70:
            print(file)
            print([np.linalg.norm(f['frame_pos']) >= 12 for f in event['path'][70:]])
        
        current_event['start'] = event['path'][0]['frame_pos']
        current_event['start_idx'] = event['path'][0]['frame']
        e.append(Sim)
        e.append(current_event['start_idx'])
        e.append(current_event['start_idx']+len(event['path']))
        e.append(next(color_cycle))
        current_event['points'] = [point['frame_pos'] for point in event['path'][1:]]
        current_event['scores'] = [point['score'] for point in event['path'][0:]]
        current_event['tensor_dist'] = np.linalg.norm(event['path'][0]['frame_pos'])
        current_event['closet_points'] = [point['closest_atom'] for point in event['path'][1:]]
        filtered_events.append(current_event)

        tcl_out(filtered_events, OUTPUT_TCL)
        dl.append(e)



