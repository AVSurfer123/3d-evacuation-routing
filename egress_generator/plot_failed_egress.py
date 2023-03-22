import numpy as np

import floorplan
import plotter

raw_plan = floorplan.raw_plan
plan = floorplan.plan


dilated = True # change this according to whether or not a minimum width wants to be applied
if dilated:
    input_plan = floorplan.offset_plan
    output_plan = floorplan.dilated_plan
    exits = floorplan.dilated_exits
    
else:
    input_plan = raw_plan
    output_plan = plan
    exits = floorplan.exits

starts = floorplan.dilated_starts


failed_starts = []
for i, start in enumerate(starts):
    loop_input_plan = np.copy(input_plan)
    loop_output_plan = np.copy(output_plan)
    print("generating egress for start "+str(i)+" "+str(start))
    if not floorplan.place_shortest_egress(loop_input_plan, loop_output_plan, start, exits): # figure out why this is bugging out
        failed_starts.append(start)
        
floorplan.place_classifier(output_plan, failed_starts, 6)
plotter.plot(output_plan)
