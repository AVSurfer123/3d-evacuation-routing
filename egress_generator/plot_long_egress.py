import numpy as np

import floorplan
import plotter
import plot_failed_egress


raw_plan = floorplan.raw_plan
plan = floorplan.plan

maximum_egress_length = 7

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

plan = plot_failed_egress.output_plan
start_id = np.where(plan == 0)
starts = np.c_[start_id[0], start_id[1]]
starts = list(map(tuple, starts))


long_egress_starts = []
for i, start in enumerate(starts):
    loop_input_plan = np.copy(input_plan)
    loop_output_plan = np.copy(output_plan)
    print("generating egress for start "+str(i)+" "+str(start))
    if floorplan.place_shortest_egress(loop_input_plan, loop_output_plan, start, exits):
        if np.count_nonzero(loop_output_plan == 4) + 2 > maximum_egress_length:
            long_egress_starts.append(start)
        
floorplan.place_classifier(output_plan, long_egress_starts, 7)
plotter.plot(output_plan)
