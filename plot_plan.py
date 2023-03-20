import floorplan
import plotter

dilated = False # change this according to whether or not a minimum width wants to be applied
if dilated:
    input_plan = floorplan.offsets_plan
    output_plan = floorplan.dilated_plan
else:
    input_plan = floorplan.raw_plan
    output_plan = floorplan.plan

start = floorplan.starts[58] # choose which cell you want to start on
exits = floorplan.exits


plotter.plot(output_plan)