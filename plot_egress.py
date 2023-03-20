import floorplan
import plotter

dilated = True # change this according to whether or not a minimum width wants to be applied
if dilated:
    input_plan = floorplan.offset_plan
    output_plan = floorplan.dilated_plan
else:
    input_plan = floorplan.raw_plan
    output_plan = floorplan.plan

start = floorplan.starts[61] # choose which cell you want to start on
exits = floorplan.exits

# generate output plan that contains all the classifiers
floorplan.place_shortest_egress(input_plan, output_plan, start, exits)

plotter.plot(output_plan)