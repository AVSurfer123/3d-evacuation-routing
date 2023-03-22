
# import floorplan
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot(plan):
    # LOAD THE PLAN
    # dilate_walls = False
    # if not dilate_walls:
    #     plan = floorplan.plan 
    # else:
    #     plan = floorplan.dilated_plan 


    # print(np.amax(plan))

    # PLOT THE PLAN TO SEE WHERE EVERYTHING IS
    # [white, black, red, green, blue, gray]
    color_full = ["#FFFFFF", "#000000", "#FF0000", "#00FF00","#0000FF", "#808080", "#A020F0", "#FFA500"]
    tick_full = ['walkable','wall','start','exit','egress','offset','failed_start', 'long_egress_start']
    
    colors = color_full[:np.amax(plan)+1]
    ticks = tick_full[:np.amax(plan)+1]
    



    # Create a colormap based on the colors and the range of the data
    cmap = mcolors.LinearSegmentedColormap.from_list("", list(zip(np.linspace(0, 1, len(colors)), colors)))

    # Plot the data using imshow with the custom colormap
    plt.imshow(plan.T, cmap=cmap)

    # Add a colorbar to show the color scale
    cbar = plt.colorbar()
    #cbar.ax.set_yticklabels(ticks)

    # plt.imshow(plan.T)
    plt.gca().invert_yaxis()
    plt.show()