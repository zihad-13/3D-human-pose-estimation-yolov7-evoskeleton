import matplotlib.pyplot as plt
import numpy as np
# %%


def re_order(skeleton, gt=True):
    skeleton = skeleton.copy().reshape(-1, 3)
    # permute the order of x,y,z axis
    skeleton[:, [0, 1, 2]] = skeleton[:, [0, 2, 1]]
    '''if gt ==False:
        return skeleton.reshape(96)
    else:
        return skeleton'''
    return skeleton
def plot_3d_ax(ax,pred,
               elev=10,
               azim=-45,
               gt=True,
               reorder=True,
               title=None,RADIUS=1
               ):
    ax.view_init(elev=elev, azim=azim)
    if reorder:
        reordered_skeleton = re_order(skeleton=pred, gt=gt)
    else:
        reordered_skeleton = pred
    show3Dpose(channels = reordered_skeleton, ax=ax, gt=gt,RADIUS=RADIUS)
    #if gt:
        #show3Dpose_gt(reordered_skeleton, ax, gt)
    #else:
        #show3Dpose(reordered_skeleton, ax)
    plt.title(title)
    return
def show3Dpose(channels, #(17,3)
               ax,
               #lcolor="#3498db",
               #rcolor="#e74c3c",
               add_labels=True,
               gt=False,RADIUS=1
               #pred=False
               ):
    #print('input to show3dpose: ', channels.shape)
    #vals = np.reshape(channels, (17, -1))
    vals = channels
    #('reshaping input to show3dpose: ', vals.shape)
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14,
                  18, 19, 14, 26, 27])-1  # start points
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18,
                  19, 20, 26, 27, 28])-1  # end points
    pose_connection =np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
                   [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]) 
    all_points_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16])
    #The order for the 3D keypoints is:
    # 'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
    # 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder'
    # 'RElbow', 'RWrist'
    #LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    # Make connection matrix
    #plot lines in the 3d coordinate system
    once = False
    for i in np.arange(len(pose_connection)):
        x, y, z = [np.array([vals[pose_connection[i][0], j], vals[pose_connection[i][1], j]]) for j in range(3)]
        if once: #x -> (2,)
            print(x.shape);print(y.shape);print(z.shape);print(x,y,z) #print x,y,z for once only to see what they contain
            once = False
        if gt :
            color = '#3498db' #blue
        else:
            color = '#e74c3c' #red
            orange='#FFA500'
            green='#00FF00'
            blue='#3498db'
            color=['#3498db','#3498db','#3498db',
                   '#00FF00','#00FF00','#00FF00',
                   '#FFA500','#FFA500','#FFA500','#FFA500',
                   '#00FF00','#00FF00','#00FF00',
                   '#3498db','#3498db','#3498db']
            color=[blue,blue,blue,green,green,green,orange,orange,orange,orange,green,green,green,blue,blue,blue]
        ax.plot(x, y, z,  lw=2, c=color[i] )
    if gt:
        point_color = 'k'
    else: 
        point_color = 'g'
    #for i in range(len(all_points_indices)): #plot keypoints for joint positions in 3d model
        #ax.scatter(vals[all_points_indices[i],0],vals[all_points_indices[i],1],vals[all_points_indices[i],2], color=point_color)
        #pass
    #ax.set_title("First Plot")


    RADIUS = RADIUS # space around the subject
    # xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    # ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    # ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    # ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    # if add_labels:
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("z")
    #     ax.set_zlabel("y")
    # ax.set_aspect('auto')
    # # Get rid of the panes (actually, make them white)
    # white = (1.0, 1.0, 1.0, 0.0)
    # ax.w_xaxis.set_pane_color(white)
    # ax.w_yaxis.set_pane_color(white)
    # # Get rid of the lines in 3d
    # ax.w_xaxis.line.set_color(white)
    # ax.w_yaxis.line.set_color(white)
    # ax.w_zaxis.line.set_color(white)
    #if gt==False:
    #ax.invert_zaxis()
    return
def adjust_figure(left=0,
                  right=1,
                  bottom=0.01,
                  top=0.95,
                  wspace=0,
                  hspace=0.4
                  ):
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    return