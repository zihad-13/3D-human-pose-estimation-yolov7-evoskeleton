from utils import analyzer_utils
import matplotlib.pyplot as plt
import io
import numpy as np
import cv2
# %%

evo_complete_ind=['hip_x','hip_y','hip_z','right_hip_x', 'right_hip_y','right_hip_z','right_knee_x', 'right_knee_y', 'right_knee_z',
 'right_ankle_x', 'right_ankle_y', 'right_ankle_z','left_hip_x', 'left_hip_y', 'left_hip_z','left_knee_x', 'left_knee_y', 'left_knee_z',
 'left_ankle_x','left_ankle_y', 'left_ankle_z','spine_x','spine_y','spine_z','thorax_x','thorax_y','thorax_z',
 'nose_x','nose_y','nose_z','head_x','head_y','head_z','left_shoulder_x', 'left_shoulder_y','left_shoulder_z',
 'left_elbow_x', 'left_elbow_y','left_elbow_z','left_wrist_x', 'left_wrist_y', 'left_wrist_z',
 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z','right_elbow_x','right_elbow_y', 'right_elbow_z',
 'right_wrist_x', 'right_wrist_y', 'right_wrist_z']

# %%

def plot_skeleton_3d_kpts(kp_data):
    kp_data_plot=kp_data.copy()
    kp_data_plot['nose_x']=(kp_data_plot['head_x']+kp_data_plot['thorax_x'])/2
    kp_data_plot['nose_y']=(kp_data_plot['head_y']+kp_data_plot['thorax_y'])/2
    kp_data_plot['nose_z']=(kp_data_plot['head_z']+kp_data_plot['thorax_z'])/2

    kp_data_plot=kp_data_plot[evo_complete_ind]
    fig=plot_3D_skel(kp_data_plot)
    return fig
# %%
def plot_3D_skel(kp_data_plot,figsize=(5,5),tiles_x=1,tiles_y=1,evo_reorder=True, RADIUS=500):
    fig = plt.figure(figsize=figsize)
    for i in range(kp_data_plot.shape[0]):
        kp_evo=kp_data_plot.iloc[i].values.reshape(17,3)
        if evo_reorder:
            kp_evo[..., 1] = -kp_evo[..., 1]
            kp_evo = kp_evo[..., [0, 2, 1]]
    
        ax = fig.add_subplot(tiles_x,tiles_y, 1,projection='3d')
        view_angle=-90
        analyzer_utils.plot_3d_ax(ax=ax,
                    pred=kp_evo,
                    elev=10.,
                    azim=view_angle,
                    gt=False, reorder=False,RADIUS=RADIUS,
                    title=' 3D prediction view angle %s'%(view_angle))
    #plt.show()            
    # plt.ioff()
    # plt.close()
    return fig

# def get_img_from_fig(fig):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=50)
#     #fig.savefig(buf, format="png")
#     buf.seek(0)
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()
#     img = cv2.imdecode(img_arr, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     return img

def get_img_from_fig(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im

# def get_img_from_fig(fig):
#     fig.canvas.draw()

#     # convert canvas to image
#     im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
#             sep='')
#     im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#     # img is rgb, convert to opencv's default bgr
#     im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
#     return im