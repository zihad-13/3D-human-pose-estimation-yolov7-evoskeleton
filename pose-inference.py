import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
import config
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.plot_3d_kp import plot_skeleton_3d_kpts
import os
import pandas as pd
from libs.dataset.h36m.data_utils import unNormalizeData
import libs.model.model as libm
from helper import inference_3d
import cv2
from utils.plot_3d_kp import plot_3D_skel,get_img_from_fig
import matplotlib.pyplot as plt
# %%
args = config.gen_parser().parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if 'cuda' in device.type:
    cuda=True
else: cude=False
keypoint_names={0: 'nose',
 1: 'left_eye',
 2: 'right_eye',
 3: 'left_ear',
 4: 'right_ear',
 5: 'left_shoulder',
 6: 'right_shoulder',
 7: 'left_elbow',
 8: 'right_elbow',
 9: 'left_wrist',
 10: 'right_wrist',
 11: 'left_hip',
 12: 'right_hip',
 13: 'left_knee',
 14: 'right_knee',
 15: 'left_ankle',
 16: 'right_ankle'}
all_kp_coord_names=np.char.add(np.array(list(keypoint_names.values())).reshape(len(keypoint_names),1),np.array(['_x','_y','_conf'])).flatten()

# %% load 3d lifting model
ckpt = torch.load(args.pretrained_model_path_3d)
stats = np.load(args.data_stats, allow_pickle=True).item()
cascade = libm.get_cascade()
input_size = 32
output_size = 48
for stage_id in range(2):
    # initialize a single deep learner
    stage_model = libm.get_model(stage_id + 1,
                                refine_3d=False,
                                norm_twoD=False,
                                num_blocks=2,
                                input_size=input_size,
                                output_size=output_size,
                                linear_size=1024,
                                dropout=0.5,
                                leaky=False)
    cascade.append(stage_model)

cascade.load_state_dict(ckpt)
if cuda:
    cascade.to(device)
cascade.eval()
# %% load 2d pose estimator
weigths = torch.load(args.weights)
model = weigths['model']
model = model.half().to(device)
_ = model.eval()
 
input_vid = args.source

# %%
vidcap = cv2.VideoCapture(input_vid)
if (vidcap.isOpened() == False):
  print('Error while trying to read video. Please check path again')
 
# Get the frame width and height.
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
 
# Pass the first frame through `letterbox` function to get the resized image,
# to be used for `VideoWriter` dimensions. Resize by larger side.
vid_write_image = letterbox(vidcap.read()[1], (frame_width), stride=64, auto=True)[0]
resize_height, resize_width = vid_write_image.shape[:2]
 
save_name = input_vid.split('/')[-1].split('.')[0]+"_2d_keypoints.mp4"
outdir=args.output_dir
os.makedirs(outdir, exist_ok=True)
savedir=os.path.join(outdir,save_name)

# Define codec and create VideoWriter object .
out = cv2.VideoWriter(savedir,cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (resize_width, resize_height))
 
 
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
# %%
import pdb
#pdb.set_trace()
each_frame_dfs=[]
while(vidcap.isOpened):
#while(1):
  # Capture each frame of the video.
  ret, frame = vidcap.read()
  #ret=True; frame=cv2.imread('personal.jpg')
  if ret:
      orig_image = frame
      image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
      image = letterbox(image, (frame_width), stride=64, auto=True)[0]
      image_ = image.copy()
      image = transforms.ToTensor()(image)
      image = torch.tensor(np.array([image.numpy()]))
      image = image.to(device)
      image = image.half()
 
      # Get the start time.
      start_time = time.time()
      with torch.no_grad():
          output, _ = model(image)

 
      output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
      output = output_to_keypoint(output)

      if output.shape[0]==0:
          frame_count+=1
          print('Could not detect any thing for q %s'%(frame_count))
          continue
      nimg = image[0].permute(1, 2, 0) * 255
      nimg = nimg.cpu().numpy().astype(np.uint8)
      nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
      multiple_person={}
      
      for idx in range(output.shape[0]):
          keypoints=output[idx, 7:].T
          plot_skeleton_kpts(nimg, keypoints, 3) 
          # Comment/Uncomment the following lines to show bounding boxes around persons.
          xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
          xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
          cv2.rectangle(nimg,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0),thickness=1,lineType=cv2.LINE_AA)
          each_person_kps_2d=dict(zip(list(all_kp_coord_names),list(keypoints)))
          get_3d_kps=inference_3d(cascade,stats,each_person_kps_2d,frame_count,cuda,LIMIT=1)
          multiple_person['person_'+str(idx+1)]=get_3d_kps
          df=pd.DataFrame(multiple_person).T
          fig=plot_skeleton_3d_kpts(df)
          img_3d_kp=get_img_from_fig(fig)
          plt.close('all')
          
      df=df.reset_index(level=0).rename(columns={'index':'persons'})
      df.insert(0, 'frame_no', frame_count)
      each_frame_dfs.append(df)
          
          
          
          
    # Get the end time.
      end_time = time.time()
      # Get the fps.
      fps = 1 / (end_time - start_time)
      # Add fps to total fps.
      total_fps += fps
      # Increment frame count.
      frame_count += 1
      # Write the FPS on the current frame.
      cv2.putText(nimg, "FPS: %s"%(fps), (15, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
 
      cv2.namedWindow("2d keypoints", cv2.WINDOW_NORMAL)
      cv2.resizeWindow("2d keypoints", 600, 600)
      cv2.moveWindow("2d keypoints", 10, 50)
      cv2.imshow('2d keypoints', nimg)
      
      cv2.namedWindow("3d keypoints", cv2.WINDOW_NORMAL)
      cv2.resizeWindow("3d keypoints", 600, 600)
      cv2.moveWindow("3d keypoints", 600, 50)
      cv2.imshow('3d keypoints', img_3d_kp)
      out.write(nimg)
      
      # Press `q` to exit.
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  else:
      break
# Release VideoCapture().
vidcap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")


# %%

df_3d=pd.concat(each_frame_dfs).reset_index(drop=True)

