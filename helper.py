import numpy as np
import torch
from libs.dataset.h36m.data_utils import unNormalizeData

# %%
def average_2_columns(data_df, column1 , column2, new_column_name): #returns a dataframe with a column with average values of 2 column
    data_df[new_column_name] = (data_df[column1]+data_df[column2])*0.5
    return data_df


def create_new_2d_keypoints(data_df):
    #data_df = pd.read_csv(data_file) #load the csv data of hrnet generated 2d keypoints

    #remove duplicates
    #data_df = remove_duplicates(data_df )

    #calculate 2d keypoint for head
    #df_1 = average_2_columns(data_df, 'eye_l_x', 'eye_r_x', 'avg_eye_x')
    #df_2 = average_2_columns(df_1, 'eye_l_y', 'eye_r_y', 'avg_eye_y')
    #df_3 = average_2_columns(df_2, 'ear_l_x', 'ear_r_x', 'avg_ear_x')
    #df_4 = average_2_columns(df_3, 'ear_l_y', 'ear_r_y', 'avg_ear_y')
    df_head_x  = average_2_columns(data_df, 'left_eye_x', 'right_eye_x', 'head_x')
    df_head_y  = average_2_columns(df_head_x, 'left_eye_y', 'right_eye_y', 'head_y')
    #df_head = df_head_y.drop(['avg_eye_x','avg_ear_x', 'avg_eye_y','avg_ear_y'], axis=1)

    #calculate 2d keypoint for thorax
    df_thorax_x = average_2_columns(df_head_y, 'left_shoulder_x','right_shoulder_x', 'thorax_x')
    df_thorax_y = average_2_columns(df_thorax_x, 'left_shoulder_y', 'right_shoulder_y', 'thorax_y')

    #calculate 2d keypoint for hip
    df_hip_x = average_2_columns(df_thorax_y, 'left_hip_x','right_hip_x', 'hip_x')
    df_hip_y = average_2_columns(df_hip_x, 'left_hip_y','right_hip_y', 'hip_y')

    #calculate 2d keypoint for spine
    df_spine_x_th = average_2_columns(df_hip_y, 'thorax_x', 'hip_x', 'spine_x') #spine calculated from thorax and hip
    df_spine_y_th = average_2_columns(df_spine_x_th, 'thorax_y', 'hip_y', 'spine_y')
    #df_spine_y_th.to_csv(output_file, index = False) #save the newly created 2d keypoints for evoskeleton in a csv file
    return df_spine_y_th


def create_data_dictionary(data_df,frame_no,limit=1):
    custom_data_df = data_df
    #limit = custom_data_df.shape[0]
    #print('Processing {} datapoints'.format(str(limit)))
    custom_data_dic = {}
    for i in ( range(limit)):
        #extract the data point from custom data's dataframe
        #datapoint = custom_data_df.iloc[i,:]
        datapoint = custom_data_df
        key = frame_no
        #print(key)
        keypoints = [] #holds the coordinates of keypoints in the order of evoskeleton dictionary data
        keypoints.append([datapoint['hip_x'], datapoint['hip_y']]) #adding hip keypoint
        #covering the right leg
        keypoints.append([datapoint['right_hip_x'], datapoint['right_hip_y']]) #adding right hip keypoint
        keypoints.append([datapoint['right_knee_x'], datapoint['right_knee_y']]) #adding right knee keypoint
        keypoints.append([datapoint['right_ankle_x'], datapoint['right_ankle_y']]) #adding ankle/foot keypoint
        
        #covering the left leg
        keypoints.append([datapoint['left_hip_x'], datapoint['left_hip_y']]) #adding left hip keypoint
        keypoints.append([datapoint['left_knee_x'], datapoint['left_knee_y']]) #adding left knee keypoint
        keypoints.append([datapoint['left_ankle_x'], datapoint['left_ankle_y']]) #adding left ankle/foot keypoint
        
        keypoints.append([datapoint['spine_x'], datapoint['spine_y']]) #adding spine keypoint
        keypoints.append([datapoint['thorax_x'], datapoint['thorax_y']]) #adding thorax keypoint
        keypoints.append([datapoint['nose_x'], datapoint['nose_y']]) #adding nose keypoint
        keypoints.append([datapoint['head_x'], datapoint['head_y']]) #adding head keypoint
        
        #covering left hand
        keypoints.append([datapoint['left_shoulder_x'], datapoint['left_shoulder_y']]) #adding left shoulder keypoint
        keypoints.append([datapoint['left_elbow_x'], datapoint['left_elbow_y']]) #adding left elbow keypoint
        keypoints.append([datapoint['left_wrist_x'], datapoint['left_wrist_y']]) #adding left wrist keypoint
        
        #covering right hand
        keypoints.append([datapoint['right_shoulder_x'], datapoint['right_shoulder_y']]) #adding right shoulder keypoint
        keypoints.append([datapoint['right_elbow_x'], datapoint['right_elbow_y']]) #adding right elbow keypoint
        keypoints.append([datapoint['right_wrist_x'], datapoint['right_wrist_y']]) #adding right wrist keypoint
        
        keypoints = np.asarray(keypoints)
        #print(keypoints.shape)
        #print(keypoints)
        #create a dictionary of the form key: 'p2d', value: keypoints array
        #keypoint_dic = {'p2d': keypoints, 'pose': datapoint['pose']}
        keypoint_dic = {'p2d': keypoints }
        
        #print(keypoint_dic)
        custom_data_dic[key] = keypoint_dic  
    #np.save(dictionary_save_file, custom_data_dic) #saving the data dictionary including the posename  
    return custom_data_dic


def normalize(skeleton, re_order=None):
    norm_skel = skeleton.copy()
    if re_order is not None:
        norm_skel = norm_skel[re_order].reshape(32)
    norm_skel = norm_skel.reshape(16, 2)
    mean_x = np.mean(norm_skel[:, 0])
    std_x = np.std(norm_skel[:, 0])
    mean_y = np.mean(norm_skel[:, 1])
    std_y = np.std(norm_skel[:, 1])
    denominator = (0.5*(std_x + std_y))
    norm_skel[:, 0] = (norm_skel[:, 0] - mean_x)/denominator
    norm_skel[:, 1] = (norm_skel[:, 1] - mean_y)/denominator
    norm_skel = norm_skel.reshape(32)
    return norm_skel


def get_pred(cascade, data):
    """
    Get prediction from a cascaded model
    """
    # forward pass to get prediction for the first stage
    num_stages = len(cascade)
    # for legacy code that does not have the num_blocks attribute
    for i in range(len(cascade)):
        cascade[i].num_blocks = len(cascade[i].res_blocks)
    prediction = cascade[0](data)
    # prediction for later stages
    for stage_idx in range(1, num_stages):
        prediction += cascade[stage_idx](data)
    return prediction


def create_3d_keypoint_dictionary(channels, image_name): #the output dictionary is used to create dataframe and saved in csv file
    keypoint_dictionary = {} #the dictionary holding joint coordinates
    vals = np.reshape(channels, (32, -1)) #(32,3)

    all_points_indices = np.array([1,2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1
    #The order for the 3D keypoints is:
    # 'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
    # 'Thorax', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder'
    # 'RElbow', 'RWrist'
    keypoint_names = np.array(['hip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle','spine', 'thorax', 'head', 'left_shoulder',
                            'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'])
    keypoint_coordinates = [vals[all_points_indices[index ]] for index in range(all_points_indices.shape[0])]
    #print('keypoint coords')
    #print(keypoint_coordinates)
    #create dictionary of the form {keypoint_name: keypoint coordinates}
    #keypoint_dictionary = dict(zip(keypoint_names, keypoint_coordinates))
    for i in range(len(keypoint_coordinates)):
        keypoint_name = keypoint_names[i]
        keypoint_coord = keypoint_coordinates[i]
        keypoint_dictionary[keypoint_name+'_x'] = keypoint_coord[0]
        keypoint_dictionary[keypoint_name+'_y'] = keypoint_coord[1]
        keypoint_dictionary[keypoint_name+'_z'] = keypoint_coord[2]

    #keypoint_dictionary['filename'] = image_name
    #print(keypoint_dictionary)

    return keypoint_dictionary
re_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

def inference_3d(cascade,stats,kp_dict,frame_num,IsCuda,LIMIT=1):
    kp_dict=create_new_2d_keypoints(kp_dict)
    data_dictionary = create_data_dictionary(kp_dict,frame_num,limit=LIMIT)
    keypoints_3d=[]
    for limit in (range((LIMIT))):
        image_name = (frame_num)
        #image_name=0
        ###Prediction starts here
        skeleton_2d = data_dictionary[image_name]['p2d'] #get 2d keypoint coordinates
        #normalize the input data before feeding into the network
        #re_order_indices
        norm_ske_gt = normalize(skeleton_2d, re_order_indices).reshape(1, -1)
        #print(norm_ske_gt.shape)
        #print(norm_ske_gt)
        
        #get predictions for the input data
        if IsCuda:
            pred = get_pred(cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)).cuda()) #.cuda()).detach().cpu().squeeze().numpy()
        else:pred = get_pred(cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))
        #print( print(pred)) #(1,48)
        
        #since the model output is in a normalized form, de-normalize it
        pred_denormalized, dimensions_to_use = unNormalizeData(pred.data.cpu().numpy(),
                            stats['mean_3d'],
                            stats['std_3d'],
                            stats['dim_ignore_3d']
                            )


        #print(print(pred_denormalized)) #(1,48)
        #create separate prediction with z axis inverted
        # inverted_prediction = pred_denormalized.copy()
        # inverted_prediction = inverted_prediction.reshape(32,3)
        # inverted_prediction[:, 1] *= -1
        # keypoints_3d_dict_inverted = create_3d_keypoint_dictionary(inverted_prediction, image_name)
        # keypoints_3d_inverted.append(keypoints_3d_dict_inverted)

        keypoints_3d_dict = create_3d_keypoint_dictionary(pred_denormalized, image_name)
        return keypoints_3d_dict
    
