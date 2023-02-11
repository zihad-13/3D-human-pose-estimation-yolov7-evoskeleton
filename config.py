import argparse



def gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data")
    parser.add_argument('--dimension', type=int, default=64, help='dimension of Generator output 3D shape object')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--weights', type=str, default="weights/yolov7-w6-pose.pt")
    parser.add_argument('--source', type=str, default="inputs/snowboarder.mp4")
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--output_vid_name', type=str, default="outputs")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--pretrained_model_path_3d', type=str, default='weights/example_model.th')
    parser.add_argument('--data_stats', type=str, default='weights/stats.npy')

    return parser
