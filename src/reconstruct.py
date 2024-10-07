import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import pandas as pd
import torch.nn as nn
from time import time 
import segmentation_models_pytorch as smp
import segmentation
from sfm.model import SfMModel
from segmentation.model import SegmentationModel
from video_utils import extract_frames_and_gopro_gravity_vector, render_video
from tqdm import tqdm
import h5py
import open3d as o3d
from sklearn.decomposition import PCA
from reconstruction_utils import get_closest_to_centroid_with_attributes_of_closest_to_cam, map_3d, get_matching_indices, get_rotation_matrix_to_align_pose_with_gravity, get_edgeness, aggregate_2d_grid
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from sfm.inverse_warp import EUCMCamera, Pose, pose_vec2mat, rectify_eucm
import scipy
import json

import sys
sys.path.append('/home/jonathan/mee-deepreefmap/src/ffmpeg-7.0.2-amd64-static/ffmpeg')

# Start by parsing args
parser = argparse.ArgumentParser(description='Reconstruct a 3D model from a video')
parser.add_argument('--input_video', type=str, help='Path to video file - can be multiple files, in which case the paths should be comma separated')
parser.add_argument('--out_dir', type=str, default='out', help='Path to output directory - will be created if does not exist')
parser.add_argument('--tmp_dir', type=str, default='tmp', help='Path to temporary directory - will be created if does not exist')
parser.add_argument('--timestamp', type=str, help='Begin and End timestamp of the transect. In case multiple videos are supplied, the format should be comma separated e.g. of the form "0:23-end,begin-1:44"')
parser.add_argument('--sfm_checkpoint', type=str, default='../sfm_net.pth', help='Path to the sfm_net checkpoint')
parser.add_argument('--segmentation_checkpoint', type=str, default='../segmentation_net.pth', help='Path to the segmentation_net checkpoint')
parser.add_argument('--height', type=int, default=384, help='Height in pixels to which input video is scaled')
parser.add_argument('--width', type=int, default=640, help='Width in pixels to which input video is scaled')
parser.add_argument('--seg_height', type=int, default=384*2, help='Height in pixels to which input video is scaled')
parser.add_argument('--seg_width', type=int, default=640*2, help='Width in pixels to which input video is scaled')
parser.add_argument('--fps', type=int, default=8, help='FPS of the input video')
parser.add_argument('--reverse', action='store_true', help='Whether the transect video is filmed backwards (with a back-facing camera)')
parser.add_argument('--number_of_points_per_image', type=int, default=2000, help='Number of points to sample from each image')
parser.add_argument('--frames_per_volume', type=int, default=500, help='Number of frames per TSDF Volume')
parser.add_argument('--tsdf_overlap', type=int, default=100, help='Overlap in frames over TSDF Volumes')
parser.add_argument('--distance_thresh', type=float, default=0.2, help='Distance threshold for points added to cloud')
parser.add_argument('--ignore_classes_in_point_cloud', type=str, default="background,fish,human", help='Classes to ignore when adding points to cloud')
parser.add_argument('--ignore_classes_in_benthic_cover', type=str, default="background,fish,human,transect tools,transect line,dark", help='Classes to ignore when calculating benthic cover percentages')
parser.add_argument('--intrinsics_file', type=str, default="../example_inputs/intrinsics_eucm.json", help='Path to intrinsics file')
parser.add_argument('--class_to_label_file', type=str, default="../example_inputs/class_to_label.json", help='Path to label_to_class_file')
parser.add_argument('--class_to_color_file', type=str, default="../example_inputs/class_to_color.json", help='Path to class_to_color_file')
parser.add_argument('--output_2d_grid_size', type=int, default=2000, help='Size of the 2D grid used for benthic cover analysis - a higher grid size will produce higher resolution outputs but takes longer to compute and may have empty grid cells')
parser.add_argument('--buffer_size', type=int, default=2, help='Number of frames to use for temporal smoothing')
parser.add_argument('--render_video', action='store_true', help='Whether to render output 4-panel video')
args = parser.parse_args()

def main(args):

    t = time()

    with open(args.class_to_color_file) as f:
        class_to_color = {k: (np.array(v)).astype(np.uint8) for k,v in json.load(f).items()}
    with open(args.class_to_label_file) as f:
        class_to_label = json.load(f)
        label_to_class = {v:k for k,v in class_to_label.items()}
    label_to_color = {k: class_to_color[v] for k,v in label_to_class.items()}
 

    grav = extract_frames_and_gopro_gravity_vector(
        args.input_video.split(","), 
        args.timestamp.split(","), 
        args.seg_width, 
        args.seg_height, 
        args.fps, 
        args.tmp_dir,
        args.reverse,
    )
    print("Extracted Frames And Gravity Vector in", time() - t, "seconds")
    

    h5f = h5py.File(args.tmp_dir + '/tmp.hdf5', 'w')
    
    img_list = [args.tmp_dir + "/rgb/" +file for file in sorted(os.listdir(args.tmp_dir + "/rgb")) if "jpg" in file]
    print("Running Neural Networks ...")
    
    depths, depth_uncertainties, poses, semantic_segmentation, intrinsics = get_nn_predictions(
        img_list, 
        grav, 
        len(class_to_label) + 1,
        h5f,
        args,
    )

    print("Ran NN Predictions in ", time() - t, "seconds")
    print("Building Point Cloud ...")
    os.makedirs(args.out_dir + "/videos", exist_ok=True)
    xyz_index_arr, distance2cam_arr, seg_arr, frame_index_arr, depth_unc_arr, keep_masks, dist_cutoffs = get_point_cloud(
        img_list, 
        depths, 
        poses, 
        depth_uncertainties, 
        semantic_segmentation, 
        intrinsics,
        label_to_color, 
        class_to_label,
        h5f,
        args
    )

    print("Integrating TSDF!")
    tsdf_xyz, tsdf_rgb = tsdf_point_cloud(img_list, depths, keep_masks, poses, intrinsics, np.mean(depths), args.frames_per_volume, args.tsdf_overlap, dist_cutoffs)
    print("Integrated TSDF Point Cloud in ", time() - t, "seconds")

    idx = get_matching_indices(tsdf_xyz, xyz_index_arr)
    print("Matched TSDF to Point Cloud in ", time() - t, "seconds")
    rgb_seg_arr = np.vectorize(lambda k: label_to_color[k], signature='()->(n)')(seg_arr[idx])
    tsdf_pc = pd.DataFrame({
        'x':tsdf_xyz[:,0],
        'y':tsdf_xyz[:,1],
        'z':tsdf_xyz[:,2],
        'r':tsdf_rgb[:,0],
        'g':tsdf_rgb[:,1],
        'b':tsdf_rgb[:,2],
        'distance_to_cam': distance2cam_arr[idx],
        'class': seg_arr[idx],
        'class_r': rgb_seg_arr[:,0],
        'class_g': rgb_seg_arr[:,1],
        'class_b': rgb_seg_arr[:,2], 
        'frame_index': frame_index_arr[idx],
        'depth_uncertainty': depth_unc_arr[idx],
    }) 
    tsdf_pc.to_csv(args.out_dir + "/point_cloud_tsdf.csv", index=False)
    print("Saved TSDF Point Cloud in ", time() - t, "seconds")


    print("Starting Benthic Cover Analsysis after ", time() - t, "seconds")
    results, percentage_covers = benthic_cover_analysis(tsdf_pc, label_to_class, args.ignore_classes_in_benthic_cover.split(","), bins=args.output_2d_grid_size)
    np.save(args.out_dir + "/results.npy", results)
    json.dump(percentage_covers, open(args.out_dir + "/percentage_covers.json", "w"))
    print("Finished Benthic Cover Analysis in ", time() - t, "seconds")

    os.system("cp "+args.class_to_color_file+" "+ args.out_dir)
    if args.render_video:
        os.system("cp "+args.tmp_dir+"/*_.mp4 "+ args.out_dir + "/videos")
        render_video(img_list, depths, semantic_segmentation, results, args.fps, class_to_label, label_to_color, args.tmp_dir, args.reverse)
        os.system("mv " + args.tmp_dir + "/out.mp4 " + args.out_dir + "/videos")
        print("Rendered Video in ", time() - t, "seconds")
    return 



def reset_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eps = 1e-4

def change_bn_momentum(model, new_value):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = new_value


def get_depths_and_poses(encoder, segmentation_head, decoder, pose_decoder, images, features_, reduction, squeeze_unsqueeze):
    b, l, c, h, w = images.shape

    ref_features = [x[:b] for x in features_]

    features = []
    lf = len(features_)
    _, c_feat, h_feat, w_feat = features_[-1].shape

    for i in range(lf):
        if i == lf - 1:
            features.append(
                features_[-1][b:] + squeeze_unsqueeze(torch.cat([
                            features_[-1][b:], 
                            ref_features[-1].reshape(b,1,c_feat,h_feat,w_feat).expand(b, l, c_feat, h_feat, w_feat).reshape(b*l,c_feat, h_feat, w_feat)
                ], dim=1)))
        else:
            features.append(features_[i][b:])


    depths = segmentation_head(decoder(*features)).reshape(b, l, 1, h, w) 
    
    _, c_feat, h_feat, w_feat = features[-1].shape
    last_feat = features[-1]

    last_feat = reduction(last_feat.reshape(b* l, c_feat, h_feat, w_feat))
    c_feat = last_feat.shape[1]
    last_feat = last_feat.reshape(b, l, c_feat, h_feat, w_feat)

    
    last_feat = last_feat.unsqueeze(2).expand(b, l, l, c_feat, h_feat, w_feat).contiguous()
    features_sq = torch.cat([last_feat, last_feat.transpose(1, 2)], dim=3).reshape(b*l*l, c_feat*2, h_feat, w_feat)


    poses = pose_decoder(features_sq)
    poses = poses.mean(dim=(2,3)).reshape(b,l,l,6)

    return depths, poses * 0.01


def expand_zeros(mask):
    # Add an extra batch dimension and channel dimension to the mask for convolution
    mask = mask.unsqueeze(0).unsqueeze(0).float()  # Shape: 1x1xHxW

    # Define a 3x3 kernel filled with ones
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)

    # Perform 2D convolution with padding=1 to keep the same output size
    conv_result = torch.nn.functional.conv2d(mask, kernel, padding=1)

    # Any place where the convolution result is less than 9 means it had a zero in the neighborhood
    result_mask = (conv_result == 9).squeeze().bool()

    return result_mask

def get_nn_predictions(img_list, grav, num_classes, h5f, args):
    totensor = torchvision.transforms.ToTensor()
    normalize = torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                std=[0.225, 0.225, 0.225])
    sfm_model = SfMModel().to(device)
    sfm_model.load_state_dict(torch.load(args.sfm_checkpoint))
    change_bn_momentum(sfm_model, 0.01)
    reset_batchnorm_layers(sfm_model)
    sfm_model.eval()

    segmentation_model = SegmentationModel(num_classes).to(device)
    segmentation_model.load_state_dict(torch.load(args.segmentation_checkpoint))
    segmentation_model.eval()
    
    intrinsics = torch.tensor(list(json.load(open(args.intrinsics_file)).values())).float().to(device).unsqueeze(0)

    buffer_size = args.buffer_size
    # Start by initializing, load the images in a buffer of buffer_size subsequent frames
    images = [normalize(totensor(Image.open(img_list[i]))).to(device).unsqueeze(0) for i in range(buffer_size-1)]
    
    counts = np.zeros(len(img_list), dtype=np.uint8)
    depths_buffered = h5f.create_dataset("depths_buffered", (args.buffer_size, len(img_list), args.height, args.width), dtype='f4')
    depths = h5f.create_dataset("depths", (len(img_list), args.height,  args.width), dtype='f4')
    depth_uncertainties = h5f.create_dataset("depth_uncertainties", (len(img_list), args.height,  args.width), dtype='f4')
    semantic_segmentation = h5f.create_dataset("semantic_segmentation", (len(img_list), args.height,  args.width), dtype='u1')
    intrinsics_predicted_buffered = h5f.create_dataset("intrinsics_buffered", (args.buffer_size, len(img_list), 6), dtype='f4')
    intrinsics_predicted = h5f.create_dataset("intrinsics", (len(img_list), 6), dtype='f4')
    poses = {}

    semseg_buffer = torch.zeros((3, num_classes, args.height, args.width), requires_grad=False).to(device)
    wtens = torch.tensor([1.0, 2.0, 1.0], requires_grad=False).to(device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    with torch.no_grad():
        semseg_logits = []
        for i in range(buffer_size-1):
            semseg_logits.append(segmentation.model.predict(segmentation_model, images[i], num_classes, args.height, args.width))

        
        for i in range(buffer_size-2):
            semantic_segmentation[i] = torch.stack(semseg_logits[max(0,i-1):i+1]).mean(dim=0).argmax(dim=0).cpu().numpy()
        
        if len(semseg_logits)==1:
            semseg_logits.append(semseg_logits[-1])
        semseg_buffer[0] = semseg_logits[-2]
        semseg_buffer[1] = semseg_logits[-1]
        del semseg_logits
    images = [F.resize(x, (args.height, args.width)) for x in images]
    depth_features = [[f.detach() for f in sfm_model.extract_features(x)] for x in images]
    

    for end_index in tqdm(range(buffer_size-1, len(img_list))):
        new_im = normalize(totensor(Image.open(img_list[end_index]))).to(device).unsqueeze(0)

        
        with torch.no_grad():
            semseg_buffer[2] = segmentation.model.predict(segmentation_model, new_im, num_classes, args.height, args.width)
            semantic_segmentation[end_index-1] = (semseg_buffer*wtens).mean(dim=0).argmax(dim=0).cpu().numpy()
            semseg_buffer[:2] = semseg_buffer[1:].clone()

        images.append(F.resize(new_im, (args.height,  args.width)))

        with torch.no_grad():
            depth_features.append([f.detach() for f in sfm_model.extract_features(images[-1])])

        depth, pose, intrinsics_updated =  sfm_model.get_depth_and_poses_from_features(images, depth_features, intrinsics)
        for i in range(buffer_size):
            idx = end_index - buffer_size + i + 1
            count = counts[idx]
            depths_buffered[count, idx] = depth[i].squeeze().detach().cpu().numpy()
            intrinsics_predicted_buffered[count, idx] = intrinsics_updated[i].detach().cpu().numpy()
            counts[idx]+=1
            for j in range(buffer_size):
                jdx = end_index - buffer_size + j + 1
                if pose[i][j] != []:
                    if idx not in poses:
                        poses[idx] = {}
                    if jdx not in poses[idx]:
                        poses[idx][jdx] = []
                    poses[idx][jdx].append(pose[i][j].detach().unsqueeze(0).cpu().numpy())


        images.pop(0)
        depth_features.pop(0)
    
    for i in tqdm(range(len(img_list))):
        depths[i] = np.mean(depths_buffered[:counts[i],i],axis=0)
    for i in tqdm(range(len(img_list))):
        depth_uncertainties[i] = np.std(depths_buffered[:counts[i],i],axis=0)
    for i in tqdm(range(len(img_list))):
        intrinsics_predicted[i] = np.median(intrinsics_predicted_buffered[:counts[i],i],axis=0)
        
    l = len(img_list)

    semantic_segmentation[-1] = (semseg_buffer * wtens).mean(dim=0).argmax(dim=0).cpu().numpy()
    
    poses = [(torch.tensor(np.median(poses[i+1][i],axis=0) - np.median(poses[i][i+1],axis=0))/2) for i in range(len(poses)-1)]
    poses = [pose_vec2mat(p).squeeze().cpu().numpy() for p in poses]
    poses = [np.vstack([p, np.array([0, 0, 0, 1]).reshape(1, 4)]) for p in poses]

    poses = np.array(poses)
    med_rot = np.median(poses[:,:3,:3]-np.eye(3), axis=0)
    poses[:,:3,:3] -= med_rot 
    
    grav_buffer = 100
    if grav is not None:
        pose0 = np.eye(4)
        new_cum_poses = np.zeros((len(poses)+1,4,4))


        grav0 = np.mean(grav[:grav_buffer],axis=0)

        correction = get_rotation_matrix_to_align_pose_with_gravity(pose0, grav0)
        pose0[:3,:3] = correction @ pose0[:3,:3] 
        new_cum_poses[0] = pose0.copy()

        for i, (pose, g_) in enumerate(zip(poses,grav[1:])):

            g = np.mean(grav[max(0, 1 + i - grav_buffer):min(i + grav_buffer, len(grav-1))], axis=0)
            pose0 = pose0 @ pose
            correction = get_rotation_matrix_to_align_pose_with_gravity(pose0, g)
            pose0[:3,:3] = correction @ pose0[:3,:3] 
            new_cum_poses[i+1] = pose0.copy()
        poses = np.array(new_cum_poses)
    else:
        new_cum_poses = np.zeros((len(poses)+1,4,4))
        new_cum_poses[0] = np.eye(4)
        for i in range(len(poses)):
            new_cum_poses[i+1] = new_cum_poses[i] @ poses[i]
        poses = np.array(new_cum_poses)

    return depths, depth_uncertainties, poses, semantic_segmentation, intrinsics_predicted


def get_point_cloud(image_list, depths, poses, depth_uncertainties, semantic_segmentation, intrinsics, label_to_color, class_to_label, h5f, args):
    ignore_classes = args.ignore_classes_in_point_cloud.split(",")

    def class_to_color(class_arr):
        color_arr = np.zeros((class_arr.shape[0], 3), dtype=np.uint8)
        for val in np.unique(class_arr):
            color_arr[class_arr==val] = label_to_color[val]
        return color_arr
    dist_cutoffs = []
    with torch.no_grad(): 
        

        xyz_arr = h5f.create_dataset("xyz_arr", (len(image_list)*args.number_of_points_per_image, 3), dtype='f4')
        distance2cam_arr = h5f.create_dataset("distance2cam_arr", (len(image_list)*args.number_of_points_per_image), dtype='f4')
        seg_arr = h5f.create_dataset("seg_arr", (len(image_list)*args.number_of_points_per_image), dtype='u1')
        keep_masks = h5f.create_dataset("keep_masks", (len(image_list),  args.height,  args.width), dtype='u1')
        depth_unc_arr = h5f.create_dataset("depth_unc_arr", (len(image_list)*args.number_of_points_per_image), dtype='f4')
        frame_index_arr = h5f.create_dataset("frame_index_arr", (len(image_list)*args.number_of_points_per_image), dtype='u2')

        cursor = 0

        for i in tqdm(range(len(poses))):

            pose =  torch.tensor((poses[i])[:3]).float().to(device)


            cam = EUCMCamera(torch.tensor(intrinsics[i]).unsqueeze(0).to(device), Tcw=Pose(T=1))
            depth_i_tensor = torch.tensor(depths[i]).to(device)
            coords = cam.reconstruct_depth_map(depth_i_tensor.unsqueeze(0).unsqueeze(0).to(device)).squeeze()
            coords = coords.reshape(3, -1)
            coords = (pose @ torch.cat([coords, torch.ones_like(coords[:1])], dim=0).reshape(4,-1)).T.cpu()


            dist_cutoffs.append(args.distance_thresh)
            keep_mask = depth_i_tensor.squeeze() < args.distance_thresh
            seg = torch.tensor(semantic_segmentation[i]).to(device)

            #Exclude points on the 'edge' of objects, i.e. where the countours depth map varies a lot
            # TODO Magic Numbers
            keep_mask = torch.logical_and(get_edgeness(depth_i_tensor) < 0.04, keep_mask)
            keep_mask[30:170,30:-30] = 0
            for class_name in ignore_classes:
                keep_mask = torch.logical_and(seg != class_to_label[class_name], keep_mask)
            keep_mask = expand_zeros(keep_mask)
            keep_mask = keep_mask.cpu().numpy()
            keep_masks[i] = keep_mask.astype(np.uint8)
            keep_mask = keep_mask.reshape(-1)            
            valid_points = keep_mask.sum().item()
            random_selection = np.random.permutation(valid_points)[:args.number_of_points_per_image]
            offset = min(valid_points, args.number_of_points_per_image)

            xyz_arr[cursor:cursor+offset]=coords[keep_mask][random_selection]
            distance2cam_arr[cursor:cursor+offset]=  depths[i].reshape(-1)[keep_mask][random_selection]


            seg_arr[cursor:cursor+offset] = semantic_segmentation[i].reshape(-1).astype(np.uint8)[keep_mask][random_selection]
            dunc = depth_uncertainties[i].reshape(-1)[keep_mask][random_selection]
            depth_unc_arr[cursor:cursor+offset]=dunc
            frame_index_arr[cursor:cursor+offset]= np.zeros_like(dunc, dtype=np.uint16)+i

            cursor += offset
        


        print("Filtering redundant points")
        xyz_index_arr = map_3d(np.concatenate([
            xyz_arr[:cursor], 
            distance2cam_arr[:cursor].reshape(-1,1), 
            np.arange(len(xyz_arr)).reshape(-1, 1)[:cursor]], axis=1), get_closest_to_centroid_with_attributes_of_closest_to_cam, 0.003)
        filtered_indices = xyz_index_arr[:, -1].astype(np.uint32)

        return xyz_index_arr[:,:3], distance2cam_arr[:cursor][filtered_indices], seg_arr[:cursor][filtered_indices], frame_index_arr[:cursor][filtered_indices], depth_unc_arr[:cursor][filtered_indices], keep_masks, dist_cutoffs
        
    
def tsdf_point_cloud(img_list, depths, masks, poses, intrinsics, cutoff, frames_per_volume, tsdf_overlap,dist_cutoffs): 
    # TODO Magic Numbers
    xyz = []
    rgb = []


    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.3 / 512.0, 
        sdf_trunc=0.035,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    totensor = torchvision.transforms.ToTensor()

    mask_out_background = np.ones_like(masks[0].astype(np.float32))
    intrinsics = torch.tensor(intrinsics).float()

    #TODO: Magic numbers
    mask_out_background[:170,80:-80] *= 0 
    for i in tqdm(range(len(poses))):
    
        if i > len(poses)-10:
            mask_out_background = np.ones_like(masks[0])

        # Rectify to linear intrinsics
        projected_img, projected_mask, projected_depth = rectify_eucm(
            totensor(Image.open(img_list[i])).unsqueeze(0), 
            torch.tensor(masks[i].astype(np.float32)*mask_out_background).unsqueeze(0).unsqueeze(0).float(),
            torch.tensor(depths[i]).unsqueeze(0).unsqueeze(0), 
            intrinsics[i]
        )

        depth = o3d.geometry.Image(projected_depth*projected_mask)
        color = o3d.geometry.Image(np.ascontiguousarray(projected_img.transpose(1, 2, 0)*255.).astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=dist_cutoffs[i], convert_rgb_to_intensity=False, depth_scale=1)
        
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
            width= args.width,
            height= args.height,
            fx=intrinsics[i][0],
            fy=intrinsics[i][1],
            cx=intrinsics[i][2],
            cy=intrinsics[i][3],
        ),
            np.linalg.inv(poses[i]))
        if (i % frames_per_volume) == (frames_per_volume - tsdf_overlap):
            volume2 = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.3 / 512.0, 
                sdf_trunc=0.015,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

        if i % frames_per_volume >= (frames_per_volume - tsdf_overlap):
            volume2.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                width= args.width,
                height= args.height,
                fx=intrinsics[i][0],
                fy=intrinsics[i][1],
                cx=intrinsics[i][2],
                cy=intrinsics[i][3],
            ),
            np.linalg.inv(poses[i]))
        if (i % frames_per_volume) == frames_per_volume - 1:
            pc = volume.extract_point_cloud()
            pc = volume.extract_point_cloud()
            xyz.append(np.array(pc.points))
            rgb.append((np.array(pc.colors)*255).astype(np.uint8))
            volume = volume2
    pc = volume.extract_point_cloud()
    pc = volume.extract_point_cloud()
    xyz.append(np.array(pc.points))
    rgb.append((np.array(pc.colors)*255).astype(np.uint8))
    return np.concatenate(xyz), np.concatenate(rgb)
 

def benthic_cover_analysis(pc, label_to_class, ignore_classes_in_benthic_cover, bins=1000):
    #step 1: fit PCA
    pca = PCA(n_components=2)
    pca.fit(pc[['x','y', 'z']].values)  
    x_axis = pca.components_[0]  # Estimated x-axis
    y_axis = pca.components_[1]  # Estimated y-axis

    # Step 2: Calculate the normal vector to the x-y plane as the z-axis
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Step 3: Create the transformation matrix
    transformation_matrix = np.vstack((x_axis, y_axis, z_axis)).T

    # Now, you can apply this transformation matrix to your point cloud
    transformed = np.dot(pc[['x','y', 'z']].values, transformation_matrix)  
    transformed -= np.min(transformed, axis=0)
    xmax, ymax, zmax = np.max(transformed, axis=0)
    
    discretization = xmax / bins
    pcarr = np.concatenate([transformed, pc.drop(columns=["x", "y", "z"]).values], axis=1)
    out = aggregate_2d_grid(pcarr, size=discretization)

    xcoords = out[:,0].astype(np.int32)
    ycoords = out[:,1].astype(np.int32)

    img = np.zeros((xcoords.max()+1, ycoords.max()+1, 12))
    
    img[xcoords, ycoords] = out[:,2:]
    
    percentage_covers = {}
    benthic_class = out[:,7].astype(np.uint8)
    all_classes = (benthic_class!=0)
    for class_label, class_name in label_to_class.items():
        if class_name not in ignore_classes_in_benthic_cover:
            percentage_covers[class_name] = (benthic_class==class_label).sum() 
        else:
            all_classes = np.logical_and(all_classes, benthic_class!=class_label)
    all_classes = all_classes.sum()
    percentage_covers = {k: v / all_classes for k,v in percentage_covers.items()}
    return img, percentage_covers

if __name__ == "__main__":
    main(args)