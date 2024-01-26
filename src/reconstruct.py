import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import pandas as pd
from time import time 
from sfm.model import SfMModel
from segmentation.model import SegmentationModel
from video_utils import extract_frames_and_gopro_gravity_vector, render_video
from tqdm import tqdm
import open3d as o3d
from sklearn.decomposition import PCA
from reconstruction_utils import get_closest_to_centroid_with_attributes_of_closest_to_cam, map_3d, get_matching_indices, get_rotation_matrix_to_align_pose_with_gravity, get_edgeness, aggregate_2d_grid
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from sfm.inverse_warp import UCMCamera, Pose, pose_vec2mat, rectify
import json

# Start by parsing args
parser = argparse.ArgumentParser(description='Reconstruct a 3D model from a video')
parser.add_argument('--input_video', type=str, help='Path to video file - can be multiple files, in which case the paths should be comma separated')
parser.add_argument('--out_dir', type=str, default='out', help='Path to output directory - will be created if does not exist')
parser.add_argument('--tmp_dir', type=str, default='tmp', help='Path to temporary directory - will be created if does not exist')
parser.add_argument('--timestamp', type=str, help='Begin and End timestamp of the transect. In case multiple videos are supplied, the format should be comma separated e.g. of the form "0:23-end,begin-1:44"')
parser.add_argument('--sfm_checkpoint', type=str, default='../checkpoints/sfm_net.pth', help='Path to the sfm_net checkpoint')
parser.add_argument('--segmentation_checkpoint', type=str, default='../checkpoints/segmentation_net.pth', help='Path to the segmentation_net checkpoint')
parser.add_argument('--height', type=int, default=352, help='Height in pixels to which input video is scaled')
parser.add_argument('--width', type=int, default=608, help='Width in pixels to which input video is scaled')
parser.add_argument('--fps', type=int, default=20, help='FPS of the input video')
parser.add_argument('--number_of_points_per_image', type=int, default=1000, help='Number of points to sample from each image')
parser.add_argument('--distance_thresh', type=float, default=0.5, help='Distance threshold for points added to cloud')
parser.add_argument('--ignore_classes_in_point_cloud', type=str, default="background,fish,human", help='Classes to ignore when adding points to cloud')
parser.add_argument('--ignore_classes_in_benthic_cover', type=str, default="background,fish,human,transect tools,transect line,dark", help='Classes to ignore when calculating benthic cover percentages')
parser.add_argument('--intrinsics_file', type=str, default="../example_inputs/intrinsics.json", help='Path to intrinsics file')
parser.add_argument('--class_to_label_file', type=str, default="../example_inputs/class_to_label.json", help='Path to label_to_class_file')
parser.add_argument('--class_to_color_file', type=str, default="../example_inputs/class_to_color.json", help='Path to class_to_color_file')
parser.add_argument('--output_2d_grid_size', type=int, default=1000, help='Size of the 2D grid used for benthic cover analysis - a higher grid size will produce higher resolution outputs but takes longer to compute and may have empty grid cells')
parser.add_argument('--buffer_size', type=int, default=2, help='Number of frames to use for temporal smoothing')
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
        args.width, 
        args.height, 
        args.fps, 
        args.tmp_dir
    )
    print("Extracted Frames And Gravity Vector in", time() - t, "seconds")
        
    img_list = [args.tmp_dir + "/rgb/" +file for file in sorted(os.listdir(args.tmp_dir + "/rgb")) if "jpg" in file]
    print("Running Neural Networks ...")
    depths, depth_uncertainties, poses, semantic_segmentation, intrinsics = get_nn_predictions(
        img_list, 
        grav, 
        len(class_to_label) + 1,
        args,
    )
    print("Ran NN Predictions in ", time() - t, "seconds")
    print("Building Point Cloud ...")
    os.makedirs(args.out_dir + "/videos", exist_ok=True)
    point_cloud, keep_masks = get_point_cloud(
        img_list, 
        depths, 
        poses, 
        depth_uncertainties, 
        semantic_segmentation, 
        intrinsics,
        label_to_color, 
        class_to_label,
        args
    )
    print("Made Point Cloud in ", time() - t, "seconds")
    keep_masks = np.array(keep_masks).astype(np.uint8)
    
    point_cloud.to_csv(args.out_dir + "/point_cloud.csv", index=False)
    print("Saved Point Cloud in ", time() - t, "seconds")

    print("Integrating TSDF!")
    tsdf_xyz, tsdf_rgb = tsdf_point_cloud(img_list, depths, keep_masks, poses, intrinsics, cutoff=np.mean(depths))
    print("Integrated TSDF Point Cloud in ", time() - t, "seconds")

    idx = get_matching_indices(tsdf_xyz, point_cloud[['x', 'y', 'z']].values)
    print("Matched TSDF to Point Cloud in ", time() - t, "seconds")
    tsdf_pc = pd.DataFrame({
        'x':tsdf_xyz[:,0],
        'y':tsdf_xyz[:,1],
        'z':tsdf_xyz[:,2],
        'r':tsdf_rgb[:,0],
        'g':tsdf_rgb[:,1],
        'b':tsdf_rgb[:,2],
        'distance_to_cam': point_cloud['distance_to_cam'].values[idx],
        'class': point_cloud['class'].values[idx],
        'class_r': point_cloud['class_r'].values[idx],
        'class_g': point_cloud['class_g'].values[idx],
        'class_b': point_cloud['class_b'].values[idx], 
        'frame_index': point_cloud['frame_index'].values[idx],
        'depth_uncertainty': point_cloud['depth_uncertainty'].values[idx],
    }) 
    tsdf_pc.to_csv(args.out_dir + "/point_cloud_tsdf.csv", index=False)
    print("Saved TSDF Point Cloud in ", time() - t, "seconds")


    print("Starting Benthic Cover Analsysis after ", time() - t, "seconds")
    results, percentage_covers = benthic_cover_analysis(tsdf_pc, label_to_class, args.ignore_classes_in_benthic_cover.split(","), bins=args.output_2d_grid_size)
    np.save(args.out_dir + "/results.npy", results)
    json.dump(percentage_covers, open(args.out_dir + "/percentage_covers.json", "w"))
    print("Finished Benthic Cover Analysis in ", time() - t, "seconds")

    os.system("cp "+args.tmp_dir+"/*_.mp4 "+ args.out_dir + "/videos")
    render_video(img_list, depths, semantic_segmentation, results, args.fps, class_to_label, label_to_color, args.tmp_dir)
    os.system("mv " + args.tmp_dir + "/out.mp4 " + args.out_dir + "/videos")
    print("Rendered Video in ", time() - t, "seconds")



def get_nn_predictions(img_list, grav, num_classes, args):
    totensor = torchvision.transforms.ToTensor()
    normalize = torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                std=[0.225, 0.225, 0.225])
    sfm_model = SfMModel().to(device)
    sfm_model.load_state_dict(torch.load(args.sfm_checkpoint))
    sfm_model.eval()

    segmentation_model = SegmentationModel(num_classes).to(device)
    segmentation_model.load_state_dict(torch.load(args.segmentation_checkpoint))
    segmentation_model.eval()
    
    intrinsics = torch.tensor(list(json.load(open(args.intrinsics_file)).values())).float().to(device).unsqueeze(0)

    buffer_size = args.buffer_size
    # Start by initializing, load the images in a buffer of buffer_size subsequent frames
    images = [normalize(totensor(Image.open(img_list[i]))).to(device).unsqueeze(0) for i in range(buffer_size)]

    semantic_segmentation = [segmentation_model(images[i]).argmax(dim=1).squeeze().detach().cpu().numpy() for i in range(buffer_size)]

    depth_features = [[f.detach() for f in sfm_model.extract_features(x)] for x in images]

    depths = {}
    depth_uncertainties = {}
    poses = {}
    intrinsics_predicted = {}

    
    for end_index in tqdm(range(buffer_size, len(img_list))):
        depth, pose, intrinsics_updated =  sfm_model.get_depth_and_poses_from_features(images, depth_features, intrinsics)
        for i in range(buffer_size):
            if end_index - buffer_size + i not in depths:
                depths[end_index - buffer_size + i] = []
                intrinsics_predicted[end_index - buffer_size + i] = []

            depths[end_index - buffer_size + i].append(depth[i].detach().cpu().numpy())
            intrinsics_predicted[end_index - buffer_size + i].append(intrinsics_updated[i].detach().cpu().numpy())
            if len(depths[end_index - buffer_size + i]) == buffer_size:
                depth_uncertainties[end_index - buffer_size + i] = np.std(depths[end_index - buffer_size + i], axis=0)[0][0]
                depths[end_index - buffer_size + i] = np.median(depths[end_index - buffer_size + i], axis=0)[0][0]
                intrinsics_predicted[end_index - buffer_size + i] = np.median(intrinsics_predicted[end_index - buffer_size + i], axis=0)[0]

            ii = end_index - buffer_size + i
            for j in range(buffer_size):
                jj = end_index - buffer_size + j
                if pose[i][j] != []:
                    if ii not in poses:
                        poses[ii] = {}
                    if jj not in poses[ii]:
                        poses[ii][jj] = []
                    poses[ii][jj].append(pose[i][j].detach().cpu().numpy())

        images.pop(0)
        depth_features.pop(0)
        new_im = normalize(totensor(Image.open(img_list[end_index]))).to(device).unsqueeze(0)
        with torch.no_grad():
            semantic_segmentation.append(segmentation_model(new_im).argmax(dim=1).squeeze().detach().cpu().numpy())

        images.append(new_im)

        with torch.no_grad():
            depth_features.append([f.detach() for f in sfm_model.extract_features(images[-1])])
        
    for i in range(buffer_size):
        if type(depths[i]) == list:
            depth_uncertainties[i] = np.std(depths[i], axis=0)[0][0]
            depths[i] = np.median(depths[i], axis=0)[0][0]
            intrinsics_predicted[i] = np.median(intrinsics_predicted[i], axis=0)[0]

        if type(depths[end_index - i - 1]) == list:
            depth_uncertainties[end_index - i - 1] = np.std(depths[end_index - i - 1], axis=0)[0][0]
            depths[end_index - i - 1] = np.median(depths[end_index - i - 1], axis=0)[0][0]
            intrinsics_predicted[end_index - i - 1] = np.median(intrinsics_predicted[end_index - i - 1], axis=0)[0]
    
    poses = [(torch.tensor(np.median(poses[i+1][i],axis=0) - np.median(poses[i][i+1],axis=0))/2) for i in range(len(poses)-1)]
    poses = [pose_vec2mat(p).squeeze().cpu().numpy() for p in poses]
    poses = [np.vstack([p, np.array([0, 0, 0, 1]).reshape(1, 4)]) for p in poses]

    poses = np.array(poses)

    if grav is not None:
        pose0 = np.eye(4)
        new_cum_poses = np.zeros((len(poses)+1,4,4))
        correction = get_rotation_matrix_to_align_pose_with_gravity(pose0, grav[0])
        pose0[:3,:3] = correction @ pose0[:3,:3] 
        new_cum_poses[0] = pose0.copy()
        for i, (pose, g) in enumerate(zip(poses,grav[1:])):
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

    intrinsics_predicted = np.array([intrinsics_predicted[i] for i in depths.keys()])
    return np.array(list(depths.values())), np.array(list([depth_uncertainties[i] for i in depths.keys()])), poses, np.array(semantic_segmentation), intrinsics_predicted


def get_point_cloud(image_list, depths, poses, depth_uncertainties, semantic_segmentation, intrinsics, label_to_color, class_to_label, args):
    ignore_classes = args.ignore_classes_in_point_cloud.split(",")

    def class_to_color(class_arr):
        color_arr = np.zeros((class_arr.shape[0], 3), dtype=np.uint8)
        for val in np.unique(class_arr):
            color_arr[class_arr==val] = label_to_color[val]
        return color_arr

    with torch.no_grad(): 
        
        rgb_arr = []
        xyz_arr = []
        distance2cam_arr = []
        seg_arr = []
        keep_masks = []
        depth_unc_arr = []
        frame_index_arr = []

        for i in tqdm(range(len(poses))):

            pose =  torch.tensor((poses[i])[:3]).float()

            cam = UCMCamera(torch.tensor(intrinsics[i]).unsqueeze(0).to(device), Tcw=Pose(T=1))
            depth_i_tensor = torch.tensor(depths[i])
            coords = cam.reconstruct_depth_map(depth_i_tensor.unsqueeze(0).unsqueeze(0).to(device)).squeeze().cpu()
            coords = coords.reshape(3, -1)
            coords = (pose @ torch.cat([coords, torch.ones_like(coords[:1])], dim=0).reshape(4,-1)).T

            keep_mask = depth_i_tensor.squeeze() < args.distance_thresh
            seg = torch.tensor(semantic_segmentation[i])

            #Exclude points on the 'edge' of objects, i.e. where the countours depth map varies a lot
            # TODO Magic Numbers
            keep_mask = torch.logical_and(get_edgeness(depth_i_tensor) < 0.04, keep_mask)
    
            for class_name in ignore_classes:
                keep_mask = torch.logical_and(seg != class_to_label[class_name], keep_mask)

            keep_masks.append(keep_mask.numpy())

            keep_mask = keep_mask.reshape(-1)            
            random_selection = np.random.permutation(keep_mask.sum().item())[:args.number_of_points_per_image]

            imarr = np.array(Image.open(image_list[i]),dtype=np.uint8)
            rgb_arr.append(imarr.reshape(-1, 3)[keep_mask][random_selection])
            xyz_arr.append(coords[keep_mask][random_selection])
            distance2cam_arr.append(depths[i].reshape(-1, 1)[keep_mask][random_selection])


            seg_arr.append(semantic_segmentation[i].reshape(-1, 1).astype(np.uint8)[keep_mask][random_selection])
            dunc = depth_uncertainties[i].reshape(-1, 1)[keep_mask][random_selection]
            depth_unc_arr.append(dunc)
            frame_index_arr.append(np.zeros_like(dunc, dtype=np.uint16)+i)
        
        # TODO
        #if inner_iterations > 0:
        #    print("Filtering uncertain points")
        #    unc_thresh = np.quantile(point_cloud_arr[:, -1], global_unc_quant)
        #    point_cloud_arr = point_cloud_arr[point_cloud_arr[:, -1] < unc_thresh]


        print("Filtering redundant points")
        # TODO Magic Numbers
        xyz_arr = np.concatenate(xyz_arr, axis=0)
        distance2cam_arr = np.concatenate(distance2cam_arr, axis=0)
        xyz_index_arr = map_3d(np.concatenate([xyz_arr, distance2cam_arr, np.arange(len(xyz_arr)).reshape(-1, 1)], axis=1), get_closest_to_centroid_with_attributes_of_closest_to_cam, 0.008)
        filtered_indices = xyz_index_arr[:, -1].astype(np.uint32)
        rgb_arr = np.concatenate(rgb_arr, axis=0)[filtered_indices]
        assert rgb_arr.dtype == np.uint8
        distance2cam_arr = distance2cam_arr[filtered_indices]
        seg_arr = np.concatenate(seg_arr, axis=0)[filtered_indices]
        assert seg_arr.dtype == np.uint8
        class_color_arr = class_to_color(np.concatenate(seg_arr, axis=0)).astype(np.uint8)
        depth_unc_arr = np.concatenate(depth_unc_arr, axis=0)[filtered_indices]

        frame_index_arr = np.concatenate(frame_index_arr, axis=0)[filtered_indices]
        assert frame_index_arr.dtype == np.uint16

        cloud = pd.DataFrame({
            "x": xyz_index_arr[:,0],
            "y": xyz_index_arr[:,1],
            "z": xyz_index_arr[:,2],
            "r": rgb_arr[:,0],
            "g": rgb_arr[:,1],
            "b": rgb_arr[:,2],
            "distance_to_cam": distance2cam_arr[:,0],
            "class": seg_arr[:,0],
            "class_r": class_color_arr[:,0],
            "class_g": class_color_arr[:,1],
            "class_b": class_color_arr[:,2],
            "frame_index": frame_index_arr[:,0],
            "depth_uncertainty": depth_unc_arr[:,0],
        })
        return cloud, keep_masks

    
def tsdf_point_cloud(img_list, depths, masks, poses, intrinsics, cutoff): 
    # TODO Magic Numbers
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=1 / 512.0, 
        sdf_trunc=0.03,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    totensor = torchvision.transforms.ToTensor()

    mask_out_background = np.ones_like(masks[0])
    intrinsics = torch.tensor(intrinsics).float()

    #TODO: Magic numbers
    mask_out_background[:170,80:-80] *= 0 
    for i in tqdm(range(len(poses))):
    
        if i > len(poses)-10:
            mask_out_background = np.ones_like(masks[0])

        # Rectify to linear intrinsics in case alpha is not zero!
        projected_img, projected_mask, projected_depth = rectify(
            totensor(Image.open(img_list[i])).unsqueeze(0), 
            torch.tensor(masks[i]*mask_out_background).unsqueeze(0).unsqueeze(0).float(),
            torch.tensor(depths[i]).unsqueeze(0).unsqueeze(0), 
            intrinsics[i]
        )

        depth = o3d.geometry.Image(projected_depth*projected_mask)
        color = o3d.geometry.Image(np.ascontiguousarray(projected_img.transpose(1, 2, 0)*255.).astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=cutoff, convert_rgb_to_intensity=False, depth_scale=1)
        
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
            width=args.width,
            height=args.height,
            fx=intrinsics[i][0],
            fy=intrinsics[i][1],
            cx=intrinsics[i][2],
            cy=intrinsics[i][3],
        ),
            np.linalg.inv(poses[i]))
    print("Extracting Point Cloud from TSDF!")
    pc = volume.extract_point_cloud()
    xyz = np.array(pc.points)
    rgb = (np.array(pc.colors)*255).astype(np.uint8)
    return xyz, rgb
 

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