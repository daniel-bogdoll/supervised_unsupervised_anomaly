import numpy as np
import os
import yaml
import cv2
import open3d as o3d
import re
from collections import defaultdict
from tqdm import tqdm

def depth_remover(pc, threshold):
    near_mask_z = np.logical_and(pc[:, 2] < threshold, pc[:, 2] > -threshold)
    near_mask_x = np.logical_and(pc[:, 0] < threshold, pc[:, 0] > -threshold)
    pc_indices = np.where(np.logical_and(near_mask_z, near_mask_x))[0]
    
    return pc_indices

def read_calib_file(filepath, data_dic):
    """
    Inspired by: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            try:
                data_dic[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data_dic

def ground_remover_segmentation(pc, ground_labels):
    #check
    assert(ground_labels.shape[0] == pc.shape[0])

    mask = ground_labels == 1
    pc = pc[mask]

    return pc, mask 

def project_velo_to_cam2(calib):
    """
    Inspired by: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
    """
    P_velo2cam_ref = np.vstack((calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ P_velo2cam_ref
    
    return proj_mat, P_velo2cam_ref       

def project_to_image(points, proj_mat):
    """
    Inspired by: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
    """
    num_pts = points.shape[1]
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    
    return points[:2, :]

def get_points_camerafov(pts_velo, calib, img_width, img_height):
    """
    Inspired by: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/lidar_camera_project.py
    """
    # projection matrix (project from velo2cam2)
    pts_velo_xyz = pts_velo[:, :3]
    proj_velo2cam2, P_velo2cam_ref = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo_xyz.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                        (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                        (pts_velo_xyz[:, 0] > 0)
                        )[0]
    imgfov_pc_velo_xyz = pts_velo_xyz[inds, :]

    #transform points in camera coordinate frame
    points_transpose = pts_velo_xyz.transpose()
    points_transpose = np.vstack((points_transpose, np.ones((1, points_transpose.shape[1])))).astype(np.float32)
    transformed_points = P_velo2cam_ref @ points_transpose
    transformed_points = transformed_points.astype(np.float32)
    points_within_image_fov = transformed_points[:, inds] 
    pc_velo_in_camera_coodinate_frame = points_within_image_fov.transpose()
    pc_velo_in_camera_coodinate_frame = pc_velo_in_camera_coodinate_frame[:,0:3] #points in camera coordinate frame
    
    return pc_velo_in_camera_coodinate_frame, imgfov_pc_velo_xyz, inds


def indices_camerafov_depththreshold_groundremover(path_root, path_infer, pc_xyz, seq, frame):
    depth_threshold = 35
    
    path_ground_labels_KITTI = os.path.join(path_infer, 'ground_removal', seq, 'predictions')
    path_image = os.path.join(path_root, seq, 'image_2')
    path_calib = os.path.join(path_root, seq, 'calib.txt')

    ground_pred_file = os.path.join(path_ground_labels_KITTI, frame)
    image_file = os.path.join(path_image, frame.split('.')[0] + '.png')        

    # get points within camera field of view (camera2 KITTI odometry) 
    rgb = cv2.cvtColor(cv2.imread(os.path.join(image_file)), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape
    calib_dic = {}
    calib = read_calib_file(path_calib, calib_dic)
    pc_xyz_cam_coor, pc_xyz_velo_coor, indices_camerafov = get_points_camerafov(pc_xyz, calib, img_width, img_height)

    pc_xyz = pc_xyz_cam_coor

    # remove points that are far away
    indices_depth = depth_remover(pc_xyz, depth_threshold)
    pc_xyz = pc_xyz[indices_depth]

    # remove ground points
    ground_labels = np.fromfile(ground_pred_file, dtype=np.int16)
    ground_labels = ground_labels[indices_camerafov]
    ground_labels = ground_labels[indices_depth]
    pc_xyz, mask_ground_remover = ground_remover_segmentation(pc_xyz, ground_labels)
    
    return pc_xyz, indices_camerafov, indices_depth, mask_ground_remover

def trafo_back_to_lidar_coor(pts_postprocessed_cam_coor, path_calib):
    calib_dic = {}
    calib = read_calib_file(path_calib, calib_dic)

    # projection matrix velo2cam2
    pts_cam_coor_xyz = pts_postprocessed_cam_coor[:, :3]
    _, P_velo2cam_ref = project_velo_to_cam2(calib)

    #transform points in velodyne coordinate frame
    points_transpose = pts_cam_coor_xyz.transpose()
    points_transpose = np.vstack((points_transpose, np.ones((1, points_transpose.shape[1])))).astype(np.float32)
    P_cam2velo_ref = np.linalg.inv(P_velo2cam_ref)
    pc_camerafov_lidar_coor = P_cam2velo_ref @ points_transpose
    pc_camerafov_lidar_coor = pc_camerafov_lidar_coor.astype(np.float32)
    pc_camerafov_lidar_coor = pc_camerafov_lidar_coor.transpose()
    pc_camerafov_lidar_coor = pc_camerafov_lidar_coor[:,0:3] #points in velodyne coordinate frame
            
    return pc_camerafov_lidar_coor

def transform_sem_labels_to_motion_labels(sup_labels, config_file):
    # config file
    data_yaml = yaml.safe_load(open(config_file, 'r'))
    combine_mos_semantic_map = data_yaml['combine_mos_semantic_map_inv']
    
    # get dynamic classes
    dynamic_classes = []
    for key in combine_mos_semantic_map.keys():
        dynamic_classes.append(key)

    # get binary motion representation of semantic motion labels     
    sup_labels_binary = np.zeros(len(sup_labels))
    for i in range(len(sup_labels)):
        if sup_labels[i] in dynamic_classes:
            sup_labels_binary[i] = 1 # 1=dynamic and 0=static
    
    return sup_labels_binary

def check_consistency_labels(sup_labels, self_labels, config_file):
    # transform semantic motion labels in motion labels for comparison
    sup_labels_binary = transform_sem_labels_to_motion_labels(sup_labels, config_file)

    # check
    assert(len(sup_labels_binary) == len(self_labels))
    
    # 4 scenarios
    anomaly_labels = np.zeros(len(self_labels))
    for idx in range(len(self_labels)):
        if self_labels[idx] == 0 and sup_labels_binary[idx] == 0: # scenario 1: self:static and sup:static
            anomaly_labels[idx] = 1
        elif self_labels[idx] == 1 and sup_labels_binary[idx] == 1: # scenario 2: self:dyn and sup:dyn
            anomaly_labels[idx] = 2
        elif self_labels[idx] == 1 and sup_labels_binary[idx] == 0: #scenario 3: self:dyn and sup:static
            anomaly_labels[idx] = 3 
        elif self_labels[idx] == 0 and sup_labels_binary[idx] == 1: #scenario 4: self:static and sup:dyn
            anomaly_labels[idx] = 4

    return anomaly_labels

def cluster_points(pc, anomaly_labels, eps, min_samples):
    clusters_incon_static_dyn = []
    clusters_incon_dyn_static = []

    # inconsistent points
    # scenario 3: self:dyn and sup:static
    mask_inconsistent_3 = anomaly_labels == 3
    pc_inconsistent_3 = pc[mask_inconsistent_3]
    # scenario 4: self:static and sup:dyn
    mask_inconsistent_4 = anomaly_labels == 4
    pc_inconsistent_4 = pc[mask_inconsistent_4]

    # cluster inconsistent points
    # scenario 3
    pcd_inconsistent_3 = o3d.geometry.PointCloud()
    pcd_inconsistent_3.points = o3d.utility.Vector3dVector(pc_inconsistent_3)
    if pc_inconsistent_3.shape[0] != 0:
        clusters_incon_static_dyn = np.array(pcd_inconsistent_3.cluster_dbscan(eps, min_samples))
    # scenario 4
    pcd_inconsistent_4 = o3d.geometry.PointCloud()
    pcd_inconsistent_4.points = o3d.utility.Vector3dVector(pc_inconsistent_4)
    if pc_inconsistent_4.shape[0] != 0:
        clusters_incon_dyn_static = np.array(pcd_inconsistent_4.cluster_dbscan(eps, min_samples))
    
    return clusters_incon_static_dyn, clusters_incon_dyn_static

def write_to_file(labels, pc_post, found_clusters_incon_static_dyn_3, found_clusters_incon_dyn_static_4, path_to_save, frame):
    #check
    assert(len(labels) == pc_post.shape[0])

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # save anomaly labels
    if not os.path.exists(os.path.join(path_to_save, 'anomaly_labels')):
        os.makedirs(os.path.join(path_to_save, 'anomaly_labels'))
    path_to_save_labels = os.path.join(path_to_save, 'anomaly_labels', frame)
    labels = np.asarray(labels, dtype=np.int16)
    labels.tofile(path_to_save_labels)

    # save post processed point clouds in lidar coordinates
    if not os.path.exists(os.path.join(path_to_save, 'pc_velo_camerafov')):
        os.makedirs(os.path.join(path_to_save, 'pc_velo_camerafov'))
    path_to_save_pc = os.path.join(path_to_save, 'pc_velo_camerafov', frame)
    pc_post = np.asarray(pc_post, dtype=np.float32)
    pc_post.tofile(path_to_save_pc)

    # save clusters
    if not os.path.exists(os.path.join(path_to_save, 'clusters')):
        os.makedirs(os.path.join(path_to_save, 'clusters'))
    # scenario 3
    path_to_save_clusters = os.path.join(path_to_save, 'clusters', frame.split('.')[0] + '_cluster_incon_static_dyn_3.bin')
    found_clusters_incon_static_dyn_3 = np.asarray(found_clusters_incon_static_dyn_3, dtype=np.int16)
    found_clusters_incon_static_dyn_3.tofile(path_to_save_clusters)
    # scenario 4
    path_to_save_clusters = os.path.join(path_to_save, 'clusters', frame.split('.')[0] + '_cluster_incon_dyn_static_4.bin')
    found_clusters_incon_dyn_static_4 = np.asarray(found_clusters_incon_dyn_static_4, dtype=np.int16)
    found_clusters_incon_dyn_static_4.tofile(path_to_save_clusters)
 
if __name__ == '__main__':
    # load config file
    config_filename = 'anomaly_detection/config/config_paths.yaml'
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    
    sequences = config['sequences']
    path_dataset = config['path_dataset']
    path_inference = config['path_inference']

    if not os.path.exists(os.path.join(path_inference, 'anomalies_sup_self')):
        os.makedirs(os.path.join(path_inference, 'anomalies_sup_self'))
    
    number_of_clusters = defaultdict(dict)
    for seq in tqdm(sequences):
        seq = '{0:04d}'.format(int(seq))
        path_raw = os.path.join(path_dataset, seq, 'velodyne')
        path_calib = os.path.join(path_dataset ,seq, 'calib.txt')
        config_file = 'anomaly_detection/config/combine_mos_semantics.yaml'

        supervised_labels = os.path.join(path_inference, 'SalsaNext_combined_semantics_mos', seq, 'predictions')
        self_supervised_labels = os.path.join(path_inference, 'self_motion_labels', seq)
        path_to_save = os.path.join(path_inference, 'anomalies_sup_self', seq)
            
        files = os.listdir(self_supervised_labels)
        files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
        for frame in range(len(files)):
            frame_path_pc = os.path.join(path_raw, files[frame])

            # load raw velodyne
            pc_raw = np.fromfile(frame_path_pc, dtype=np.float32)
            pc_raw = pc_raw.reshape((-1,4))
            pc_raw_xyz = pc_raw[:, 0:3]  # get xyz

            # get indices of preprocessing
            pc_xyz_post_cam_coor, indices_camerafov, indices_depth, mask_ground_remover = indices_camerafov_depththreshold_groundremover(path_dataset, path_inference, pc_raw_xyz, seq, files[frame])
            pc_xyz_post_velo_coor = trafo_back_to_lidar_coor(pc_xyz_post_cam_coor, path_calib)

            # remove points > 25m
            depth_mask_x = pc_xyz_post_velo_coor[:, 0] < 25 
            pc_xyz_post_velo_coor = pc_xyz_post_velo_coor[depth_mask_x]

            # load labels and select supervised labels
            sup_sem_labels = np.fromfile(os.path.join(supervised_labels,files[frame]), dtype=np.int16).reshape((-1))
            sup_sem_labels = sup_sem_labels[indices_camerafov]
            sup_sem_labels = sup_sem_labels[indices_depth]
            sup_sem_labels = sup_sem_labels[mask_ground_remover]
            sup_sem_labels = sup_sem_labels[depth_mask_x]
            self_labels = np.fromfile(os.path.join(self_supervised_labels,files[frame]), dtype=np.int16).reshape((-1))
            
            #check
            assert(len(sup_sem_labels) == len(self_labels))

            #check consistency between supervised and self-supervised labels
            anomaly_labels = check_consistency_labels(sup_sem_labels, self_labels, config_file)
            
            # get frame name
            if files[frame].split('_')[-1] == '0.bin':
                frame_number = '0'
            else:
                frame_number = files[frame].split('_')[-1].split('.')[0].lstrip('0')

            # cluster inconsistent points
            clusters_incon_static_dyn, clusters_incon_dyn_static = cluster_points(pc_xyz_post_velo_coor, anomaly_labels, 1, 25)

            # write to file
            write_to_file(anomaly_labels, pc_xyz_post_velo_coor, clusters_incon_static_dyn, clusters_incon_dyn_static, path_to_save, files[frame])    
