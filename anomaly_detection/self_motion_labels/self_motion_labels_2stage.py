import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import numpy.ma as ma
import yaml
from collections import defaultdict

def read_calib_file(filepath, data_dic):
    """
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
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

def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))
        
    return P_velo2cam_ref

def trafo_back_to_lidar_coor(pc_postprocssed, calib):
    calib_dic = {}
    calib = read_calib_file(calib, calib_dic)

    # flip x and y axis by 180 degrees
    pc_postprocssed[:,0]= pc_postprocssed[:,0] *(-1)
    pc_postprocssed[:,1]= pc_postprocssed[:,1] *(-1)

    # projection matrix (project from velo2cam2)
    pts_cam_coor_xyz = pc_postprocssed[:, :3]
    P_velo2cam_ref = project_velo_to_cam2(calib)

    # transform points in camera coordinate frame
    points_transpose = pts_cam_coor_xyz.transpose()
    points_transpose = np.vstack((points_transpose, np.ones((1, points_transpose.shape[1])))).astype(np.float32)
    P_cam2velo_ref = np.linalg.inv(P_velo2cam_ref)
    pc_camerafov_lidar_coor = P_cam2velo_ref @ points_transpose
    pc_camerafov_lidar_coor = pc_camerafov_lidar_coor.astype(np.float32)
    pc_camerafov_lidar_coor = pc_camerafov_lidar_coor.transpose()
    pc_camerafov_lidar_coor = pc_camerafov_lidar_coor[:,0:3] #points in camera coordinate frame
           
    return pc_camerafov_lidar_coor

def transform_lidar_in_same_coordinate_frame(post_processed_pts_1, post_processed_pts_2, scene_flow_pred, seq, frame, path_infer, path_to_calib, visualize):
    pc1 = np.fromfile(post_processed_pts_1, dtype=np.float32)
    pc1 = pc1.reshape((-1,3))
    pc1 = pc1[:,0:3]

    pc2 = np.fromfile(post_processed_pts_2, dtype=np.float32)
    pc2 = pc2.reshape((-1,3))
    pc2 = pc2[:,0:3]

    flow_pred = np.fromfile(scene_flow_pred, dtype=np.float32)
    flow_pred = flow_pred.reshape((-1,3))
        
    pc1_plus_flow = pc1[:,:3] + flow_pred[:,:3]
    
    # transform from cam coor back to velo coor
    pc1_lidar_coor = trafo_back_to_lidar_coor(pc1, path_to_calib)
    pc1_plus_flow_lidar_coor = trafo_back_to_lidar_coor(pc1_plus_flow, path_to_calib)

    path_to_trafo = os.path.join(path_infer, 'self_pose_estimation', 'transformations_kitti_' + seq +'.npy')
    trafo = np.load(path_to_trafo)
    trafo_pc2 = trafo[frame][0]
    
    # transform in same frame
    pc1_plus_flow_lidar_coor = pc1_plus_flow_lidar_coor.transpose()
    pc1_plus_flow_lidar_coor = np.vstack((pc1_plus_flow_lidar_coor, np.ones((1, pc1_plus_flow_lidar_coor.shape[1]))))
    pc1_plus_flow_lidar_coor_transformed = trafo_pc2.dot(pc1_plus_flow_lidar_coor).T
    pc1_plus_flow_lidar_coor_transformed = pc1_plus_flow_lidar_coor_transformed[:,0:3]
    
    # ego_motion
    dist_with_ego_motion = np.linalg.norm(flow_pred, axis=1)
    flow_transformed = pc1_plus_flow_lidar_coor_transformed[:,:3] - pc1_lidar_coor[:,:3]
    dist_wo_ego_motion = np.linalg.norm(flow_transformed, axis=1)
    ego_motion = abs(dist_with_ego_motion - dist_wo_ego_motion)

    if visualize: 
        color1 = np.asarray([34,139,34])
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1_lidar_coor)
        pcd1.paint_uniform_color(color1.astype(np.float32)/255.0)

        color2 = np.asarray([178,34,34])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc1_plus_flow_lidar_coor_transformed)
        pcd2.paint_uniform_color(color2.astype(np.float32)/255.0)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
        o3d.visualization.draw_geometries([pcd1] + [pcd2] + [axis], window_name = 'Point cloud t and t+1 where t+1 is adjusted for the ego-motion')

    return pc1_lidar_coor, pc1_plus_flow_lidar_coor_transformed

def cluster_points(pc, eps, min_samples):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    clusters = np.array(pcd.cluster_dbscan(eps, min_samples))
    
    return clusters

def get_motion_kmh(distance_m, difference_timestamp):
    distance_km = distance_m/1000
    hours = (difference_timestamp/60)/60
    motion_kmh = distance_km/hours
    return motion_kmh

def get_distance_m(motion_kmh, difference_timestamp):
    hours = (difference_timestamp/60)/60
    distance_km = motion_kmh * hours
    distance_m = distance_km * 1000
    return distance_m

def get_motion_label(pc1_lidar_coor, pc1_plus_flow_lidar_coor_transformed, frame_number, path_timestamps, visualize):
    # load timestamps
    timestamp_file = open(path_timestamps, 'r')
    timestamps = timestamp_file.read().split('\n') #timestamps in seconds
    difference_timestamp = float(timestamps[frame_number+1])-float(timestamps[frame_number])

    # get compensated scene flow -> own induced motion of each point
    flow_transformed = pc1_plus_flow_lidar_coor_transformed[:,:3] - pc1_lidar_coor[:,:3]
    dist = np.linalg.norm(flow_transformed, axis=1)

    # remove depth > 25m
    depth_mask_x = pc1_lidar_coor[:, 0] < 25
    pc1_lidar_coor = pc1_lidar_coor[depth_mask_x]
    flow_transformed = flow_transformed[depth_mask_x]
    pc1_plus_flow_lidar_coor_transformed = pc1_plus_flow_lidar_coor_transformed[depth_mask_x]
    dist = dist[depth_mask_x]
    
    # first clustering -> based on points
    clusters = cluster_points(pc1_lidar_coor, 0.6, 30)
    list_clusters = np.unique(clusters)

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc1_lidar_coor)
        max_cluster = clusters.max()
        colors = plt.get_cmap("tab20b")(clusters / (max_cluster if max_cluster > 0 else 1))
        colors[clusters < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
        o3d.visualization.draw_geometries([pcd], window_name='first/point-based clustering')
    
    # get mean distance and std per cluster
    dict_per_cluster = defaultdict(dict)
    mean_clusters_pot_dyn = []
    list_all_points_kmh = []
    name_clusters_pot_dyn = []
    for cluster in list_clusters:
        mask = clusters == cluster 
        dist_selection = dist[mask]
        dist_selection_kmh = [get_motion_kmh(k, difference_timestamp) for k in dist_selection]
        
        # remove outliers
        wo_outliers = []
        for distance_kmh in dist_selection_kmh:
            if distance_kmh <= 80:  # threshold 80km/h
                wo_outliers.append(distance_kmh)
        dist_selection_kmh = wo_outliers
        
        list_all_points_kmh += dist_selection_kmh

        if len(dist_selection_kmh) != 0:
            dict_per_cluster[cluster]['median'] = np.median(dist_selection_kmh)
            dict_per_cluster[cluster]['normalized_std'] = np.std(dist_selection_kmh)/np.mean(dist_selection_kmh)
            if np.std(dist_selection_kmh)/np.mean(dist_selection_kmh) < 0.12: # threshold normalized standard deviation
                mean_clusters_pot_dyn.append(np.median(dist_selection_kmh))    
                name_clusters_pot_dyn.append(cluster)
        else:
            dict_per_cluster[cluster]['median'] = 0
            dict_per_cluster[cluster]['normalized_std'] = 0
    
    # cluster flow of potentially movable objects
    mask_pot_dy = np.in1d(clusters, name_clusters_pot_dyn)
    flow_transformed_pot_dyn = flow_transformed[mask_pot_dy]
    dist_pot_dyn = np.linalg.norm(flow_transformed_pot_dyn, axis=1)
    pc1_lidar_coor_pot_dyn = pc1_lidar_coor[mask_pot_dy]

    if flow_transformed_pot_dyn.shape[0] != 0:

        # second clustering -> based on flow
        clusters_flow_pot_dyn = cluster_points(flow_transformed_pot_dyn, 0.015, 25) #eps_flow=0.015

        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc1_lidar_coor_pot_dyn)
            max_cluster = clusters_flow_pot_dyn.max()
            colors = plt.get_cmap("tab20")(clusters_flow_pot_dyn / (max_cluster if max_cluster > 0 else 1))
            colors[clusters_flow_pot_dyn < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
            o3d.visualization.draw_geometries([pcd], window_name = 'second/flow-based clustering')
        
        # check newly found clusters and get mean distance and std per clusterxs
        list_clusters_pot_dy = np.unique(clusters_flow_pot_dyn)
        dict_per_cluster_pot_dy = defaultdict(dict)
        mean_clusters_pot_dyn_based_flow = []
        name_clusters_pot_dyn_based_flow = []
        for cluster in list_clusters_pot_dy:
            mask = clusters_flow_pot_dyn == cluster 
            dist_selection_pot_dyn = dist_pot_dyn[mask]
            dist_selection_kmh_pot_dyn = [get_motion_kmh(k, difference_timestamp) for k in dist_selection_pot_dyn]

            if len(dist_selection_kmh_pot_dyn) != 0:
                dict_per_cluster_pot_dy[cluster]['median'] = np.median(dist_selection_kmh_pot_dyn)
                dict_per_cluster_pot_dy[cluster]['normalized_std'] = np.std(dist_selection_kmh_pot_dyn)/np.mean(dist_selection_kmh_pot_dyn)
                if np.std(dist_selection_kmh_pot_dyn)/np.mean(dist_selection_kmh_pot_dyn) < 0.12: #0.12
                    mean_clusters_pot_dyn_based_flow.append(np.median(dist_selection_kmh_pot_dyn))    
                    name_clusters_pot_dyn_based_flow.append(cluster)
            else:
                dict_per_cluster_pot_dy[cluster]['median'] = 0
                dict_per_cluster_pot_dy[cluster]['normalized_std'] = 0
        
        # assign each point a motion label
        for key in dict_per_cluster_pot_dy.keys():
            if key == -1: #outlier as static
                dict_per_cluster_pot_dy[key]['label'] = 0
            else:
                if dict_per_cluster_pot_dy[key]['normalized_std'] <= 0.12:  # threshold normalized standard deviation
                    if dict_per_cluster_pot_dy[key]['median'] >= 4: # threshold speed
                        dict_per_cluster_pot_dy[key]['label'] = 1
                    else:
                        dict_per_cluster_pot_dy[key]['label'] = 0
                else:
                    dict_per_cluster_pot_dy[key]['label'] = 0
                
        # label points based on labeled cluster
        labels = np.zeros(pc1_lidar_coor.shape[0])
        for cluster in range(len(list_clusters_pot_dy)):
            mask = clusters_flow_pot_dyn == list_clusters_pot_dy[cluster]
            labels_pot_dyn = labels[mask_pot_dy]
            labels_pot_dyn[mask] = dict_per_cluster_pot_dy[list_clusters_pot_dy[cluster]]['label']
            labels[mask_pot_dy] = labels_pot_dyn
    else:
        # no potential dynamic points availabel -> all static
        labels = np.zeros(pc1_lidar_coor.shape[0])
    
    return labels, pc1_lidar_coor

def write_to_file(labels, path_infer, seq, frame_t0):
    if not os.path.exists(os.path.join(path_infer, 'self_motion_labels')):
            os.makedirs(os.path.join(path_infer, 'self_motion_labels'))
    if not os.path.exists(os.path.join(path_infer, 'self_motion_labels', seq)):
            os.makedirs(os.path.join(path_infer, 'self_motion_labels', seq))
    path_to_save = os.path.join(path_infer, 'self_motion_labels', seq, frame_t0)
    labels = np.asarray(labels, dtype=np.int16)
    labels.tofile(path_to_save)

def visualize_motion_segmentation(pc_lidar_coor, labels):
    #divide points into dynamic and static points
    static_mask = ma.masked_where(labels == 0, labels)
    dynamic_mask = ma.masked_where(labels == 1, labels)
    static_points = pc_lidar_coor[static_mask.mask]
    dynamic_points = pc_lidar_coor[dynamic_mask.mask]

    #assign color to static points 
    color_static = np.asarray([34,139,34])
    pcd_static = o3d.geometry.PointCloud()
    pcd_static.points = o3d.utility.Vector3dVector(static_points)
    pcd_static.paint_uniform_color(color_static.astype(np.float32)/255.0)

    #assign color to dynamic points
    color_dynamic = np.asarray([178,34,34])
    pcd_dynamic = o3d.geometry.PointCloud()
    pcd_dynamic.points = o3d.utility.Vector3dVector(dynamic_points)
    pcd_dynamic.paint_uniform_color(color_dynamic.astype(np.float32)/255.0)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
    o3d.visualization.draw_geometries([pcd_static] + [pcd_dynamic] + [axis], window_name='motion segmentation results')

def main(seq, path_raw, path_infer, visualize):
    seq = '{0:02d}'.format(int(seq))

    path_calib = os.path.join(path_raw ,seq, 'calib.txt')
    post_processed_pts = os.path.join(path_infer, 'scene_flow' , seq, 'post_processed_points_cam_coor')
    sf = os.path.join(path_infer, 'scene_flow', seq, 'predictions')
    path_to_timestamps = os.path.join(path_raw, seq, 'times.txt')
        
    files = os.listdir(post_processed_pts)
    files.sort()
        
    for frame in range(len(files)-1):
        #get file names
        if files[frame] == '000000.bin':
            frame_num = 0
        else:
            frame_num = int(files[frame].split('.')[0].lstrip('0'))
        frame_t0 = files[frame]
        frame_t1 = files[frame+1]
        scene_flow_t0 = files[frame].split('.')[0] + '_flow.bin'

        post_processed_pc_1 = os.path.join(post_processed_pts, frame_t0)
        post_processed_pc_2 = os.path.join(post_processed_pts, frame_t1)
        sf_pred = os.path.join(sf,scene_flow_t0)

        pc1_lidarcoor, pc1_flow_lidarcoor_transformed = transform_lidar_in_same_coordinate_frame(post_processed_pc_1, post_processed_pc_2, sf_pred, seq, frame, path_infer, path_calib, visualize)
        self_motion_labels, pc1_lidar_coor = get_motion_label(pc1_lidarcoor, pc1_flow_lidarcoor_transformed, frame_num, path_to_timestamps, visualize)
        write_to_file(self_motion_labels, path_infer, seq, frame_t0)

        if visualize:
            visualize_motion_segmentation(pc1_lidar_coor, self_motion_labels)

if __name__ == '__main__':
    # load config file
    config_filename = 'config/config_paths.yaml'
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    
    visualize = config['visualize']
    sequences = config['sequences']
    path_dataset = config['path_dataset']
    path_inference = config['path_inference']
    
    for seq in sequences:
        main(seq, path_dataset, path_inference, visualize)
    
    