import sys, os
import os.path as osp
import numpy as np
from random import random

import numpy as np
import os.path as osp
import os
import cv2
import matplotlib.pyplot as plt
import yaml
import re

import torch.utils.data as data

__all__ = ['KITTI_odometry_raw']


class KITTI_odometry_raw(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 save_path,
                 sequence,
                 remove_ground=True):
        
        self.seq = '{0:04d}'.format(int(sequence))
        self.root = osp.join(data_root, self.seq)
        self.save_path = osp.join(save_path)
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))
        
        #directory to save results
        if not os.path.exists(os.path.join(self.save_path,'scene_flow')):
            os.makedirs(os.path.join(self.save_path,'scene_flow'))
        if not os.path.exists(os.path.join(self.save_path, 'scene_flow', self.seq)):
            os.makedirs(os.path.join(self.save_path,'scene_flow', self.seq))
        if not os.path.exists(os.path.join(self.save_path, 'scene_flow', self.seq, 'predictions')):
            os.makedirs(os.path.join(self.save_path, 'scene_flow', self.seq, 'predictions'))
        if not os.path.exists(os.path.join(self.save_path, 'scene_flow', self.seq, 'post_processed_points_cam_coor')):
            os.makedirs(os.path.join(self.save_path, 'scene_flow', self.seq, 'post_processed_points_cam_coor'))

    def __len__(self):
        return (len(self.samples) - 1)

    def __getitem__(self, index):
        #pc1_loaded, pc2_loaded, sf_loaded = self.pc_loader(self.samples[index])
        #pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded, sf_loaded])
        pc1_transformed, pc2_transformed = self.pc_loader(self.samples, index, self.root, self.save_path, self.seq)
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        sf_dummy = np.zeros(pc1_transformed.shape[0])
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_dummy, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        
        root = osp.realpath(osp.expanduser(self.root))
        #useful_paths_velodyne = []
        #useful_paths_velodyne.append(root)
        
        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]

        useful_paths_velodyne = []
        for item in useful_paths:
            if item.split('/')[-1] == 'velodyne':
                useful_paths_velodyne.append(item)

        res_paths = useful_paths_velodyne
        assert(len(res_paths) == 1)

        all_velo_files = os.listdir(res_paths[0])
        all_velo_files = sorted(all_velo_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
        all_velo_files_paths = []
        for file in all_velo_files:
            all_velo_files_paths.append(os.path.join(res_paths[0], file))

        return all_velo_files_paths
        
    #preproessing of raw velodyne scans
    def pc_loader(self, samples, index, root, save_path, seq):

        depth_threshold = 35
        save_to_file = True
    
        path_velodyne_KITTI = os.path.join(root, 'velodyne')
        velo_file_pc1 = samples[index]
        velo_file_pc2 = samples[index + 1]
        path_ground_labels_KITTI = os.path.join(save_path,'ground_removal', seq, 'predictions')
        path_image = os.path.join(root, 'image_2')
        path_calib = os.path.join(root, 'calib.txt')

        #get files
        ground_pred_file_pc1 = os.path.join(path_ground_labels_KITTI, velo_file_pc1.split('/')[-1])
        ground_pred_file_pc2 = os.path.join(path_ground_labels_KITTI, velo_file_pc2.split('/')[-1])
        image_file_pc1 = os.path.join(path_image, velo_file_pc1.split('/')[-1].split('.')[0] + '.png')
        image_file_pc2 = os.path.join(path_image, velo_file_pc2.split('/')[-1].split('.')[0] + '.png')
        
        #open point cloud time t
        pc = np.fromfile(velo_file_pc1, dtype=np.float32)
        pc = pc.reshape((-1,4))
        pc_xyz_1 = pc[:, 0:3]  # get xyz

        #open point cloud time t+1
        pc = np.fromfile(velo_file_pc2, dtype=np.float32)
        pc = pc.reshape((-1,4))
        pc_xyz_2 = pc[:, 0:3]  # get xyz
        
        ground_path_1 = ground_pred_file_pc1
        ground_path_2 = ground_pred_file_pc2
        path_image_1 = image_file_pc1
        path_image_2 = image_file_pc2

        # get points within camera field of view (cam2 KITTI odometry) 
        rgb_1 = cv2.cvtColor(cv2.imread(os.path.join(path_image_1)), cv2.COLOR_BGR2RGB)
        img_height_1, img_width_1, img_channel_1 = rgb_1.shape
        rgb_2 = cv2.cvtColor(cv2.imread(os.path.join(path_image_2)), cv2.COLOR_BGR2RGB)
        img_height_2, img_width_2, img_channel_2 = rgb_2.shape
        calib_dic = {}
        calib = self.read_calib_file(path_calib, calib_dic)
        pc_velo_in_camera_coodinate_frame_1, inds_1 = self.get_point_camerafov(pc_xyz_1, calib, rgb_1, img_width_1, img_height_1)
        pc_velo_in_camera_coodinate_frame_2, inds_2 = self.get_point_camerafov(pc_xyz_2, calib, rgb_2, img_width_2, img_height_2)
        #flip x and y axis by 180 degrees
        pc_velo_in_camera_coodinate_frame_1[:,0]= pc_velo_in_camera_coodinate_frame_1[:,0] *(-1)
        pc_velo_in_camera_coodinate_frame_1[:,1]= pc_velo_in_camera_coodinate_frame_1[:,1] *(-1)
        pc_velo_in_camera_coodinate_frame_2[:,0]= pc_velo_in_camera_coodinate_frame_2[:,0] *(-1)
        pc_velo_in_camera_coodinate_frame_2[:,1]= pc_velo_in_camera_coodinate_frame_2[:,1] *(-1)
        pc_xyz_1 = pc_velo_in_camera_coodinate_frame_1
        pc_xyz_2 = pc_velo_in_camera_coodinate_frame_2

        #remove points that are further away than the threshold
        if depth_threshold > 0:
            pc_indices_1 = self.depth_remover(pc_xyz_1, depth_threshold)
            pc_xyz_1 = pc_xyz_1[pc_indices_1]
            pc_indices_2 = self.depth_remover(pc_xyz_2, depth_threshold)
            pc_xyz_2 = pc_xyz_2[pc_indices_2]

        #remove ground points
        ground_labels_1 = np.fromfile(ground_path_1, dtype=np.int16)
        ground_labels_2 = np.fromfile(ground_path_2, dtype=np.int16)
        ground_labels_1 = ground_labels_1[inds_1]
        ground_labels_2 = ground_labels_2[inds_2]
        if depth_threshold > 0:
            ground_labels_1 = ground_labels_1[pc_indices_1]
            ground_labels_2 = ground_labels_2[pc_indices_2]
        pc_xyz_1, mask_1 = self.ground_remover_segmentation(pc_xyz_1, ground_labels_1)
        pc_xyz_2, mask_2 = self.ground_remover_segmentation(pc_xyz_2, ground_labels_2)
        
        pc1 = pc_xyz_1
        pc2 = pc_xyz_2

        last_file_in_seq = False
        if samples[index + 1] == samples[-1]:
            last_file_in_seq = True

        #save postprocessed points
        save_path_post_processed = os.path.join(save_path, 'scene_flow', seq, 'post_processed_points_cam_coor')
        if save_to_file:
            file_name_pc1 = str(velo_file_pc1.split('/')[-1].split('.')[0]+'.bin')
            save_path_post_processed_points = os.path.join(save_path_post_processed, file_name_pc1)
            pc1.tofile(save_path_post_processed_points)
            if last_file_in_seq:
                file_name_pc2 = str(velo_file_pc2.split('/')[-1].split('.')[0]+'.bin')
                save_path_post_processed_points = os.path.join(save_path_post_processed, file_name_pc2)
                pc2.tofile(save_path_post_processed_points)
                    
        return pc1, pc2

    def depth_remover(self, pc, threshold):
        near_mask_z = np.logical_and(pc[:, 2] < threshold, pc[:, 2] > -threshold)
        near_mask_x = np.logical_and(pc[:, 0] < threshold, pc[:, 0] > -threshold)
        pc_indices = np.where(np.logical_and(near_mask_z, near_mask_x))[0]
        return pc_indices

    def read_calib_file(self, filepath, data_dic):
        """
        Read in a calibration file and parse into a dictionary.
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

    def ground_remover_segmentation(self, pc, ground_labels):
        #GndNet -1: out of range, 0: ground, 1: not ground

        assert(ground_labels.shape[0] == pc.shape[0])

        mask = ground_labels == 1
        pc = pc[mask]

        return pc, mask 

    def project_velo_to_cam2(self, calib): 
        """
        Ref: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
        """
        P_velo2cam_ref = np.vstack((calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
        P_rect2cam2 = calib['P2'].reshape((3, 4))
        proj_mat = P_rect2cam2 @ P_velo2cam_ref
        return proj_mat, P_velo2cam_ref
        

    def project_to_image(self, points, proj_mat):
        """
        Ref: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
        """
        
        num_pts = points.shape[1]

        # Change to homogenous coordinate
        points = np.vstack((points, np.ones((1, num_pts))))
        points = proj_mat @ points
        points[:2, :] /= points[2, :]
        return points[:2, :]

    def get_point_camerafov(self, pts_velo, calib, img, img_width, img_height):
        """
        Ref: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/lidar_camera_project.py
        """
        # projection matrix (project from velo2cam2)
        pts_velo_xyz = pts_velo[:, :3]
        proj_velo2cam2, P_velo2cam_ref = self.project_velo_to_cam2(calib)

        # apply projection
        pts_2d = self.project_to_image(pts_velo_xyz.transpose(), proj_velo2cam2)

        # Filter lidar points to be within image FOV
        inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                        (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                        (pts_velo_xyz[:, 0] > 0)
                        )[0]

        #transform points in camera coordinate frame
        points_transpose = pts_velo_xyz.transpose()
        points_transpose = np.vstack((points_transpose, np.ones((1, points_transpose.shape[1])))).astype(np.float32)
        transformed_points = P_velo2cam_ref @ points_transpose
        transformed_points = transformed_points.astype(np.float32)
        points_within_image_fov = transformed_points[:, inds] 
        pc_velo_in_camera_coodinate_frame = points_within_image_fov.transpose()
        pc_velo_in_camera_coodinate_frame = pc_velo_in_camera_coodinate_frame[:,0:3] #points in camera coordinate frame
        
        return pc_velo_in_camera_coodinate_frame, inds