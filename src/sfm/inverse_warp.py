# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import numpy as np
from abc import ABC
from functools import partial
# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

## TODO: link to repository!

import torch
import torch.nn.functional as F

def is_tuple(data):
    """Checks if data is a tuple."""
    return isinstance(data, tuple)
def is_seq(data):
    """Checks if data is a list or tuple."""
    return is_tuple(data) or is_list(data)
def is_dict(data):
    """Checks if data is a dictionary."""
    return isinstance(data, dict)

def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)

def iterate1(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var, *args, **kwargs):
        if is_seq(var):
            return [func(v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(var, *args, **kwargs)
    return inner
def is_int(data):
    """Checks if data is an integer."""
    return isinstance(data, int)

def to_global_pose(pose, zero_origin=False):
    """Get global pose coordinates from current and context poses"""
    if zero_origin:
        pose[0].T[[0]] = torch.eye(4, device=pose[0].device, dtype=pose[0].dtype)
    for b in range(1, len(pose[0])):
        pose[0].T[[b]] = (pose[0][b] * pose[0][0]).T.float()
    for key in pose.keys():
        if key != 0:
            pose[key] = pose[key] * pose[0]
    return pose


# def to_global_pose(pose, zero_origin=False):
#     """Get global pose coordinates from current and context poses"""
#     if zero_origin:
#         pose[(0, 0)].T = torch.eye(4, device=pose[(0, 0)].device, dtype=pose[(0, 0)].dtype). \
#             repeat(pose[(0, 0)].shape[0], 1, 1)
#     for key in pose.keys():
#         if key[0] == 0 and key[1] != 0:
#             pose[key].T = (pose[key] * pose[(0, 0)]).T
#     for key in pose.keys():
#         if key[0] != 0:
#             pose[key] = pose[key] * pose[(0, 0)]
#     return pose


def euler2mat(angle):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([ cosz, -sinz, zeros,
                         sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([ cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat

def pose_vec2mat(vec, mode='euler'):
    """Convert translation and Euler rotation to a [B,4,4] torch.Tensor transformation matrix"""
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:]
    if mode == 'euler':
        rot_mat = euler2mat(rot)
    else:
        raise ValueError('Rotation mode not supported {}'.format(mode))
    mat = torch.cat([rot_mat, trans], dim=2)  # [B,3,4]
    return mat

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor


@iterate1
def invert_pose(T):
    """Invert a [B,4,4] torch.Tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv
    # return torch.linalg.inv(T)


def tvec_to_translation(tvec):
    """Convert translation vector to translation matrix (no rotation)"""
    batch_size = tvec.shape[0]
    T = torch.eye(4).to(device=tvec.device).repeat(batch_size, 1, 1)
    t = tvec.contiguous().view(-1, 3, 1)
    T[:, :3, 3, None] = t
    return T


def euler2rot(euler):
    """Convert Euler parameters to a [B,3,3] torch.Tensor rotation matrix"""
    euler_norm = torch.norm(euler, 2, 2, True)
    axis = euler / (euler_norm + 1e-7)

    cos_a = torch.cos(euler_norm)
    sin_a = torch.sin(euler_norm)
    cos1_a = 1 - cos_a

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    x_sin = x * sin_a
    y_sin = y * sin_a
    z_sin = z * sin_a
    x_cos1 = x * cos1_a
    y_cos1 = y * cos1_a
    z_cos1 = z * cos1_a

    xx_cos1 = x * x_cos1
    yy_cos1 = y * y_cos1
    zz_cos1 = z * z_cos1
    xy_cos1 = x * y_cos1
    yz_cos1 = y * z_cos1
    zx_cos1 = z * x_cos1

    batch_size = euler.shape[0]
    rot = torch.zeros((batch_size, 4, 4)).to(device=euler.device)

    rot[:, 0, 0] = torch.squeeze(xx_cos1 + cos_a)
    rot[:, 0, 1] = torch.squeeze(xy_cos1 - z_sin)
    rot[:, 0, 2] = torch.squeeze(zx_cos1 + y_sin)
    rot[:, 1, 0] = torch.squeeze(xy_cos1 + z_sin)
    rot[:, 1, 1] = torch.squeeze(yy_cos1 + cos_a)
    rot[:, 1, 2] = torch.squeeze(yz_cos1 - x_sin)
    rot[:, 2, 0] = torch.squeeze(zx_cos1 - y_sin)
    rot[:, 2, 1] = torch.squeeze(yz_cos1 + x_sin)
    rot[:, 2, 2] = torch.squeeze(zz_cos1 + cos_a)
    rot[:, 3, 3] = 1

    return rot


def vec2mat(euler, translation, invert=False):
    """Convert Euler rotation and translation to a [B,4,4] torch.Tensor transformation matrix"""
    R = euler2rot(euler)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = tvec_to_translation(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def rot2quat(R):
    """Convert a [B,3,3] rotation matrix to [B,4] quaternions"""
    b, _, _ = R.shape
    q = torch.ones((b, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 3] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 0] = (R21 - R12) / (4 * q[:, 3])
    q[:, 1] = (R02 - R20) / (4 * q[:, 3])
    q[:, 2] = (R10 - R01) / (4 * q[:, 3])

    return q


def quat2rot(q):
    """Convert [B,4] quaternions to [B,3,3] rotation matrix"""
    b, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((b, 3, 3), device=q.device)

    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)

    return R


def from_dict_sample(T, to_global=False, zero_origin=False, to_matrix=False):
    """
    Create poses from a sample dictionary

    Parameters
    ----------
    T : Dict
        Dictionary containing input poses [B,4,4]
    to_global : Bool
        Whether poses should be converted to global frame of reference
    zero_origin : Bool
        Whether the target camera should be the center of the frame of reference
    to_matrix : Bool
        Whether output poses should be classes or tensors

    Returns
    -------
    pose : Dict
        Dictionary containing output poses
    """
    pose = {key: Pose(val) for key, val in T.items()}
    if to_global:
        pose = to_global_pose(pose, zero_origin=zero_origin)
    if to_matrix:
        pose = {key: val.T for key, val in pose.items()}
    return pose


def from_dict_batch(T, **kwargs):
    """Create poses from a batch dictionary"""
    pose_batch = [from_dict_sample({key: val[b] for key, val in T.items()}, **kwargs)
                  for b in range(T[0].shape[0])]
    return {key: torch.stack([v[key] for v in pose_batch], 0) for key in pose_batch[0]}


class Pose:
    """
    Pose class for 3D operations

    Parameters
    ----------
    T : torch.Tensor or Int
        Transformation matrix [B,4,4], or batch size (poses initialized as identity)
    """
    def __init__(self, T=1):
        if is_int(T):
            T = torch.eye(4).repeat(T, 1, 1)
        self.T = T if T.dim() == 3 else T.unsqueeze(0)

    def __len__(self):
        """Return batch size"""
        return len(self.T)

    def __getitem__(self, i):
        """Return batch-wise pose"""
        return Pose(self.T[[i]])

    def __mul__(self, data):
        """Transforms data (pose or 3D points)"""
        if isinstance(data, Pose):
            return Pose(self.T.bmm(data.T))
        elif isinstance(data, torch.Tensor):
            return self.T[:, :3, :3].bmm(data) + self.T[:, :3, -1].unsqueeze(-1)
        else:
            raise NotImplementedError()

    def detach(self):
        """Return detached pose"""
        return Pose(self.T.detach())

    @property
    def shape(self):
        """Return pose shape"""
        return self.T.shape

    @property
    def device(self):
        """Return pose device"""
        return self.T.device

    @property
    def dtype(self):
        """Return pose type"""
        return self.T.dtype

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Initializes as a [4,4] identity matrix"""
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))

    @staticmethod
    def from_dict(T, to_global=False, zero_origin=False, to_matrix=False):
        """Create poses from a dictionary"""
        if T[0].dim() == 3:
            return from_dict_sample(T, to_global=to_global, zero_origin=zero_origin, to_matrix=to_matrix)
        elif T[0].dim() == 4:
            return from_dict_batch(T, to_global=to_global, zero_origin=zero_origin, to_matrix=True)

    @classmethod
    def from_vec(cls, vec, mode):
        """Initializes from a [B,6] batch vector"""
        mat = pose_vec2mat(vec, mode)
        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]
        return cls(pose)

    def repeat(self, *args, **kwargs):
        """Repeats the transformation matrix multiple times"""
        self.T = self.T.repeat(*args, **kwargs)
        return self

    def inverse(self):
        """Returns a new Pose that is the inverse of this one"""
        return Pose(invert_pose(self.T))

    def to(self, *args, **kwargs):
        """Copy pose to device"""
        self.T = self.T.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Copy pose to CUDA"""
        self.to('cuda')
        return self

    def translate(self, xyz):
        """Translate pose"""
        self.T[:, :3, -1] = self.T[:, :3, -1] + xyz.to(self.device)
        return self

    def rotate(self, rpw):
        """Rotate pose"""
        rot = euler2mat(rpw)
        T = invert_pose(self.T).clone()
        T[:, :3, :3] = T[:, :3, :3] @ rot.to(self.device)
        self.T = invert_pose(T)
        return self

    def rotateRoll(self, r):
        """Rotate pose in the roll axis"""
        return self.rotate(torch.tensor([[0, 0, r]]))

    def rotatePitch(self, p):
        """Rotate pose in the pitcv axis"""
        return self.rotate(torch.tensor([[p, 0, 0]]))

    def rotateYaw(self, w):
        """Rotate pose in the yaw axis"""
        return self.rotate(torch.tensor([[0, w, 0]]))

    def translateForward(self, t):
        """Translate pose forward"""
        return self.translate(torch.tensor([[0, 0, -t]]))

    def translateBackward(self, t):
        """Translate pose backward"""
        return self.translate(torch.tensor([[0, 0, +t]]))

    def translateLeft(self, t):
        """Translate pose left"""
        return self.translate(torch.tensor([[+t, 0, 0]]))

    def translateRight(self, t):
        """Translate pose right"""
        return self.translate(torch.tensor([[-t, 0, 0]]))

    def translateUp(self, t):
        """Translate pose up"""
        return self.translate(torch.tensor([[0, +t, 0]]))

    def translateDown(self, t):
        """Translate pose down"""
        return self.translate(torch.tensor([[0, -t, 0]]))


#def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
import torch.nn as nn

from functools import lru_cache
import torch
import torch.nn as nn


def pixel_grid(hw, b=None, with_ones=False, device=None, normalize=False):
    """
    Creates a pixel grid for image operations
    Parameters
    ----------
    hw : Tuple
        Height/width of the grid
    b : Int
        Batch size
    with_ones : Bool
        Stack an extra channel with 1s
    device : String
        Device where the grid will be created
    normalize : Bool
        Whether the grid is normalized between [-1,1]
    Returns
    -------
    grid : torch.Tensor
        Output pixel grid [B,2,H,W]
    """
    if is_tensor(hw):
        b, hw = hw.shape[0], hw.shape[-2:]
    if is_tensor(device):
        device = device.device
    hi, hf = 0, hw[0] - 1
    wi, wf = 0, hw[1] - 1
    yy, xx = torch.meshgrid([torch.linspace(hi, hf, hw[0], device=device),
                             torch.linspace(wi, wf, hw[1], device=device)], indexing='ij')
    if with_ones:
        grid = torch.stack([xx, yy, torch.ones(hw, device=device)], 0)
    else:
        grid = torch.stack([xx, yy], 0)
    if b is not None:
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    if normalize:
        grid = norm_pixel_grid(grid)
    return grid


def norm_pixel_grid(grid, hw=None, in_place=False):
    """
    Normalize a pixel grid to be between [0,1]
    Parameters
    ----------
    grid : torch.Tensor
        Grid to be normalized [B,2,H,W]
    hw : Tuple
        Height/Width for normalization
    in_place : Bool
        Whether the operation is done in place or not
    Returns
    -------
    grid : torch.Tensor
        Normalized grid [B,2,H,W]
    """
    if hw is None:
        hw = grid.shape[-2:]
    if not in_place:
        grid = grid.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / (hw[1] - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (hw[0] - 1) - 1.0
    return grid


class UCMCamera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for the unified camera model (UCM).
    """
    def __init__(self, I, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        I : torch.Tensor [5]
            Camera intrinsics parameter vector
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.I = I
        if Tcw is None:
            self.Tcw = Pose.identity(len(I))
        elif isinstance(Tcw, Pose):
            self.Tcw = Tcw
        else:
            self.Tcw = Pose(Tcw)

        self.Tcw.to(self.I.device)

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.I)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.I = self.I.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

########################################################################################################################

    @property
    def fx(self):
        """Focal length in x"""
        return self.I[:, 0].unsqueeze(1).unsqueeze(2)

    @property
    def fy(self):
        """Focal length in y"""
        return self.I[:, 1].unsqueeze(1).unsqueeze(2)

    @property
    def cx(self):
        """Principal point in x"""
        return self.I[:, 2].unsqueeze(1).unsqueeze(2)

    @property
    def cy(self):
        """Principal point in y"""
        return self.I[:, 3].unsqueeze(1).unsqueeze(2)

    @property
    def alpha(self):
        """alpha in UCM model"""
        return self.I[:, 4].unsqueeze(1).unsqueeze(2)

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

########################################################################################################################

    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """

        if depth is None:
            return None
        b, c, h, w = depth.shape
        assert c == 1

        grid = pixel_grid(depth, with_ones=True, device=depth.device)

        # Estimate the outward rays in the camera frame
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha # [B,1,1]

        if torch.any(torch.isnan(alpha)):
            raise ValueError('alpha is nan')

        u = grid[:,0,:,:]
        v = grid[:,1,:,:]

        mx = (u - cx) / fx * (1 - alpha)
        my = (v - cy) / fy * (1 - alpha)
        r_square = mx ** 2 + my ** 2
        xi = alpha / (1 - alpha) # [B, 1, 1]
        coeff = (xi + torch.sqrt(1 + (1 - xi ** 2) * r_square)) / (1 + r_square) # [B, H, W]
        
        x = coeff * mx
        y = coeff * my
        z = coeff * 1 - xi
        z = z.clamp(min=1e-7)
        
        x_norm = x / z
        y_norm = y / z
        z_norm = z / z
        xnorm = torch.stack(( x_norm, y_norm, z_norm ), dim=1).float()

        # Scale rays to metric depth
        Xc = xnorm * depth

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return (self.Twc * Xc.view(b, 3, -1)).view(b,3,h,w)
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, frame='w', return_z=True):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            X = X
        elif frame == 'w':
            X = (self.Tcw * X.view(B,3,-1)).view(B,3,H,W)
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        
        d = torch.norm(X, dim=1)
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha
        x, y, z = X[:,0,:], X[:,1,:], X[:,2,:]
        z = z.clamp(min=1e-7)
        
        Xnorm = fx * x / (alpha * d + (1 - alpha) * z + 1e-7) + cx
        Ynorm = fy * y / (alpha * d + (1 - alpha) * z + 1e-7) + cy
        Xnorm = 2 * Xnorm / (W-1) - 1
        Ynorm = 2 * Ynorm / (H-1) - 1

        coords = torch.stack([Xnorm, Ynorm], dim=-1).permute(0,3,1,2)
        z = z.unsqueeze(1)

        invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                      (coords[:, 1] < -1) | (coords[:, 1] > 1) | (z[:, 0] < 0)
        coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        # Return pixel coordinates
        if return_z:
            return coords.permute(0, 2, 3, 1), z
        return coords.permute(0, 2, 3, 1)

    def reconstruct_depth_map(self, depth, to_world=True):
        if to_world:
            return self.reconstruct(depth, frame='w')
        else:
            return self.reconstruct(depth, frame='c')

    def project_points(self, points, from_world=True, normalize=True, return_z=True):
        if from_world:
            return self.project(points, return_z=return_z, frame='w')
        else:
            return self.project(points, return_z=return_z, frame='c')

    def coords_from_depth(self, depth, ref_cam=None):
        if ref_cam is None:
            return self.project_points(self.reconstruct_depth_map(depth, to_world=False), from_world=True)
        else:
            return ref_cam.project_points(self.reconstruct_depth_map(depth, to_world=True), from_world=True)

        
class Camera(nn.Module, ABC):
    """
    Camera class for 3D reconstruction

    Parameters
    ----------
    K : torch.Tensor
        Camera intrinsics [B,3,3]
    hw : Tuple
        Camera height and width
    Twc : Pose or torch.Tensor
        Camera pose (world to camera) [B,4,4]
    Tcw : Pose or torch.Tensor
        Camera pose (camera to world) [B,4,4]
    """
    def __init__(self, K, hw, Twc=None, Tcw=None):
        super().__init__()

        # Asserts

        assert Twc is None or Tcw is None

        # Fold if multi-batch

        if K.dim() == 4:
            K = rearrange(K, 'b n h w -> (b n) h w')
            if Twc is not None:
                Twc = rearrange(Twc, 'b n h w -> (b n) h w')
            if Tcw is not None:
                Tcw = rearrange(Tcw, 'b n h w -> (b n) h w')

        # Intrinsics

        if same_shape(K.shape[-2:], (3, 3)):
            self._K = torch.eye(4, dtype=K.dtype, device=K.device).repeat(K.shape[0], 1, 1)
            self._K[:, :3, :3] = K
        else:
            self._K = K

        # Pose

        if Twc is None and Tcw is None:
            self._Twc = torch.eye(4, dtype=K.dtype, device=K.device).unsqueeze(0).repeat(K.shape[0], 1, 1)
        else:
            self._Twc = invert_pose(Tcw) if Tcw is not None else Twc
        if is_tensor(self._Twc):
            self._Twc = Pose(self._Twc)

        # Resolution

        self._hw = hw
        if is_tensor(self._hw):
            self._hw = self._hw.shape[-2:]

    def __getitem__(self, idx):
        """Return batch-wise pose"""
        if is_seq(idx):
            return type(self).from_list([self.__getitem__(i) for i in idx])
        else:
            return type(self)(
                K=self._K[[idx]],
                Twc=self._Twc[[idx]] if self._Twc is not None else None,
                hw=self._hw,
            )

    def __len__(self):
        """Return length as intrinsics batch"""
        return self._K.shape[0]

    def __eq__(self, cam):
        """Check if two cameras are the same"""
        if not isinstance(cam, type(self)):
            return False
        if self._hw[0] != cam.hw[0] or self._hw[1] != cam.hw[1]:
            return False
        if not torch.allclose(self._K, cam.K):
            return False
        if not torch.allclose(self._Twc.T, cam.Twc.T):
            return False
        return True

    def clone(self):
        """Return a copy of this camera"""
        return deepcopy(self)

    @property
    def pose(self):
        """Return camera pose (world to camera)"""
        return self._Twc.T

    @property
    def K(self):
        """Return camera intrinsics"""
        return self._K

    @K.setter
    def K(self, K):
        """Set camera intrinsics"""
        self._K = K

    @property
    def invK(self):
        """Return inverse of camera intrinsics"""
        return invert_intrinsics(self._K)

    @property
    def batch_size(self):
        """Return batch size"""
        return self._Twc.T.shape[0]

    @property
    def hw(self):
        """Return camera height and width"""
        return self._hw

    @hw.setter
    def hw(self, hw):
        """Set camera height and width"""
        self._hw = hw

    @property
    def wh(self):
        """Get camera width and height"""
        return self._hw[::-1]

    @property
    def n_pixels(self):
        """Return number of pixels"""
        return self._hw[0] * self._hw[1]

    @property
    def fx(self):
        """Return horizontal focal length"""
        return self._K[:, 0, 0]

    @property
    def fy(self):
        """Return vertical focal length"""
        return self._K[:, 1, 1]

    @property
    def cx(self):
        """Return horizontal principal point"""
        return self._K[:, 0, 2]

    @property
    def cy(self):
        """Return vertical principal point"""
        return self._K[:, 1, 2]

    @property
    def fxy(self):
        """Return focal length"""
        return torch.tensor([self.fx, self.fy], dtype=self.dtype, device=self.device)

    @property
    def cxy(self):
        """Return principal points"""
        return self._K[:, :2, 2]
        # return torch.tensor([self.cx, self.cy], dtype=self.dtype, device=self.device)

    @property
    def Tcw(self):
        """Return camera pose (camera to world)"""
        return None if self._Twc is None else self._Twc.inverse()

    @Tcw.setter
    def Tcw(self, Tcw):
        """Set camera pose (camera to world)"""
        self._Twc = Tcw.inverse()

    @property
    def Twc(self):
        """Return camera pose (world to camera)"""
        return self._Twc

    @Twc.setter
    def Twc(self, Twc):
        """Set camera pose (world to camera)"""
        self._Twc = Twc

    @property
    def dtype(self):
        """Return tensor type"""
        return self._K.dtype

    @property
    def device(self):
        """Return device"""
        return self._K.device

    def detach_pose(self):
        """Detach pose from the graph"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def detach_K(self):
        """Detach intrinsics from the graph"""
        return type(self)(K=self._K.detach(), hw=self._hw, Twc=self._Twc)

    def detach(self):
        """Detach camera from the graph"""
        return type(self)(K=self._K.detach(), hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def inverted_pose(self):
        """Invert camera pose"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.inverse() if self._Twc is not None else None)

    def no_translation(self):
        """Return new camera without translation"""
        Twc = self.pose.clone()
        Twc[:, :-1, -1] = 0
        return type(self)(K=self._K, hw=self._hw, Twc=Twc)

    @staticmethod
    def from_dict(K, hw, Twc=None):
        """Create cameras from a pose dictionary"""
        return {key: Camera(K=K[0], hw=hw[0], Twc=val) for key, val in Twc.items()}

    # @staticmethod
    # def from_dict(K, hw, Twc=None):
    #     return {key: Camera(K=K[(0, 0)], hw=hw[(0, 0)], Twc=val) for key, val in Twc.items()}

    @staticmethod
    def from_list(cams):
        """Create cameras from a list"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return Camera(K=K, Twc=Twc, hw=cams[0].hw)

    def scaled(self, scale_factor):
        """Return a scaled camera"""
        if scale_factor is None or scale_factor == 1:
            return self
        if is_seq(scale_factor):
            if len(scale_factor) == 4:
                scale_factor = scale_factor[-2:]
            scale_factor = [float(scale_factor[i]) / float(self._hw[i]) for i in range(2)]
        else:
            scale_factor = [scale_factor] * 2
        return type(self)(
            K=scale_intrinsics(self._K, scale_factor),
            hw=[int(self._hw[i] * scale_factor[i]) for i in range(len(self._hw))],
            Twc=self._Twc
        )

    def offset_start(self, start):
        """Offset camera intrinsics based on a crop"""
        new_cam = self.clone()
        start = start.to(self.device)
        new_cam.K[:, 0, 2] -= start[:, 1]
        new_cam.K[:, 1, 2] -= start[:, 0]
        return new_cam

    def interpolate(self, rgb):
        """Interpolate an image to fit the camera"""
        if rgb.dim() == 5:
            rgb = rearrange(rgb, 'b n c h w -> (b n) c h w')
        return interpolate(rgb, scale_factor=None, size=self.hw, mode='bilinear', align_corners=True)

    def interleave_K(self, b):
        """Interleave camera intrinsics to fit multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=self._Twc,
            hw=self._hw,
        )

    def interleave_Twc(self, b):
        """Interleave camera pose to fit multiple batches"""
        return type(self)(
            K=self._K,
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def interleave(self, b):
        """Interleave camera to fit multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def Pwc(self, from_world=True):
        """Return projection matrix"""
        return self._K[:, :3] if not from_world or self._Twc is None else \
            torch.matmul(self._K, self._Twc.T)[:, :3]

    def to_world(self, points):
        """Transform points to world coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self.Tcw is None else self.Tcw * points

    def from_world(self, points):
        """Transform points back to camera coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self._Twc is None else \
            torch.matmul(self._Twc.T, cat_channel_ones(points, 1))[:, :3]

    def to(self, *args, **kwargs):
        """Copy camera to device"""
        self._K = self._K.to(*args, **kwargs)
        if self._Twc is not None:
            self._Twc = self._Twc.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Copy camera to CUDA"""
        return self.to('cuda')

    def relative_to(self, cam):
        """Create a new camera relative to another camera"""
        return Camera(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc.inverse())

    def global_from(self, cam):
        """Create a new camera in global coordinates relative to another camera"""
        return Camera(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc)

    def reconstruct_depth_map(self, depth, to_world=False):
        """
        Reconstruct a depth map from the camera viewpoint

        Parameters
        ----------
        depth : torch.Tensor
            Input depth map [B,1,H,W]
        to_world : Bool
            Transform points to world coordinates

        Returns
        -------
        points : torch.Tensor
            Output 3D points [B,3,H,W]
        """
        if depth is None:
            return None
        b, _, h, w = depth.shape
        grid = pixel_grid(depth, with_ones=True, device=depth.device).view(b, 3, -1)
        points = depth.view(b, 1, -1) * torch.matmul(self.invK[:, :3, :3], grid)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        return points.view(b, 3, h, w)

    def reconstruct_cost_volume(self, volume, to_world=True, flatten=True):
        """
        Reconstruct a cost volume from the camera viewpoint

        Parameters
        ----------
        volume : torch.Tensor
            Input depth map [B,1,D,H,W]
        to_world : Bool
            Transform points to world coordinates
        flatten: Bool
            Flatten volume points

        Returns
        -------
        points : torch.Tensor
            Output 3D points [B,3,D,H,W]
        """
        c, d, h, w = volume.shape
        grid = pixel_grid((h, w), with_ones=True, device=volume.device).view(3, -1).repeat(1, d)
        points = torch.stack([
            (volume.view(c, -1) * torch.matmul(invK[:3, :3].unsqueeze(0), grid)).view(3, d * h * w)
            for invK in self.invK], 0)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        if flatten:
            return points.view(-1, 3, d, h * w).permute(0, 2, 1, 3)
        else:
            return points.view(-1, 3, d, h, w)

    def project_points(self, points, from_world=True, normalize=True, return_z=False):
        """
        Project points back to image plane

        Parameters
        ----------
        points : torch.Tensor
            Input 3D points [B,3,H,W] or [B,3,N]
        from_world : Bool
            Whether points are in the global frame
        normalize : Bool
            Whether projections should be normalized to [-1,1]
        return_z : Bool
            Whether projected depth is return as well

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        depth : torch.Tensor
            Projected depth [B,1,H,W]
        """
        is_depth_map = points.dim() == 4
        hw = self._hw if not is_depth_map else points.shape[-2:]

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))

        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-7)
        depth = points[:, 2]

        if not is_depth_map:
            if normalize:
                coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
                invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                          (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth < 0)
                coords[invalid.unsqueeze(1).repeat(1, 2, 1)] = -2
            if return_z:
                return coords.permute(0, 2, 1), depth
            else:
                return coords.permute(0, 2, 1)

        coords = coords.view(b, 2, *hw)
        depth = depth.view(b, 1, *hw)

        if normalize:
            coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
            invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                      (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth[:, 0] < 0)
            coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        if return_z:
            return coords.permute(0, 2, 3, 1), depth
        else:
            return coords.permute(0, 2, 3, 1)

    def project_cost_volume(self, points, from_world=True, normalize=True):
        """
        Project points back to image plane

        Parameters
        ----------
        points : torch.Tensor
            Input 3D points [B,3,H,W] or [B,3,N]
        from_world : Bool
            Whether points are in the global frame
        normalize : Bool
            Whether projections should be normalized to [-1,1]

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        """
        if points.dim() == 4:
            points = points.permute(0, 2, 1, 3).reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))

        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-7)
        coords = coords.view(b, 2, -1, *self._hw).permute(0, 2, 3, 4, 1)

        if normalize:
            coords[..., 0] /= self._hw[1] - 1
            coords[..., 1] /= self._hw[0] - 1
            return 2 * coords - 1
        else:
            return coords

    def coords_from_cost_volume(self, volume, ref_cam=None):
        """
        Get warp coordinates from a cost volume

        Parameters
        ----------
        volume : torch.Tensor
            Input cost volume [B,1,D,H,W]
        ref_cam : Camera
            Optional to generate cross-camera coordinates

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        """
        if ref_cam is None:
            return self.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=False), from_world=True)
        else:
            return ref_cam.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=True), from_world=True)

    def coords_from_depth(self, depth, ref_cam=None):
        """
        Get warp coordinates from a depth map

        Parameters
        ----------
        depth : torch.Tensor
            Input cost volume [B,1,D,H,W]
        ref_cam : Camera
            Optional to generate cross-camera coordinates

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        """
        if ref_cam is None:
            return self.project_points(self.reconstruct_depth_map(depth, to_world=False), from_world=True, return_z=True)
        else:
            return ref_cam.project_points(self.reconstruct_depth_map(depth, to_world=True), from_world=True, return_z=True)
import torch.nn.functional as tfn


def grid_sample(tensor, grid, padding_mode, mode, align_corners):
    return tfn.grid_sample(tensor, grid,
        padding_mode=padding_mode, mode=mode, align_corners=align_corners)


def invert_intrinsics(K):
    """Invert camera intrinsics"""
    Kinv = K.clone()
    Kinv[:, 0, 0] = 1. / K[:, 0, 0]
    Kinv[:, 1, 1] = 1. / K[:, 1, 1]
    Kinv[:, 0, 2] = -1. * K[:, 0, 2] / K[:, 0, 0]
    Kinv[:, 1, 2] = -1. * K[:, 1, 2] / K[:, 1, 1]
    return Kinv

def interpolate(tensor, size, scale_factor, mode, align_corners):
    """
    Interpolate a tensor to a different resolution

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor [B,?,H,W]
    size : Tuple
        Interpolation size (H,W)
    scale_factor : Float
        Scale factor for interpolation
    mode : String
        Interpolation mode
    align_corners : Bool
        Corner alignment flag

    Returns
    -------
    tensor : torch.Tensor
        Interpolated tensor [B,?,h,w]
    """
    if is_tensor(size):
        size = size.shape[-2:]
    return tfn.interpolate(
        tensor, size=size, scale_factor=scale_factor,
        mode=mode, align_corners=align_corners, recompute_scale_factor=False,
    )


class ViewSynthesis(nn.Module, ABC):
    """
    Class for view synthesis calculation based on image warping

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.grid_sample = partial(
            grid_sample, mode='bilinear', padding_mode='border', align_corners=True)
        self.interpolate = partial(
            interpolate, mode='bilinear', scale_factor=None, align_corners=True)
        self.grid_sample_zeros = partial(
            grid_sample, mode='nearest', padding_mode='zeros', align_corners=True)
        self.upsample_depth = False
    @staticmethod
    def get_num_scales(depths, optical_flow):
        """Return number of scales based on input"""
        if depths is not None:
            return len(depths)
        if optical_flow is not None:
            return len(optical_flow)
        else:
            raise ValueError('Invalid inputs for view synthesis')

    @staticmethod
    def get_tensor_ones(depths, optical_flow, scale):
        """Return unitary tensor based on input"""
        if depths is not None:
            return torch.ones_like(depths[scale])
        elif optical_flow is not None:
            b, _, h, w = optical_flow[scale].shape
            return torch.ones((b, 1, h, w), device=optical_flow[scale].device)
        else:
            raise ValueError('Invalid inputs for view synthesis')

    def get_coords(self, rgbs, depths, cams, context, scale):
        """
        Calculate projection coordinates for warping

        Parameters
        ----------
        rgbs : list[torch.Tensor]
            Input images (for dimensions) [B,3,H,W]
        depths : list[torch.Tensor]
            Target depth maps [B,1,H,W]
        cams : list[Camera]
            Input cameras
        optical_flow : list[torch.Tensor]
            Input optical flow for alternative warping
        context : list[Int]
            Context indices
        scale : Int
            Current scale
        tgt : Int
            Target index

        Returns
        -------
        output : Dict
            Dictionary containing warped images and masks
        """
        if depths is not None and cams is not None:
            cams_tgt = cams[0] if is_list(cams) else cams
            cams_ctx = cams[1] if is_list(cams) else cams
            depth = depths[scale]
            return {
                ctx: cams_tgt[0].coords_from_depth(depth, cams_ctx[ctx]) for ctx in context
            }
        else:
            raise ValueError('Invalid input for view synthesis')

    def forward(self, rgbs, depths=None, 
                #tgt_depths=None, 
                cams=None,
                return_masks=False, tgt=0):

        num_scales = 1
        warps, warped_depths, masks = [], [], []
        scale = 0
        
        coords, warped_depths = self.get_coords(rgbs, depths, cams, [1], scale)[1]
        src=0
        print(coords.shape)
        warps = self.grid_sample(
            rgbs[1][src], coords.type(rgbs[1][src].dtype))
        #warped_depths  = self.grid_sample(
        #    depths[src], coords[1].type(rgbs[1][src].dtype))
        #computed_depths = self.grid_sample(
        #    tgt_depths[tgt], coords[1].type(rgbs[1][src].dtype)
        #)
        if return_masks:
            ones = self.get_tensor_ones(depths, None, scale)
            masks = self.grid_sample_zeros(
                ones, coords.type(ones.dtype))

        return {
            'warps': warps,
            'warped_depths': warped_depths,
            #'computed_depths': computed_depths,
            'masks': masks if return_masks else None
        }
def same_shape(shape1, shape2):
    """Checks if two shapes are the same"""
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

def cat_channel_ones(tensor, n=1):
    """
    Concatenate tensor with an extra channel of ones

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be concatenated
    n : Int
        Which channel will be concatenated

    Returns
    -------
    cat_tensor : torch.Tensor
        Concatenated tensor
    """
    # Get tensor shape with 1 channel
    shape = list(tensor.shape)
    shape[n] = 1
    # Return concatenation of tensor with ones
    return torch.cat([tensor, torch.ones(shape,
                      device=tensor.device, dtype=tensor.dtype)], n)


def inverse_warp_ucm(img, depth, ref_depth, pose, intrinsic_a, intrinsic_b, padding_mode):
    cam1 = UCMCamera(intrinsic_b)
    if pose.shape[-1] == 6:
        cam0 = UCMCamera(intrinsic_a, Tcw = pose_vec2mat(-pose, mode='euler'))
    else:
        cam0 = UCMCamera(intrinsic_a, Tcw = torch.linalg.inv(pose))

    coords, warped_depths = cam0.coords_from_depth(depth, cam1)
    projected_img = torch.nn.functional.grid_sample(img, coords,
        padding_mode=padding_mode, mode='bilinear', align_corners=False)
    projected_mask = torch.nn.functional.grid_sample(torch.ones_like(depth), coords,
        padding_mode=padding_mode, mode='bilinear', align_corners=False)
    projected_depth = torch.nn.functional.grid_sample((ref_depth), coords,
        padding_mode=padding_mode, mode='bilinear', align_corners=False)

    return projected_img, projected_mask, projected_depth,  warped_depths


def inverse_warp_ucm3(img, depth, ref_depth, pose, intrinsic, padding_mode):
    cam1 = UCMCamera(intrinsic)
    cam0 = UCMCamera(intrinsic, Tcw = -pose)
 

    coords, warped_depths = cam0.coords_from_depth(depth, cam1)
    projected_img = torch.nn.functional.grid_sample(img, coords,
        padding_mode=padding_mode, mode='bilinear', align_corners=False)
    projected_mask = torch.nn.functional.grid_sample(torch.ones_like(depth), coords,
        padding_mode=padding_mode, mode='bilinear', align_corners=False)
    projected_depth = torch.nn.functional.grid_sample((ref_depth), coords,
        padding_mode=padding_mode, mode='bilinear', align_corners=False)

    return projected_img, projected_mask, projected_depth,  warped_depths



def rectify(img, mask, depth, intrinsic):
    
    with torch.no_grad():
        cam1 = UCMCamera(intrinsic.unsqueeze(0))
        linear_intrinsic = torch.tensor([[
                [intrinsic[0], 0,  intrinsic[2]],
                [0,  intrinsic[1],  intrinsic[3]],
                [0, 0, 1]
            ]])
        cam0 = Camera(linear_intrinsic, hw=(img.shape[2], img.shape[1]))

        coords, warped_depths = cam0.coords_from_depth(depth, cam1)
        projected_img = torch.nn.functional.grid_sample(img, coords,
            padding_mode='zeros', mode='bilinear', align_corners=False)
        projected_mask = torch.nn.functional.grid_sample(mask, coords,
            padding_mode='zeros', mode='nearest', align_corners=False)
        projected_depth = torch.nn.functional.grid_sample(depth, coords,
            padding_mode='zeros', mode='bilinear', align_corners=False)
        return projected_img.squeeze().cpu().numpy(), projected_mask.squeeze().cpu().numpy(), projected_depth.squeeze().cpu().numpy()