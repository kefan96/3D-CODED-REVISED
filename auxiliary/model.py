from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from my_utils import sampleSphere
import trimesh
import pointcloud_processor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class PointNetfeat(nn.Module):
    def __init__(self, npoint=2500, nlatent=1024):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        print("bottleneck_size", bottleneck_size)
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    # input: batch_size, 1027, num_points
    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2 * self.th(self.conv4(x))
        return x.unsqueeze(-1)      # batch_size, 3, num_points, 1
    
class FeatureMerge(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(FeatureMerge, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 256, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        return x

class Template(object):
    def __init__(self, path, num_points=6890):
        self.init_template(path, num_points)

    def init_template(self, path, num_points):
        # if not os.path.exists("./data/template/" + path):
        #     os.system("chmod +x ./data/download_template.sh")
        #     os.system("./data/download_template.sh")
        self.num_points = num_points 
        mesh = trimesh.load("./data/template/" + path, process=False)
        self.mesh = mesh
        point_set = mesh.vertices
        point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)

        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set_HR = mesh_HR.vertices
        point_set_HR, _, _ = pointcloud_processor.center_bounding_box(point_set_HR)

        self.vertex = torch.from_numpy(point_set).cuda().float()
        # rand_columns = torch.randperm(self.vertex.size(0))[:self.num_points]
        # self.vertex = self.vertex[rand_columns, :]
        
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"
        self.prop = torch.from_numpy(self.prop).cuda().unsqueeze(0).float()
        print(f"Using template to initialize template")
        
class SelfAttention(nn.Module):
    def __init__(self, num_points=6890):
        super(SelfAttention, self).__init__()
        self.num_points = num_points
    
        self.lin1 = torch.nn.Linear(self.num_points * 3 * 3, self.num_points)
        self.lin2 = torch.nn.Linear(self.num_points, self.num_points * 3)
        self.sig = torch.sigmoid
        
        self.bn1 = torch.nn.BatchNorm1d(self.num_points)
        self.bn2 = torch.nn.BatchNorm1d(self.num_points * 3)
        
    def forward(self, x):
        batchsize = x.size()[0]
        h = x.view(batchsize, -1)
        h = F.relu(self.bn1(self.lin1(h).unsqueeze(-1)))
        h = self.sig(self.bn2(self.lin2(h.squeeze(2))))
        value, idx = torch.max(h, -1, keepdim = True)
        z = torch.zeros_like(h)
        z[torch.arange(h.size(0))[:, None, None], torch.arange(h.size(1))[None,:, None], idx] = 1
        z = z.view(batchsize, -1, 3, 1).expand(batchsize, -1, 3, 3)
        # print("Weights: ", h)
        # print("Coords: ", x)
        return z
        
    

# one encoder, multiple decoders
class OEMDNet(nn.Module):
    def __init__(self, num_points=6890, bottleneck_size=1024, point_translation=False, dim_template=3, patch_deformation=False, dim_out_patch=3):
        super(OEMDNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.point_translation = point_translation
        self.dim_template = dim_template
        self.patch_deformation = patch_deformation
        self.dim_out_patch = dim_out_patch
        self.dim_before_decoder = 3
        self.count = 0

        self.templates = [Template("template.ply"), Template("tr_reg_006.ply"), Template("tr_reg_009.ply")]
        # self.templates = [Template("template.ply"), Template("template.ply"), Template("template.ply")]

        self.encoder = PointNetfeat(num_points, bottleneck_size)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=self.dim_before_decoder + self.bottleneck_size), PointGenCon(bottleneck_size=self.dim_before_decoder + self.bottleneck_size), PointGenCon(bottleneck_size=self.dim_before_decoder + self.bottleneck_size)])
        self.attention = SelfAttention(num_points=6890)
        
    def morph_points(self, x, idx=None):
        if not idx is None:
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            
        rand_grids = [self.templates[i].vertex for i in range(3)] # num_points, 3
        if not idx is None:
            rand_grids = [r[idx, :].view(x.size(0), -1, self.dim_template).transpose(1, 2).contiguous() for r in rand_grids] 
        else:
            rand_grids = [r.transpose(0, 1).contiguous().unsqueeze(0).expand(x.size(0), self.dim_template, -1) for r in rand_grids]     # 3, num_points => batch_size, 3, num_points

        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grids[0].size(2)).contiguous() # batch_size, 1024, num_points
        ys = [torch.cat((r, y), 1).contiguous() for r in rand_grids]    # batch_size, 1027, num_pounts
        out = [self.decoder[i](ys[i]).contiguous().transpose(2, 1).contiguous() for i in range(3)]     # batch_size, 3, num_points, 1 => batch_size, num_points, 3, 1
        out = [out[i] + rand_grids[i].unsqueeze(-1).transpose(2, 1).contiguous() for i in range(3)]
        return torch.cat(out, 3)            # batch_size, num_points, 3, 3

    def decode(self, x, idx=None):
        x = self.morph_points(x, idx)
        w = self.attention(x)
        return torch.mean(x * w, 3), x, w

    def forward(self, x, idx=None):
        x = self.encoder(x)
        return self.decode(x, idx)
    
class MEODNet(nn.Module):
    def __init__(self, num_points=6890, bottleneck_size=1024, point_translation=False, dim_template=3, patch_deformation=False, dim_out_patch=3):
        super(MEODNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.point_translation = point_translation
        self.dim_template = dim_template
        self.patch_deformation = patch_deformation
        self.dim_out_patch = dim_out_patch
        self.dim_before_decoder = 3
        self.count = 0
        
        self.templates = [Template("template.ply"), Template("tr_reg_006.ply"), Template("tr_reg_009.ply")]

        self.encoder = PointNetfeat(num_points, bottleneck_size)
        self.mergers = nn.ModuleList([FeatureMerge(bottleneck_size=self.dim_before_decoder + self.bottleneck_size) for _ in range(3)])
        self.decoder = PointGenCon(bottleneck_size=256)
        
    def morph_points(self, x, idx=None):
        if not idx is None:
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
        rand_grids = [self.templates[i].vertex for i in range(3)] # num_points, 3
        
        if not idx is None:
            rand_grids = [r[idx, :].view(x.size(0), -1, self.dim_template).transpose(1, 2).contiguous() for r in rand_grids] 
        else:
            rand_grids = [r.transpose(0, 1).contiguous().unsqueeze(0).expand(x.size(0), self.dim_template, -1) for r in rand_grids]     # 3, num_points => batch_size, 3, num_points
        
        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grids[0].size(2)).contiguous() # batch_size, 1024, num_points
        ys = [torch.cat((r, y), 1).contiguous() for r in rand_grids]    # batch_size, 1027, num_pounts
        ys = [self.mergers[i](ys[i]).unsqueeze(3) for i in range(3)]     # batch_size, 256, num_points
        y = torch.cat(ys, 3).contiguous()    # batch_size, 256, num_points, 3
        y, _ = torch.max(y, 3)  # batch_size, 256, num_points
        
        return self.decoder(y).contiguous().transpose(2, 1).contiguous()
    
    def decode(self, x, idx=None):
        return self.morph_points(x, idx)
    
    def forward(self, x, idx=None):
        x = self.encoder(x)
        return self.decode(x, idx)
        
if __name__ == '__main__':
    a = OEMDNet()
    b = MEODNet()
