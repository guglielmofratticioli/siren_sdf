import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np


from mesh_to_sdf import get_surface_point_cloud
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from math import sqrt
import trimesh
import sys
import re

# import pyrender
# import copy
# import os
# from PIL import Image
# import skimage
# import time

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, has_skip=False, skip_idx=1):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.has_skip = has_skip
        self.skip_idx = skip_idx
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1. / self.in_features, 
                                             1. / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        intermediate = torch.sin(self.omega_0 * self.linear(input))
        if self.has_skip:
            intermediate = intermediate/self.skip_idx + input
        return intermediate
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 omega=30, first_linear=False):
        super().__init__()
        self.omega = omega
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.first_linear=first_linear
        self.net = []
        if first_linear:
            linear = nn.Linear(in_features, hidden_features)
            with torch.no_grad():
                linear.weight.uniform_(-1. / self.in_features / omega, 
                                        1. / self.in_features / omega) 
            self.net.append(linear)
        else:
            self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=omega))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=omega, has_skip=True, skip_idx=sqrt(i+1)))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega, 
                                              np.sqrt(6 / hidden_features) / omega)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=omega))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
        
class SDFFitting(Dataset):
    def __init__(self, filename, samples):
        super().__init__()
        mesh = trimesh.load(filename)
        #mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0
        surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method='sample')

        self.coords, self.samples = surface_point_cloud.sample_sdf_near_surface(samples//2, use_scans=False, sign_method='normal')
        unit_sphere_points = sample_uniform_points_in_unit_sphere(samples//2)
        samples = surface_point_cloud.get_sdf_in_batches(unit_sphere_points, use_depth_buffer=False)
        self.coords = np.concatenate([self.coords, unit_sphere_points]).astype(np.float32)
        self.samples = np.concatenate([self.samples, samples]).astype(np.float32)
        
        #colors = np.zeros(self.coords.shape)
        #colors[self.samples < 0, 2] = 1
        #colors[self.samples > 0, 0] = 1
        #cloud = pyrender.Mesh.from_points(self.coords, colors=colors)
        #scene = pyrender.Scene()
        #scene.add(cloud)
        #viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

        self.samples = torch.from_numpy(self.samples)[:,None]
        self.coords = torch.from_numpy(self.coords)
        print(self.coords.shape, self.samples.shape)
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.samples
        
def dump_data(dat):
  dat = dat.cpu().detach().numpy()
  return dat

def print_vec4(ws):
  vec = "vec4(" + ",".join(["{0:.2f}".format(w) for w in ws]) + ")"
  vec = re.sub(r"\b0\.", ".", vec)
  return vec

def print_mat4(ws):
  mat = "mat4(" + ",".join(["{0:.2f}".format(w) for w in np.transpose(ws).flatten()]) + ")"
  mat = re.sub(r"\b0\.", ".", mat)
  return mat

def serialize_to_shadertoy(siren, varname):
  #first layer
  omega = siren.omega
  chunks = int(siren.hidden_features/4)
  lin = siren.net[0] if siren.first_linear else siren.net[0].linear
  in_w = dump_data(lin.weight)
  in_bias = dump_data(lin.bias)
  om = 1 if siren.first_linear else omega
  for row in range(chunks):
    if siren.first_linear:
        line = "vec4 %s0_%d=(" % (varname, row)
    else:
        line = "vec4 %s0_%d=sin(" % (varname, row)

    for ft in range(siren.in_features):
        feature = x_vec = in_w[row*4:(row+1)*4,ft]*om
        line += ("p.%s*" % ["y","z","x"][ft]) + print_vec4(feature) + "+"
    bias = in_bias[row*4:(row+1)*4]*om
    line += print_vec4(bias) + ");"
    print(line)

  #hidden layers
  for layer in range(siren.hidden_layers):
    layer_w = dump_data(siren.net[layer+1].linear.weight)
    layer_bias = dump_data(siren.net[layer+1].linear.bias)
    for row in range(chunks):
      line = ("vec4 %s%d_%d" % (varname, layer+1, row)) + "=sin("
      for col in range(chunks):
        mat = layer_w[row*4:(row+1)*4,col*4:(col+1)*4]*omega
        line += print_mat4(mat) + ("*%s%d_%d"%(varname, layer, col)) + "+\n    "
      bias = layer_bias[row*4:(row+1)*4]*omega
      line += print_vec4(bias)+")/%0.1f+%s%d_%d;"%(sqrt(layer+1), varname, layer, row)
      print(line)

  #output layer
  out_w = dump_data(siren.net[-1].weight)
  out_bias = dump_data(siren.net[-1].bias)
  for outf in range(siren.out_features):
    line = "return "
    for row in range(chunks):
      vec = out_w[outf,row*4:(row+1)*4]
      line += ("dot(%s%d_%d,"%(varname, siren.hidden_layers, row)) + print_vec4(vec) + ")+\n    "
    print(line + "{:0.3f}".format(out_bias[outf])+";") 
    
def train_siren(dataloader, hidden_features, hidden_layers, omega, steps = 20000, device="mps"):
  model_input, ground_truth = next(iter(dataloader))
  if device == "mps":
    model_input, ground_truth = model_input.to('mps'), ground_truth.to('mps')
  elif device == "cuda":
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
  else : 
    model_input, ground_truth = model_input.cpu(), ground_truth.cpu()

  img_curr = Siren(in_features=3, out_features=1, hidden_features=hidden_features, 
                   hidden_layers=hidden_layers, outermost_linear=True, omega=omega, first_linear=False)
  if device == "mps":
    img_curr.to('mps')
  elif device == "cuda":
    img_curr.cuda()
  else : 
    img_curr.cpu()

  #optim = torch.optim.Adagrad(params=img_curr.parameters())
  #optim = torch.optim.Adam(lr=1e-3, params=img_curr.parameters())
  optim = torch.optim.Adam(lr=1e-4, params=img_curr.parameters(), weight_decay=.01)
  perm = torch.randperm(model_input.size(1))

  total_steps = steps
  update = int(total_steps/50)
  batch_size = 256*256
  for step in range(total_steps):
    if step == 500:
        optim.param_groups[0]['weight_decay'] = 0.
    idx = step % int(model_input.size(1)/batch_size)
    model_in = model_input[:,perm[batch_size*idx:batch_size*(idx+1)],:]
    truth = ground_truth[:,perm[batch_size*idx:batch_size*(idx+1)],:]
    model_output, coords = img_curr(model_in)

    loss = (model_output - truth)**2
    loss = loss.mean()

    optim.zero_grad()
    loss.backward()
    optim.step()
           
    if (step % update) == update-1:
      perm = torch.randperm(model_input.size(1))
      print("Step %d, Current loss %0.6f" % (step, loss))

  return img_curr


def convert(path=None, steps=20000, omega=15, hidden_features=16, hidden_layers=2  ):
    if path is not None:
        mesh_path = path
    elif len(sys.argv) < 1:
        print("Usage: sirend_sdf <path_to_mesh> <steps> <omega> <hidden_features> <hidden_layers>")
        sys.exit(1)
    else: 
        if len(sys.argv)>=1:
            mesh_path = sys.argv[1]
        if len(sys.argv)>=2:
            steps = int(sys.argv[2])
        if len(sys.argv)>=3:
            omega = int(sys.argv[3])
        if len(sys.argv)>=4:
            hidden_layers = int(sys.argv[4])
        if len(sys.argv)>=5:
            omega = int(sys.argv[5])
        
    

    sdf = SDFFitting(mesh_path, 256 * 256 * 4)
    print("Fitted SDF for %s" % mesh_path)
    sdfloader = DataLoader(sdf, batch_size=1, pin_memory=False, num_workers=0)
    sdf_siren = train_siren(sdfloader, hidden_features, hidden_layers, omega, steps)
    print("> > > Trained Siren, here's the shader Code")

    serialize_to_shadertoy(sdf_siren, "f")

if __name__ == "__main__":
    convert()
   