import io
import os
import time
import json
import torch
import zipfile
import numpy as np
import torch.nn as nn
from PIL import Image,ImageOps
import torch.nn.functional as F
from vidaug import augmentors as va
from einops import rearrange, repeat
import math
from torch import einsum
from argparse import ArgumentParser
from core.models.curvenet_cls import CurveNet
from tqdm import tqdm
from torchsummary import summary
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
np.seterr(invalid='ignore')


torch.backends.cudnn.benchmark = True # Default


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        normalized = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return normalized
    
def get_embedder(multires = 10, i=0):
    if i == -1:
        return nn.Identity(), 1

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

embeder = get_embedder()[0]    
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# GELU -> Gaussian Error Linear Units
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class RemixerBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        causal = False,
        bias = False
    ):
        super().__init__()
        self.causal = causal
        self.proj_in = nn.Linear(dim, 2 * dim, bias = bias)
        self.mixer = nn.Parameter(torch.randn(seq_len, seq_len))
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.proj_out = nn.Linear(dim, dim, bias = bias)

    def forward(self, x):
        mixer, causal, device = self.mixer, self.causal, x.device
        x, gate = self.proj_in(x).chunk(2, dim = -1)
        x = F.gelu(gate) * x

        if self.causal:
            seq = x.shape[1]
            mask_value = -torch.finfo(x.dtype).max
            mask = torch.ones((seq, seq), device = device, dtype=torch.bool).triu(1)
            mixer = mixer[:seq, :seq]
            mixer = mixer.masked_fill(mask, mask_value)

        mixer = mixer.softmax(dim = -1)
        mixed = einsum('b n d, m n -> b m d', x, mixer)

        alpha = self.alpha.sigmoid()
        out = (x * mixed) * alpha + (x - mixed) * (1 - alpha)

        return self.proj_out(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        # print(f'Attention:: {dim} - {heads} - {dim_head} - {dropout}')

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_embedding = PositionalEncoding(dim,0.1,128)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x += self.pos_embedding(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()

        # print('\n')
        # print(f'Transformers:: {dim} - {depth} - {heads} - {dim_head} - {mlp_dim}')

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                #PreNorm(dim, RemixerBlock(dim,17))
            ]))

    def forward(self, x, swap = False):
        if swap: # for the self.transformer(x,swap = True)
            b, t, n , c = x.size() 
        for idx, (attn, ff) in enumerate(self.layers):
            if swap: # for the self.transformer(x,swap = True)
                if idx % 2 == 0:
                    #* attention along with all timesteps(frames) for each point(landmark)
                    x = rearrange(x, "b t n c -> (b n) t c")
                else:
                    #* attention to all points(landmarks) in each timestep(frame)
                    x = rearrange(x, "b t n c -> (b t) n c")
            x = attn(x) + x  # skip connections
            x = ff(x) + x    # skip connections
            
            # Now return the input x to its original formation
            if swap: # for the self.transformer(x,swap = True)
                if idx % 2 == 0:
                    x = rearrange(x, "(b n) t c -> b t n c", b = b)
                else:
                    x = rearrange(x, "(b t) n c -> b t n c", b = b)
                
        return x


class TemporalModel(nn.Module):
    
    def __init__(self):
        super(TemporalModel,self).__init__()
                
        self.encoder  =  CurveNet() # curve aggregation, needed for Point Clouds Shape Analysis. 
        self.downsample = nn.Sequential(
                            nn.Conv1d(478, 32, kernel_size=1, bias=False),
                            nn.BatchNorm1d(32),
                            # nn.Dropout(p=0.25), #* NEW
                            #nn.ReLU(inplace=True),
                            #nn.Conv1d(128, 32, kernel_size=1, bias=False),
                            #nn.BatchNorm1d(32),
                            )
        
        self.transformer = Transformer(256, 6, 4, 256//4, 256 * 2, 0.1)
        self.time = Transformer(256, 3, 4, 256//4, 256 * 2, 0.1)
        self.dropout = nn.Dropout(0.1)
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256,216)
        )
        
    def forward(self,x):
        b,t,n,c = x.size()
    
        x = rearrange(x, "b t n c -> (b t) c n")
        x = rearrange(self.dropout(self.encoder(x)), "b c n -> b n c") 
        x = self.downsample(x).view(b,t,32,-1) #b t 32 c
        x = self.transformer(x,swap = True).view(b,t,-1,256).mean(2)
        x1 = self.time(x).mean(1)
        x2 = self.mlp_head1(x1)
        x3 = self.mlp_head2(x1)
        return x2, x3
        

min_xyz = np.array([0.06372425, 0.05751023, -0.08976112]).reshape(1,1,3)
max_xyz = np.array([0.63246971, 1.01475966, 0.14436169]).reshape(1,1,3)



class DataGenerator(torch.utils.data.Dataset):
    
    def __init__(self,data,label_path,test = False):
        self.data = data
        self.label_path = label_path
        self.__dataset_information()
        self.test = test

    def __dataset_information(self):
        self.numbers_of_data = 0

        with open(self.label_path) as f:
            labels = json.load(f)

        self.index_name_dic = dict()
        for index,(k,v) in enumerate(labels.items()):
            self.index_name_dic[index] = [k,v]

        self.numbers_of_data = index + 1

        output(f"Load {self.numbers_of_data} videos")
        print(f"Load {self.numbers_of_data} videos")

    def __len__(self):
        
        return self.numbers_of_data

    def __getitem__(self,idx):
        ids = self.index_name_dic[idx]
        size = 5 if self.test else 1 
        x, y = self.__data_generation(ids, size)
        
        return x,y
             
    def __data_generation(self,ids, size):
        name,label = ids
        y = torch.FloatTensor([label])
        
        clips = []
        for _ in range(size):
          x = np.load(os.path.join(self.data,f"{name}.mp4.npy"))
          start = x.shape[0] - 16
          if start > 0:
            start = np.random.randint(0,start) 
            x = x[start:][:16]
          else:
            start = np.random.randint(0,1)
            x = np.array(x)[start:]
        
          x = (x - min_xyz) / (max_xyz - min_xyz)
          pad_x = np.zeros((16,478,3))
          if x.shape[0] == 16:
            pad_x = x
          else:
            pad_x[:x.shape[0]] = x
          pad_x = torch.FloatTensor(pad_x) 
          clips.append(pad_x)
        clips = torch.stack(clips,0)
        return clips,y
    
perf = ""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import argrelextrema

l1=33 #right eye right
l2=159  #right eye center
l3=133 #right eye left
l4=362 #left eye right
l5=386 #left eye center
l6=263 #left eye left
l7=50 #right cheek
l9=1   #nose tip
l8=280 #left cheek
l10=62 #lip corner right
l11=308#lip corner left

# Compute distance between two points
def compute_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Compute rotation matrix
def compute_rotation_matrix(theta_x, theta_y, theta_z):
    r = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True)
    return r.as_matrix()

# Function to calculate onset, apex and offset of smile
def smile_phases(Dlip):
    onset = np.min(argrelextrema(Dlip, np.less)[0]) # assuming onset is minima
    offset = np.max(argrelextrema(Dlip, np.greater)[0]) # assuming offset is maxima
    apex = np.argmax(Dlip) # assuming apex is max value of Dlip
    return onset, apex, offset

# Calculate first derivative (speed) and second derivative (acceleration) of a signal
def compute_derivatives(signal):
    speed = np.diff(signal) # first derivative
    acceleration = np.diff(speed) # second derivative
    return speed, acceleration

def find_sequences(Dlips):
    longest_inc_seq, longest_dec_seq = [], []
    inc_seq, dec_seq = [], []

    for i in range(1, len(Dlips)):
        if Dlips[i] > Dlips[i-1]:
            if not inc_seq:  # Start of a new increasing sequence
                inc_seq.extend([i-1, i])
            else:
                inc_seq.append(i)
            if dec_seq:  # End of a decreasing sequence
                if len(dec_seq) > len(longest_dec_seq):
                    longest_dec_seq = dec_seq[:]
                dec_seq = []  # Clear the decreasing sequence
        elif Dlips[i] < Dlips[i-1]:
            if not dec_seq:  # Start of a new decreasing sequence
                dec_seq.extend([i-1, i])
            else:
                dec_seq.append(i)
            if inc_seq:  # End of an increasing sequence
                if len(inc_seq) > len(longest_inc_seq):
                    longest_inc_seq = inc_seq[:]
                inc_seq = []  # Clear the increasing sequence

    # Check the last sequence
    if inc_seq and len(inc_seq) > len(longest_inc_seq):
        longest_inc_seq = inc_seq[:]
    if dec_seq and len(dec_seq) > len(longest_dec_seq):
        longest_dec_seq = dec_seq[:]

    return longest_inc_seq, longest_dec_seq

# define rotation matrices
def Rx(theta):
    return np.array([[1, 0, 0], 
                     [0, np.cos(theta), np.sin(theta)], 
                     [0, -np.sin(theta), np.cos(theta)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, -np.sin(theta)], 
                     [0, 1, 0], 
                     [np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], 
                     [np.sin(theta), np.cos(theta), 0], 
                     [0, 0, 1]])

def k(li, lj):
    return -1 if lj[2] < li[2] else 1

def extract_features(data):

    landmarks = data

    # landmarks is your 4D numpy array [1240, num_frames, 478, 3]
    num_samples = len(data)

    # Compute the 24 features for each video sample
    features = np.zeros((num_samples, 3, 24))
    features_eye = np.zeros((num_samples, 3, 24))
    features_cheek = np.zeros((num_samples, 3, 24))


    for sample in range(num_samples):
        num_frames = landmarks[sample].shape[0]

        l1_one = None
        l2_one = None
        l3_one = None
        l4_one = None
        l5_one = None
        l6_one = None
        l7_one = None
        l8_one = None
        l9_one = None
        l10_one = None
        l11_one = None

        Dlips = np.zeros(num_frames)
        Deyes = np.zeros(num_frames)
        Dcheeks = np.zeros(num_frames)

        for frame in range(num_frames):
            # Extract the points
            point_c1 = (landmarks[sample][frame][l1] + landmarks[sample][frame][l3]) / 2 
            point_c2 = (landmarks[sample][frame][l4] + landmarks[sample][frame][l6]) / 2 

            point_l1 = landmarks[sample][frame][l1]
            point_l2 = landmarks[sample][frame][l2]
            point_l3 = landmarks[sample][frame][l3]
            point_l4 = landmarks[sample][frame][l4]
            point_l5 = landmarks[sample][frame][l5]
            point_l6 = landmarks[sample][frame][l6]
            point_l7 = landmarks[sample][frame][l7]
            point_l8 = landmarks[sample][frame][l8]
            point_l9 = landmarks[sample][frame][l9]
            point_l10 = landmarks[sample][frame][l10]
            point_l11 = landmarks[sample][frame][l11]

            # print(point_c1)
            # print(point_c2)
            # print(point_l9)


            # Calculate plane
            NP = np.cross(point_c2 - point_l9, point_c1 - point_l9)

            # print(NP)

            if NP[1] < 0:
                NP = -NP  # If the y-component of the normal is negative, flip the normal vector

            # Calculate yaw, roll, and pitch (Theta_x, Theta_y, Theta_z)
            yaw = np.arccos(NP[1] / np.linalg.norm(NP)) # yaw is rotation around y axis
            pitch = np.arccos(NP[0] / np.linalg.norm(NP)) # pitch is rotation around x axis
            roll = np.arccos(NP[2] / np.linalg.norm(NP)) # roll is rotation around z axis

            # print(yaw)
            # print(pitch)
            # print(roll)



            # Compute rotation matrix
            rotation_matrix = compute_rotation_matrix(-pitch, -yaw, -roll)

            # Normalize the points
            point_l9_prime = (point_l9 - (point_c1 + point_c2) / 2)
            point_l9_prime = np.matmul(point_l9_prime, rotation_matrix)
            point_l9_prime = 100 * point_l9_prime / compute_distance(point_c1, point_c2)

            point_l1_prime = (point_l1 - (point_c1 + point_c2) / 2)
            point_l1_prime = np.matmul(point_l1_prime, rotation_matrix)
            point_l1_prime = 100 * point_l1_prime / compute_distance(point_c1, point_c2)

            point_l2_prime = (point_l2 - (point_c1 + point_c2) / 2)
            point_l2_prime = np.matmul(point_l2_prime, rotation_matrix)
            point_l2_prime = 100 * point_l2_prime / compute_distance(point_c1, point_c2)

            point_l3_prime = (point_l3 - (point_c1 + point_c2) / 2)
            point_l3_prime = np.matmul(point_l3_prime, rotation_matrix)
            point_l3_prime = 100 * point_l3_prime / compute_distance(point_c1, point_c2)

            point_l4_prime = (point_l4 - (point_c1 + point_c2) / 2)
            point_l4_prime = np.matmul(point_l4_prime, rotation_matrix)
            point_l4_prime = 100 * point_l4_prime / compute_distance(point_c1, point_c2)

            point_l5_prime = (point_l5 - (point_c1 + point_c2) / 2)
            point_l5_prime = np.matmul(point_l5_prime, rotation_matrix)
            point_l5_prime = 100 * point_l5_prime / compute_distance(point_c1, point_c2)

            point_l6_prime = (point_l6 - (point_c1 + point_c2) / 2)
            point_l6_prime = np.matmul(point_l6_prime, rotation_matrix)
            point_l6_prime = 100 * point_l6_prime / compute_distance(point_c1, point_c2)

            point_l7_prime = (point_l7 - (point_c1 + point_c2) / 2)
            point_l7_prime = np.matmul(point_l7_prime, rotation_matrix)
            point_l7_prime = 100 * point_l7_prime / compute_distance(point_c1, point_c2)

            point_l8_prime = (point_l8 - (point_c1 + point_c2) / 2)
            point_l8_prime = np.matmul(point_l8_prime, rotation_matrix)
            point_l8_prime = 100 * point_l8_prime / compute_distance(point_c1, point_c2)

            point_l10_prime = (point_l10 - (point_c1 + point_c2) / 2)
            point_l10_prime = np.matmul(point_l10_prime, rotation_matrix)
            point_l10_prime = 100 * point_l10_prime / compute_distance(point_c1, point_c2)

            point_l11_prime = (point_l11 - (point_c1 + point_c2) / 2)
            point_l11_prime = np.matmul(point_l11_prime, rotation_matrix)
            point_l11_prime = 100 * point_l11_prime / compute_distance(point_c1, point_c2)

            if frame == 0:
                l1_one = point_l1_prime
                l2_one = point_l2_prime
                l3_one = point_l3_prime
                l4_one = point_l4_prime
                l5_one = point_l5_prime
                l6_one = point_l6_prime
                l7_one = point_l7_prime
                l8_one = point_l8_prime
                l9_one = point_l9_prime
                l10_one = point_l10_prime
                l11_one = point_l11_prime


            # Extract features from mouth
            Dlip = (compute_distance((l10_one + l11_one) / 2, point_l10_prime) + compute_distance((l10_one + l11_one) / 2, point_l11_prime)) / (2 * compute_distance(l10_one, l11_one))

            Deye = (k((point_l1_prime + point_l3_prime) / 2, point_l2_prime) * compute_distance((point_l1_prime + point_l3_prime) / 2, point_l2_prime) + k((point_l4_prime + point_l6_prime) / 2, point_l5_prime) * compute_distance((point_l4_prime + point_l6_prime) / 2, point_l5_prime)) / (2 * compute_distance(point_l1_prime, point_l3_prime))

            Dcheek = (compute_distance((l7_one + l8_one) / 2, point_l7_prime) + compute_distance((l7_one + l8_one) / 2, point_l8_prime)) / (2 * compute_distance(l7_one, l8_one))

            Dlips[frame] = Dlip

            Deyes[frame] = Deye

            Dcheeks[frame] = Dcheek


        # After Dlips calculation
        longest_inc_seq, longest_dec_seq = find_sequences(Dlips)

        # Getting the actual sequences
        longest_inc_Dlips = Dlips[longest_inc_seq]
        longest_dec_Dlips = Dlips[longest_dec_seq]

        # print(Dlips)
        # print("Longest Increasing:")
        # print(longest_inc_Dlips)
        # print("Longest Decreasing:")
        # print(longest_dec_Dlips)

        frame_rate = 50

        features[sample, 0, 0] = len(longest_inc_Dlips) / frame_rate
        features[sample, 0, 1] = len(longest_dec_Dlips) / frame_rate
        features[sample, 0, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features[sample, 0, 3] = len(longest_inc_Dlips) / len(Dlips)
            
            features[sample, 0, 4] = len(longest_dec_Dlips) / len(Dlips)
        
            features[sample, 0, 6] = np.sum(Dlips) / len(Dlips)
            features[sample, 0, 5] = np.max(Dlips)
            features[sample, 0, 9] = np.std(Dlips)
            
        features[sample, 0, 7] = np.sum(longest_inc_Dlips) / len(longest_inc_Dlips)
        features[sample, 0, 8] = np.sum(np.abs(longest_dec_Dlips)) / len(longest_dec_Dlips)
        
        features[sample, 0, 10] = np.sum(longest_inc_Dlips)
        features[sample, 0, 11] = np.sum(np.abs(longest_dec_Dlips))
        features[sample, 0, 12] = features[sample, 0, 10] - features[sample, 0, 11]
        features[sample, 0, 13] = features[sample, 0, 10] / (features[sample, 0, 10] + features[sample, 0, 11])
        features[sample, 0, 14] = features[sample, 0, 11] / (features[sample, 0, 10] + features[sample, 0, 11])
        
        if len(longest_inc_Dlips) > 1:
            
            features[sample, 0, 15] = np.max(np.diff(longest_inc_Dlips))
            features[sample, 0, 17] = np.mean(np.diff(longest_inc_Dlips))
            
        if len(longest_dec_Dlips) > 1:
            features[sample, 0, 16] = np.max(np.diff(longest_dec_Dlips))

            features[sample, 0, 18] = np.mean(np.diff(longest_dec_Dlips))

        if len(longest_inc_Dlips) <= 2:
            features[sample, 0, 19] = 0
        else:
            features[sample, 0, 19] = np.max(np.diff(np.diff(longest_inc_Dlips)))
            features[sample, 0, 21] = np.mean(np.diff(np.diff(longest_inc_Dlips)))

        if len(longest_dec_Dlips) <= 2:
            features[sample, 0, 20] = 0
        else:
            features[sample, 0, 20] = np.max(np.diff(np.diff(longest_dec_Dlips)))

        
            features[sample, 0, 22] = np.mean(np.diff(np.diff(longest_dec_Dlips)))
            
        if len(Dlips) > 0:
            features[sample, 0, 23] = features[sample, 0, 12] * frame_rate / len(Dlips)

        Dlips = longest_inc_Dlips
        longest_inc_Dlips = longest_inc_Dlips
        longest_dec_Dlipss = np.zeros(1)

        features[sample, 1, 0] = len(longest_inc_Dlips) / frame_rate
        features[sample, 1, 1] = len(longest_dec_Dlipss) / frame_rate
        features[sample, 1, 2] = len(Dlips) / frame_rate
        if len(Dlips) > 0:
            features[sample, 1, 3] = len(longest_inc_Dlips) / len(Dlips)
            features[sample, 1, 4] = len(longest_dec_Dlipss) / len(Dlips)
            features[sample, 1, 6] = np.sum(Dlips) / len(Dlips)
            features[sample, 1, 5] = np.max(Dlips)
            features[sample, 1, 9] = np.std(Dlips)
        
        features[sample, 1, 7] = np.sum(longest_inc_Dlips) / len(longest_inc_Dlips)
        features[sample, 1, 8] = np.sum(np.abs(longest_dec_Dlipss)) / len(longest_dec_Dlipss)
        
        features[sample, 1, 10] = np.sum(longest_inc_Dlips)
        features[sample, 1, 11] = np.sum(np.abs(longest_dec_Dlipss))
        features[sample, 1, 12] = features[sample, 1, 10] - features[sample, 1, 11]
        features[sample, 1, 13] = features[sample, 1, 10] / (features[sample, 1, 10] + features[sample, 1, 11])
        features[sample, 1, 14] = features[sample, 1, 11] / (features[sample, 1, 10] + features[sample, 1, 11])
        
        

        if len(longest_dec_Dlipss) <= 1:
            features[sample, 1, 16] = 0
        else:
            features[sample, 1, 16] = np.max(np.diff(longest_dec_Dlipss))
            
        if len(longest_inc_Dlips) <= 1:
            features[sample, 1, 17] = 0
            
        else:
            features[sample, 1, 17] = np.mean(np.diff(longest_inc_Dlips))
            features[sample, 1, 15] = np.max(np.diff(longest_inc_Dlips))
            
        if len(longest_dec_Dlipss) <= 1:
            features[sample, 1, 18] = 0
            
        else:
            features[sample, 1, 18] = np.mean(np.diff(longest_dec_Dlipss))

        if len(longest_inc_Dlips) <= 2:
            features[sample, 1, 19] = 0
            features[sample, 1, 21] = 0
        else:
            features[sample, 1, 19] = np.max(np.diff(np.diff(longest_inc_Dlips)))
            features[sample, 1, 21] = np.mean(np.diff(np.diff(longest_inc_Dlips)))

        if len(longest_dec_Dlipss) <= 2:
            features[sample, 1, 20] = 0
            features[sample, 1, 22] = 0
        else:
            features[sample, 1, 20] = np.max(np.diff(np.diff(longest_dec_Dlipss)))
            features[sample, 1, 22] = np.mean(np.diff(np.diff(longest_dec_Dlipss)))
            
        if len(Dlips) > 0:
            features[sample, 1, 23] = features[sample, 1, 12] * frame_rate / len(Dlips)

        Dlips = longest_dec_Dlips
        longest_inc_Dlipss = np.zeros(1)
        longest_dec_Dlips = longest_dec_Dlips

        features[sample, 2, 0] = len(longest_inc_Dlipss) / frame_rate
        features[sample, 2, 1] = len(longest_dec_Dlips) / frame_rate
        features[sample, 2, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features[sample, 2, 3] = len(longest_inc_Dlipss) / len(Dlips)
            features[sample, 2, 4] = len(longest_dec_Dlips) / len(Dlips)
            features[sample, 2, 6] = np.sum(Dlips) / len(Dlips)
            features[sample, 2, 5] = np.max(Dlips)
            features[sample, 2, 9] = np.std(Dlips)
        
        features[sample, 2, 7] = np.sum(longest_inc_Dlipss) / len(longest_inc_Dlipss)
        features[sample, 2, 8] = np.sum(np.abs(longest_dec_Dlips)) / len(longest_dec_Dlips)
        
        features[sample, 2, 10] = np.sum(longest_inc_Dlipss)
        features[sample, 2, 11] = np.sum(np.abs(longest_dec_Dlips))
        features[sample, 2, 12] = features[sample, 2, 10] - features[sample, 2, 11]
        features[sample, 2, 13] = features[sample, 2, 10] / (features[sample, 2, 10] + features[sample, 2, 11])
        features[sample, 2, 14] = features[sample, 2, 11] / (features[sample, 2, 10] + features[sample, 2, 11])

        if len(longest_inc_Dlipss) <= 1:
            features[sample, 2, 15] = 0

        else:
            features[sample, 2, 15] = np.max(np.diff(longest_inc_Dlipss))
            features[sample, 2, 17] = np.mean(np.diff(longest_inc_Dlipss))

        if len(longest_dec_Dlips) <= 1:
            features[sample, 2, 16] = 0

        else:
            features[sample, 2, 16] = np.max(np.diff(longest_dec_Dlips))
        
            features[sample, 2, 18] = np.mean(np.diff(longest_dec_Dlips))

        if len(longest_inc_Dlipss) <= 2:
            features[sample, 2, 19] = 0
        else:
            features[sample, 2, 19] = np.max(np.diff(np.diff(longest_inc_Dlipss)))
            features[sample, 2, 21] = np.mean(np.diff(np.diff(longest_inc_Dlipss)))

        if len(longest_dec_Dlips) <= 2:
            features[sample, 2, 20] = 0
        else:
            features[sample, 2, 20] = np.max(np.diff(np.diff(longest_dec_Dlips)))

        
            features[sample, 2, 22] = np.mean(np.diff(np.diff(longest_dec_Dlips)))
            
        if len(Dlips) > 0:
            features[sample, 2, 23] = features[sample, 2, 12] * frame_rate / len(Dlips)

        features = np.nan_to_num(features)

        Dlips = Deyes
        # After Dlips calculation
        longest_inc_seq, longest_dec_seq = find_sequences(Dlips)

        # Getting the actual sequences
        longest_inc_Dlips = Dlips[longest_inc_seq]
        longest_dec_Dlips = Dlips[longest_dec_seq]

        # print(Dlips)
        # print("Longest Increasing:")
        # print(longest_inc_Dlips)
        # print("Longest Decreasing:")
        # print(longest_dec_Dlips)

        frame_rate = 50

        features_eye[sample, 0, 0] = len(longest_inc_Dlips) / frame_rate
        features_eye[sample, 0, 1] = len(longest_dec_Dlips) / frame_rate
        features_eye[sample, 0, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features_eye[sample, 0, 3] = len(longest_inc_Dlips) / len(Dlips)
            features_eye[sample, 0, 4] = len(longest_dec_Dlips) / len(Dlips)
            features_eye[sample, 0, 6] = np.sum(Dlips) / len(Dlips)
            features_eye[sample, 0, 5] = np.max(Dlips)
            features_eye[sample, 0, 9] = np.std(Dlips)
        
        features_eye[sample, 0, 7] = np.sum(longest_inc_Dlips) / len(longest_inc_Dlips)
        features_eye[sample, 0, 8] = np.sum(np.abs(longest_dec_Dlips)) / len(longest_dec_Dlips)
        
        features_eye[sample, 0, 10] = np.sum(longest_inc_Dlips)
        features_eye[sample, 0, 11] = np.sum(np.abs(longest_dec_Dlips))
        features_eye[sample, 0, 12] = features_eye[sample, 0, 10] - features_eye[sample, 0, 11]
        features_eye[sample, 0, 13] = features_eye[sample, 0, 10] / (features_eye[sample, 0, 10] + features_eye[sample, 0, 11])
        features_eye[sample, 0, 14] = features_eye[sample, 0, 11] / (features_eye[sample, 0, 10] + features_eye[sample, 0, 11])
        
        if len(longest_inc_Dlips) > 1:
            
            features_eye[sample, 0, 15] = np.max(np.diff(longest_inc_Dlips))
            features_eye[sample, 0, 17] = np.mean(np.diff(longest_inc_Dlips))
            
        if len(longest_dec_Dlips) > 1:
            features_eye[sample, 0, 16] = np.max(np.diff(longest_dec_Dlips))

            features_eye[sample, 0, 18] = np.mean(np.diff(longest_dec_Dlips))

        if len(longest_inc_Dlips) <= 2:
            features_eye[sample, 0, 19] = 0
        else:
            features_eye[sample, 0, 19] = np.max(np.diff(np.diff(longest_inc_Dlips)))
            features_eye[sample, 0, 21] = np.mean(np.diff(np.diff(longest_inc_Dlips)))

        if len(longest_dec_Dlips) <= 2:
            features_eye[sample, 0, 20] = 0
        else:
            features_eye[sample, 0, 20] = np.max(np.diff(np.diff(longest_dec_Dlips)))

        
            features_eye[sample, 0, 22] = np.mean(np.diff(np.diff(longest_dec_Dlips)))
            
        if len(Dlips) > 0:
            features_eye[sample, 0, 23] = features_eye[sample, 0, 12] * frame_rate / len(Dlips)

        Dlips = longest_inc_Dlips
        longest_inc_Dlips = longest_inc_Dlips
        longest_dec_Dlipss = np.zeros(1)

        features_eye[sample, 1, 0] = len(longest_inc_Dlips) / frame_rate
        features_eye[sample, 1, 1] = len(longest_dec_Dlipss) / frame_rate
        features_eye[sample, 1, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features_eye[sample, 1, 3] = len(longest_inc_Dlips) / len(Dlips)
            features_eye[sample, 1, 4] = len(longest_dec_Dlipss) / len(Dlips)
            features_eye[sample, 1, 6] = np.sum(Dlips) / len(Dlips)
            features_eye[sample, 1, 5] = np.max(Dlips)
            features_eye[sample, 1, 9] = np.std(Dlips)
        
        features_eye[sample, 1, 7] = np.sum(longest_inc_Dlips) / len(longest_inc_Dlips)
        features_eye[sample, 1, 8] = np.sum(np.abs(longest_dec_Dlipss)) / len(longest_dec_Dlipss)
        
        features_eye[sample, 1, 10] = np.sum(longest_inc_Dlips)
        features_eye[sample, 1, 11] = np.sum(np.abs(longest_dec_Dlipss))
        features_eye[sample, 1, 12] = features_eye[sample, 1, 10] - features_eye[sample, 1, 11]
        features_eye[sample, 1, 13] = features_eye[sample, 1, 10] / (features_eye[sample, 1, 10] + features_eye[sample, 1, 11])
        features_eye[sample, 1, 14] = features_eye[sample, 1, 11] / (features_eye[sample, 1, 10] + features_eye[sample, 1, 11])
        

        if len(longest_dec_Dlipss) <= 1:
            features_eye[sample, 1, 16] = 0
        else:
            features_eye[sample, 1, 16] = np.max(np.diff(longest_dec_Dlipss))
            features_eye[sample, 1, 18] = np.mean(np.diff(longest_dec_Dlipss))
            
        if len(longest_inc_Dlipss) > 1:
            features_eye[sample, 1, 17] = np.mean(np.diff(longest_inc_Dlips))
            features_eye[sample, 1, 15] = np.max(np.diff(longest_inc_Dlips))
            
        

        if len(longest_inc_Dlips) <= 2:
            features_eye[sample, 1, 19] = 0
        else:
            features_eye[sample, 1, 19] = np.max(np.diff(np.diff(longest_inc_Dlips)))
            features_eye[sample, 1, 21] = np.mean(np.diff(np.diff(longest_inc_Dlips)))

        if len(longest_dec_Dlipss) <= 2:
            features_eye[sample, 1, 20] = 0
        else:
            features_eye[sample, 1, 20] = np.max(np.diff(np.diff(longest_dec_Dlipss)))

        
            features_eye[sample, 1, 22] = np.mean(np.diff(np.diff(longest_dec_Dlipss)))
        if len(Dlips) > 0:
            features_eye[sample, 1, 23] = features_eye[sample, 1, 12] * frame_rate / len(Dlips)

        Dlips = longest_dec_Dlips
        longest_inc_Dlipss = np.zeros(1)
        longest_dec_Dlips = longest_dec_Dlips

        features_eye[sample, 2, 0] = len(longest_inc_Dlipss) / frame_rate
        features_eye[sample, 2, 1] = len(longest_dec_Dlips) / frame_rate
        features_eye[sample, 2, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features_eye[sample, 2, 3] = len(longest_inc_Dlipss) / len(Dlips)
            features_eye[sample, 2, 4] = len(longest_dec_Dlips) / len(Dlips)
            features_eye[sample, 2, 6] = np.sum(Dlips) / len(Dlips)
            features_eye[sample, 2, 5] = np.max(Dlips)
            features_eye[sample, 2, 9] = np.std(Dlips)
        
        features_eye[sample, 2, 7] = np.sum(longest_inc_Dlipss) / len(longest_inc_Dlipss)
        features_eye[sample, 2, 8] = np.sum(np.abs(longest_dec_Dlips)) / len(longest_dec_Dlips)
        
        features_eye[sample, 2, 10] = np.sum(longest_inc_Dlipss)
        features_eye[sample, 2, 11] = np.sum(np.abs(longest_dec_Dlips))
        features_eye[sample, 2, 12] = features_eye[sample, 2, 10] - features_eye[sample, 2, 11]
        features_eye[sample, 2, 13] = features_eye[sample, 2, 10] / (features_eye[sample, 2, 10] + features_eye[sample, 2, 11])
        features_eye[sample, 2, 14] = features_eye[sample, 2, 11] / (features_eye[sample, 2, 10] + features_eye[sample, 2, 11])

        if len(longest_inc_Dlipss) <= 1:
            features_eye[sample, 2, 15] = 0

        else:
            features_eye[sample, 2, 15] = np.max(np.diff(longest_inc_Dlipss))
            features_eye[sample, 2, 17] = np.mean(np.diff(longest_inc_Dlipss))

        if len(longest_dec_Dlips) <= 1:
            features_eye[sample, 2, 16] = 0

        else:
            features_eye[sample, 2, 16] = np.max(np.diff(longest_dec_Dlips))
        
            features_eye[sample, 2, 18] = np.mean(np.diff(longest_dec_Dlips))

        if len(longest_inc_Dlipss) <= 2:
            features_eye[sample, 2, 19] = 0
        else:
            features_eye[sample, 2, 19] = np.max(np.diff(np.diff(longest_inc_Dlipss)))
            features_eye[sample, 2, 21] = np.mean(np.diff(np.diff(longest_inc_Dlipss)))

        if len(longest_dec_Dlips) <= 2:
            features_eye[sample, 2, 20] = 0
        else:
            features_eye[sample, 2, 20] = np.max(np.diff(np.diff(longest_dec_Dlips)))

        
            features_eye[sample, 2, 22] = np.mean(np.diff(np.diff(longest_dec_Dlips)))
        if len(Dlips) > 0:
            features_eye[sample, 2, 23] = features_eye[sample, 2, 12] * frame_rate / len(Dlips)

        features_eye = np.nan_to_num(features_eye)

        Dlips = Dcheeks
        longest_inc_seq, longest_dec_seq = find_sequences(Dlips)

        # Getting the actual sequences
        longest_inc_Dlips = Dlips[longest_inc_seq]
        longest_dec_Dlips = Dlips[longest_dec_seq]

        # print(Dlips)
        # print("Longest Increasing:")
        # print(longest_inc_Dlips)
        # print("Longest Decreasing:")
        # print(longest_dec_Dlips)

        frame_rate = 50

        features_cheek[sample, 0, 0] = len(longest_inc_Dlips) / frame_rate
        features_cheek[sample, 0, 1] = len(longest_dec_Dlips) / frame_rate
        features_cheek[sample, 0, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features_cheek[sample, 0, 3] = len(longest_inc_Dlips) / len(Dlips)
            features_cheek[sample, 0, 4] = len(longest_dec_Dlips) / len(Dlips)
            features_cheek[sample, 0, 6] = np.sum(Dlips) / len(Dlips)
            features_cheek[sample, 0, 5] = np.max(Dlips)
            features_cheek[sample, 0, 9] = np.std(Dlips)
        
        features_cheek[sample, 0, 7] = np.sum(longest_inc_Dlips) / len(longest_inc_Dlips)
        features_cheek[sample, 0, 8] = np.sum(np.abs(longest_dec_Dlips)) / len(longest_dec_Dlips)
        
        features_cheek[sample, 0, 10] = np.sum(longest_inc_Dlips)
        features_cheek[sample, 0, 11] = np.sum(np.abs(longest_dec_Dlips))
        features_cheek[sample, 0, 12] = features_cheek[sample, 0, 10] - features_cheek[sample, 0, 11]
        features_cheek[sample, 0, 13] = features_cheek[sample, 0, 10] / (features_cheek[sample, 0, 10] + features_cheek[sample, 0, 11])
        features_cheek[sample, 0, 14] = features_cheek[sample, 0, 11] / (features_cheek[sample, 0, 10] + features_cheek[sample, 0, 11])
        
        if len(longest_inc_Dlips) > 1:
            
            features_cheek[sample, 0, 15] = np.max(np.diff(longest_inc_Dlips))
        
            features_cheek[sample, 0, 17] = np.mean(np.diff(longest_inc_Dlips))
            
        if len(longest_dec_Dlips) > 1:    
            features_cheek[sample, 0, 18] = np.mean(np.diff(longest_dec_Dlips))
            features_cheek[sample, 0, 16] = np.max(np.diff(longest_dec_Dlips))

        if len(longest_inc_Dlips) <= 2:
            features_cheek[sample, 0, 19] = 0
        else:
            features_cheek[sample, 0, 19] = np.max(np.diff(np.diff(longest_inc_Dlips)))
            features_cheek[sample, 0, 21] = np.mean(np.diff(np.diff(longest_inc_Dlips)))

        if len(longest_dec_Dlips) <= 2:
            features_cheek[sample, 0, 20] = 0
        else:
            features_cheek[sample, 0, 20] = np.max(np.diff(np.diff(longest_dec_Dlips)))

        
            features_cheek[sample, 0, 22] = np.mean(np.diff(np.diff(longest_dec_Dlips)))
        
        if len(Dlips) > 0:
            features_cheek[sample, 0, 23] = features_cheek[sample, 0, 12] * frame_rate / len(Dlips)

        Dlips = longest_inc_Dlips
        longest_inc_Dlips = longest_inc_Dlips
        longest_dec_Dlipss = np.zeros(1)

        features_cheek[sample, 1, 0] = len(longest_inc_Dlips) / frame_rate
        features_cheek[sample, 1, 1] = len(longest_dec_Dlipss) / frame_rate
        features_cheek[sample, 1, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features_cheek[sample, 1, 3] = len(longest_inc_Dlips) / len(Dlips)
            features_cheek[sample, 1, 4] = len(longest_dec_Dlipss) / len(Dlips)
            features_cheek[sample, 1, 6] = np.sum(Dlips) / len(Dlips)
            features_cheek[sample, 1, 5] = np.max(Dlips)
            features_cheek[sample, 1, 9] = np.std(Dlips)
        
        features_cheek[sample, 1, 7] = np.sum(longest_inc_Dlips) / len(longest_inc_Dlips)
        features_cheek[sample, 1, 8] = np.sum(np.abs(longest_dec_Dlipss)) / len(longest_dec_Dlipss)
        
        features_cheek[sample, 1, 10] = np.sum(longest_inc_Dlips)
        features_cheek[sample, 1, 11] = np.sum(np.abs(longest_dec_Dlipss))
        features_cheek[sample, 1, 12] = features_cheek[sample, 1, 10] - features_cheek[sample, 1, 11]
        features_cheek[sample, 1, 13] = features_cheek[sample, 1, 10] / (features_cheek[sample, 1, 10] + features_cheek[sample, 1, 11])
        features_cheek[sample, 1, 14] = features_cheek[sample, 1, 11] / (features_cheek[sample, 1, 10] + features_cheek[sample, 1, 11])
        
        if len(longest_inc_Dlipss) > 1:
            features_cheek[sample, 1, 15] = np.max(np.diff(longest_inc_Dlips))
            features_cheek[sample, 1, 17] = np.mean(np.diff(longest_inc_Dlips))

        if len(longest_dec_Dlipss) <= 1:
            features_cheek[sample, 1, 16] = 0
        else:
            features_cheek[sample, 1, 16] = np.max(np.diff(longest_dec_Dlipss))
        
            features_cheek[sample, 1, 18] = np.mean(np.diff(longest_dec_Dlipss))

        if len(longest_inc_Dlips) <= 2:
            features_cheek[sample, 1, 19] = 0
        else:
            features_cheek[sample, 1, 19] = np.max(np.diff(np.diff(longest_inc_Dlips)))
            features_cheek[sample, 1, 21] = np.mean(np.diff(np.diff(longest_inc_Dlips)))

        if len(longest_dec_Dlipss) <= 2:
            features_cheek[sample, 1, 20] = 0
        else:
            features_cheek[sample, 1, 20] = np.max(np.diff(np.diff(longest_dec_Dlipss)))

        
            features_cheek[sample, 1, 22] = np.mean(np.diff(np.diff(longest_dec_Dlipss)))
        
        if len(Dlips) > 0:
            features_cheek[sample, 1, 23] = features_cheek[sample, 1, 12] * frame_rate / len(Dlips)

        Dlips = longest_dec_Dlips
        longest_inc_Dlipss = np.zeros(1)
        longest_dec_Dlips = longest_dec_Dlips

        features_cheek[sample, 2, 0] = len(longest_inc_Dlipss) / frame_rate
        features_cheek[sample, 2, 1] = len(longest_dec_Dlips) / frame_rate
        features_cheek[sample, 2, 2] = len(Dlips) / frame_rate
        
        if len(Dlips) > 0:
            features_cheek[sample, 2, 3] = len(longest_inc_Dlipss) / len(Dlips)
            features_cheek[sample, 2, 4] = len(longest_dec_Dlips) / len(Dlips)
            features_cheek[sample, 2, 6] = np.sum(Dlips) / len(Dlips)
            features_cheek[sample, 2, 5] = np.max(Dlips)
            features_cheek[sample, 2, 9] = np.std(Dlips)
        
        features_cheek[sample, 2, 7] = np.sum(longest_inc_Dlipss) / len(longest_inc_Dlipss)
        features_cheek[sample, 2, 8] = np.sum(np.abs(longest_dec_Dlips)) / len(longest_dec_Dlips)
        
        features_cheek[sample, 2, 10] = np.sum(longest_inc_Dlipss)
        features_cheek[sample, 2, 11] = np.sum(np.abs(longest_dec_Dlips))
        features_cheek[sample, 2, 12] = features_cheek[sample, 2, 10] - features_cheek[sample, 2, 11]
        features_cheek[sample, 2, 13] = features_cheek[sample, 2, 10] / (features_cheek[sample, 2, 10] + features_cheek[sample, 2, 11])
        features_cheek[sample, 2, 14] = features_cheek[sample, 2, 11] / (features_cheek[sample, 2, 10] + features_cheek[sample, 2, 11])

        if len(longest_inc_Dlipss) <= 1:
            features_cheek[sample, 2, 15] = 0

        else:
            features_cheek[sample, 2, 15] = np.max(np.diff(longest_inc_Dlipss))
            features_cheek[sample, 2, 17] = np.mean(np.diff(longest_inc_Dlipss))

        if len(longest_dec_Dlips) <= 1:
            features_cheek[sample, 2, 16] = 0

        else:
            features_cheek[sample, 2, 16] = np.max(np.diff(longest_dec_Dlips))
        
            features_cheek[sample, 2, 18] = np.mean(np.diff(longest_dec_Dlips))

        if len(longest_inc_Dlipss) <= 2:
            features_cheek[sample, 2, 19] = 0
        else:
            features_cheek[sample, 2, 19] = np.max(np.diff(np.diff(longest_inc_Dlipss)))
            features_cheek[sample, 2, 21] = np.mean(np.diff(np.diff(longest_inc_Dlipss)))

        if len(longest_dec_Dlips) <= 2:
            features_cheek[sample, 2, 20] = 0
        else:
            features_cheek[sample, 2, 20] = np.max(np.diff(np.diff(longest_dec_Dlips)))

        
            features_cheek[sample, 2, 22] = np.mean(np.diff(np.diff(longest_dec_Dlips)))
        
        if len(Dlips) > 0:
            features_cheek[sample, 2, 23] = features_cheek[sample, 2, 12] * frame_rate / len(Dlips)

        features_cheek = np.nan_to_num(features_cheek)
        
    features_eye = np.zeros((num_samples, 3, 24))
    
    features = np.concatenate([features[:, 0, :], features[:, 1, :], features[:, 2,:],features_eye[:, 0, :], features_eye[:, 1, :], features_eye[:, 2,:], features_cheek[:, 0, :], features_cheek[:, 1, :], features_cheek[:, 2,:]], axis=1)
    
    #print(f"Fetaures shape = {features.shape}")

    return features


import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

input_dim = 216  # number of features
hidden_dim = 1024  # number of hidden nodes
output_dim = 2  # binary classification


criterion2 = nn.MSELoss()

alphaa = 0.2
betaa = 0.8



def train(epochs,training_generator,test_generator,file):

    con = []      
    net = TemporalModel()
    net.cuda()
    
    
    lr = 0.0005
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay= 0.0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[299], gamma=0.1)
    loss_func = nn.BCELoss()
    start_time = time.time()
    best_accuracy = 0

    for epoch in range(epochs):
        train_loss  = 0
        pred_label = []
        true_label = []
        number_batch = 0
        for x, y in tqdm(training_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):
            if torch.cuda.device_count() > 0:
                x = x.cuda()
                y = y.cuda()
            
            b,d,t,n,c = x.size()
            #print(x.shape)
            x = x.view(-1,t,n,c)
            

            pred, pred1 = net(x)
            
            #print(f"Predictions shape{pred1.shape}")
            
            x = x.cpu().detach().numpy()
            
            
            features = extract_features(x)
            
            features = torch.from_numpy(features).float()
            
            features.requires_grad = True
            
            features = features.cuda()
            
            
            
            #print(f'predictions = {pred1.shape} and features = {features.shape}')
            
            
            loss2 = torch.sigmoid(criterion2(pred1, features))
            
            
            #print(f"loss = {loss_func(pred,y)}")
            
            loss = alphaa * loss_func(pred,y) + betaa * loss2
            
            pred_y = (pred >= 0.5).float()
            pred_label.append(pred_y)
            true_label.append(y)
            
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            number_batch += 1
            lr = lr_scheduler.get_last_lr()[0]

        lr_scheduler.step()
        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)
        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        output('Epoch: ' + 'train' + str(epoch) + 
              '| train accuracy: ' + str(train_accuracy.item())  + 
              '| train  loss: ' + str(train_loss / number_batch))
        print('Epoch: ' + 'train' + str(epoch) + 
              '| train accuracy: ' + str(train_accuracy.item())  + 
              '| train  loss: ' + str(train_loss / number_batch))
        
        net.eval()
        pred_label = []
        pred_avg   = []
        true_label = []
        with torch.no_grad():
          for x, y in tqdm(test_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):

              if torch.cuda.device_count() > 0:
                  x = x.cuda()
                  y = y.cuda()
                  
              b,d,t,n,c = x.size()
              x = x.view(-1,t,n,c)
              pred_y,_    = net(x)
              pred_mean = (pred_y.view(b,d).mean(1,keepdim = True) >= 0.5).float().cpu().detach()
              pred_y    = ((pred_y).view(b,d).mean(1,keepdim = True) >= 0.5).float().cpu().detach()
              pred_label.append(pred_y)
              pred_avg.append(pred_mean)
              true_label.append(y.cpu())
              
          pred_label = torch.cat(pred_label,0)
          pred_avg   = torch.cat(pred_avg,0)  
          true_label = torch.cat(true_label,0)
          
          test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
          test_avg      = torch.sum(pred_avg   == true_label).type(torch.FloatTensor) / true_label.size(0)
          con.append([epoch,test_accuracy])
          output('test accuracy: ' + str(test_accuracy.item()) + 
                '| avg accuracy: '  + str(test_avg.item()))
          print(Fore.GREEN + 'test accuracy: ' + str(test_accuracy.item()) + 
                '| avg accuracy: '  + str(test_avg.item()))

          if test_accuracy > best_accuracy:
              filepath = f"uva/{file}-{epoch:}-{loss}-{test_accuracy}.pt"
              torch.save(net.state_dict(), filepath)
            #   torch.save(net, filepath)
            #   test_frames(f'{test_accuracy}={test_f}')
              best_accuracy = test_accuracy

        net.train()
        
        output(f"ETA Per Epoch:{(time.time() - start_time) / (epoch + 1)}")
        # print(f"ETA Per Epoch:{(time.time() - start_time) / (epoch + 1)}")

    best_v = max(con,key = lambda x:x[1])
    global perf
    perf += f"best accruacy is {best_v[1]} in epoch {best_v[0]}" + "\n"
    output(perf)
    
    
image_size = 48
label_path = "labels"
data = "npy"

sometimes = lambda aug: va.Sometimes(0.5, aug)
seq = va.Sequential([
    va.RandomCrop(size=(image_size, image_size)),       
    sometimes(va.HorizontalFlip()),              
])


label_path = "labels"

def main(args):
    global output
    def output(s):
        with open(f"log_m{args.fold}a","a") as f:
            f.write(str(s) + "\n")
            
    paths = [os.path.join(label_path,file) for file in sorted(os.listdir(label_path)) if os.path.join(label_path,file)] 
    for current_path in [paths[args.fold]]: 
    
        train_labels = os.path.join(current_path,"train.json")         
        params = {"label_path": train_labels,
                  "data": data} 
                
        dg = DataGenerator(**params)
        training_generator = torch.utils.data.DataLoader(dg,batch_size=16,shuffle=True,num_workers = 2, drop_last = True)
        
                       
        test_labels    = os.path.join(current_path,"test.json")
        params = {"label_path": test_labels,
                  "data": data,
                  "test": True}    
                
        test_generator = torch.utils.data.DataLoader(DataGenerator(**params),batch_size=16,shuffle=False, num_workers = 2)
        
        train(300,training_generator,test_generator,current_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fold", default = 0, type = int)
    args = parser.parse_args()
    main(args)
