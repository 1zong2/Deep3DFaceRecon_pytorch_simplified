"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
from models.networks import ReconNet
from models.bfm import ParametricFaceModel
from util.nvdiffrast import MeshRenderer
import torch
from facenet_pytorch import MTCNN

class FaceReconModel():
    def __init__(self, ckpt_path, device):
        self.device = device

        # face detection model
        self.mtcnn = MTCNN()

        # reconstruction model
        self.net_recon = ReconNet().to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.net_recon.load_state_dict(state_dict["net_recon"])
        self.net_recon.eval()

        # computing model
        self.facemodel = ParametricFaceModel(device=self.device)
        
        # mesh renderer
        self.renderer = MeshRenderer(device=self.device)


    def get_landmark(self, img):

        img = img.crop(((256 - 224)/2, (256 - 224)/2, (256 + 224)/2, (256 + 224)/2))
        _, _, lms = self.mtcnn.detect(img, landmarks=True)
        lm = np.array(lms).astype(np.float32)[0]

        return lm

    def get_coeff(self, img, lm):

        img = img.crop(((256 - 224)/2, (256 - 224)/2, (256 + 224)/2, (256 + 224)/2))
        img_tensor = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm_tensor = torch.tensor(lm).unsqueeze(0)

        self.net_recon.input_img = img_tensor.to(self.device) 
        self.net_recon.gt_lm = lm_tensor.to(self.device)

        output_coeff = self.net_recon(self.net_recon.input_img)

        return output_coeff

    def coeff_to_3dface(self, output_coeff):
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
            
        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        
        return self.pred_face
