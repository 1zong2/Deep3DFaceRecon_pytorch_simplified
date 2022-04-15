import os
from models.facerecon_model import FaceReconModel
from PIL import Image
import numpy as np
import torch 
import torch.nn.functional as F
from torchvision import transforms



if __name__ == "__main__":

    # gpu setting
    device = torch.device(0)
    torch.cuda.set_device(device)

    # models
    face_recon_model = FaceReconModel(ckpt_path="checkpoints/epoch_20.pth", device=device)

    # path
    load_root = "k-celeb"
    save_root = "result"
    os.makedirs(save_root, exist_ok=True)

    source_fname = "00000.png"
    target_fname = "00005.png"
    result_fname = "0to5.png"

    source_load_path = f"{load_root}/{source_fname}"
    target_load_path = f"{load_root}/{target_fname}"
    source_save_path = f"{save_root}/{source_fname}"
    target_save_path = f"{save_root}/{target_fname}"
    result_save_path = f"{save_root}/{result_fname}"
    combine_save_path = f"{save_root}/{result_fname[:-4]}_grid.png"

    ### source
    source_img = Image.open(source_load_path).convert('RGB').resize((256, 256))
    source_lm = face_recon_model.get_landmark(source_img)
    source_coeff = face_recon_model.get_coeff(source_img, source_lm)
    source_3dface = face_recon_model.coeff_to_3dface(source_coeff)
    source_3dface = F.pad(source_3dface, (16,16,16,16), "constant", 0).squeeze().cpu().detach().numpy().transpose([1,2,0]).clip(0,1)*255
    source_combine = (source_3dface + np.array(source_img))/2
    transforms.ToPILImage()(source_3dface.astype(np.uint8)).save(source_save_path)
    transforms.ToPILImage()(source_combine.astype(np.uint8)).save(f"{source_save_path[:-4]}_combine.png")

    ### target
    target_img = Image.open(target_load_path).convert('RGB').resize((256, 256))
    target_lm = face_recon_model.get_landmark(target_img)
    target_coeff = face_recon_model.get_coeff(target_img, target_lm)
    target_3dface = face_recon_model.coeff_to_3dface(target_coeff)
    target_3dface = F.pad(target_3dface, (16,16,16,16), "constant", 0).squeeze().cpu().detach().numpy().transpose([1,2,0]).clip(0,1)*255
    target_combine = (target_3dface + np.array(target_img))/2
    transforms.ToPILImage()(target_3dface.astype(np.uint8)).save(target_save_path)
    transforms.ToPILImage()(target_combine.astype(np.uint8)).save(f"{target_save_path[:-4]}_combine.png")

    ### mix
    mix_coeff = target_coeff.clone()
    # mix_coeff[:, :80] = source_coeff[:, :80] # identity
    mix_coeff[:, 80:144] = source_coeff[:, 80:144] # expression 
    # mix_coeff[:, 144:224] = source_coeff[:, 144:224] # texture
    mix_coeff[:, 224:227] = source_coeff[:, 224:227] # angles
    # mix_coeff[:, 227:254] = source_coeff[:, 227:254] # gammas
    # mix_coeff[:, 254:] = source_coeff[:, 254:] # translations

    mix_3dface = face_recon_model.coeff_to_3dface(mix_coeff)
    mix_3dface = transforms.ToPILImage()(F.pad(mix_3dface, (16,16,16,16), "constant", 0).squeeze().cpu())
    mix_3dface.save(result_save_path)

    # make grid
    img_grid = np.concatenate((source_3dface, target_3dface, mix_3dface), axis=1)
    transforms.ToPILImage()(img_grid.astype(np.uint8)).save(combine_save_path)

