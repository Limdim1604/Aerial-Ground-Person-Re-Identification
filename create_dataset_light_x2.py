# ==============================================================================
# [CELL 1]: CÀI ĐẶT MÔI TRƯỜNG VÀ TẢI MODEL SWINIR
# ==============================================================================
import os
import sys
import subprocess

print("Đang thiết lập môi trường và tải mã nguồn SwinIR...")
if not os.path.exists('/kaggle/working/SwinIR'):
    subprocess.run(['git', 'clone', 'https://github.com/JingyunLiang/SwinIR.git', '/kaggle/working/SwinIR'])

sys.path.append('/kaggle/working/SwinIR')

weights_path = '/kaggle/working/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
if not os.path.exists(weights_path):
    print("Đang tải bộ trọng số SwinIR-S (Lightweight) 2x...")
    subprocess.run(['wget', 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth', '-O', weights_path])

# ==============================================================================
# [CELL 2]: IMPORT THƯ VIỆN
# ==============================================================================
import cv2
import zipfile
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.network_swinir import SwinIR

# ==============================================================================
# [CELL 3]: ĐỊNH NGHĨA DATASET
# ==============================================================================
class WHUMarsDataset(Dataset):
    def __init__(self, target_dir, base_dataset_name="whu-mars"):
        self.image_paths = []
        self.rel_paths = []
        
        if not os.path.exists(target_dir):
             print(f"LỖI: Không tìm thấy thư mục {target_dir}")
             return

        print(f"Đang quét thư mục: {target_dir}")
        
        for root, _, files in os.walk(target_dir):
            for file in sorted(files):
                if file.lower().endswith('.jpg'):
                    full_path = os.path.join(root, file)
                    try:
                        split_idx = full_path.index(base_dataset_name) + len(base_dataset_name) + 1
                        rel_path = full_path[split_idx:]
                    except ValueError:
                        rel_path = os.path.relpath(full_path, target_dir)
                    
                    self.image_paths.append(full_path)
                    self.rel_paths.append(rel_path)
        
        print(f"Tổng số ảnh quét được: {len(self.image_paths)} ảnh.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        rel_path = self.rel_paths[idx]
        
        # Vẫn dùng OpenCV thay cho PIL để tăng tốc độ I/O
        img = cv2.imread(img_path)
        if img is None:
            # Xử lý an toàn ảnh lỗi
            return torch.Tensor(), rel_path
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tiền xử lý (C, H, W) và scale về [0, 1]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img_tensor, rel_path
    
# ==============================================================================
# [CELL 4]: HÀM LOAD MODEL VÀ CHẠY THỬ 5 ẢNH BẤT KỲ
# ==============================================================================
def load_s3clip_sr_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Khởi tạo mô hình SwinIR-S trên {device}...")
    
    # Kiến trúc chuẩn cho Lightweight SR (SwinIR-S)
    model = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    
    pretrained_model = torch.load(model_path, map_location=device)
    param_key = 'params_ema' if 'params_ema' in pretrained_model else 'params'
    model.load_state_dict(pretrained_model[param_key], strict=True)
    model.eval()
    return model.to(device), device

def test_5_random_images(model, device, dataset):
    print("\n--- CHẠY THỬ 5 ẢNH BẤT KỲ ĐỂ KIỂM TRA ---")
    if len(dataset) == 0: return
    
    indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    fig.suptitle('SwinIR-S (Lightweight x2) - Trước và Sau khi làm nét', fontsize=16)
    
    for i, idx in enumerate(indices):
        img_tensor, rel_path = dataset[idx]
        if img_tensor.numel() == 0: continue
            
        orig_img = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        with torch.no_grad():
            input_tensor = img_tensor.unsqueeze(0).to(device)
            _, _, h_old, w_old = input_tensor.size()
            
            # Padding
            h_pad = (h_old // 8 + 1) * 8 - h_old
            w_pad = (w_old // 8 + 1) * 8 - w_old
            h_pad = h_pad if h_pad < 8 else 0
            w_pad = w_pad if w_pad < 8 else 0
            
            if h_pad > 0:
                input_tensor = torch.cat([input_tensor, torch.flip(input_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
            if w_pad > 0:
                input_tensor = torch.cat([input_tensor, torch.flip(input_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
            
            output = model(input_tensor)
            
            # Unpadding
            output = output[..., :h_old * 2, :w_old * 2].clamp_(0, 1)
            output = (output * 255.0).round().to(torch.uint8).cpu().squeeze(0)
            
            sr_img = output.permute(1, 2, 0).numpy()
            
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Bản Gốc: {os.path.basename(rel_path)}\nShape: {orig_img.shape}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sr_img)
        axes[i, 1].set_title(f'SwinIR-S x2\nShape: {sr_img.shape}')
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.show()

# ==============================================================================
# [CELL 5]: XỬ LÝ LÀM NÉT (BATCH_SIZE=1) VÀ NÉN TRỰC TIẾP ZIP
# ==============================================================================
def process_and_zip_dataset(model, device, dataset, output_zip):
    # GIỮ NGUYÊN BATCH_SIZE=1 ĐỂ ĐẢM BẢO CHÍNH XÁC KHI PADDING THEO YÊU CẦU
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        progress_bar = tqdm(dataloader, desc="Đang xử lý Super Resolution", total=len(dataset))
        
        with torch.no_grad():
            for img_tensor, rel_path_tuple in progress_bar:
                rel_path = rel_path_tuple[0]
                # Nếu ảnh lỗi, tensor sẽ rỗng
                if img_tensor.numel() == 0:
                    continue
                    
                img_tensor = img_tensor.to(device)
                
                _, _, h_old, w_old = img_tensor.size()
                h_pad = (h_old // 8 + 1) * 8 - h_old
                w_pad = (w_old // 8 + 1) * 8 - w_old
                
                h_pad = h_pad if h_pad < 8 else 0
                w_pad = w_pad if w_pad < 8 else 0
                
                if h_pad > 0:
                    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
                if w_pad > 0:
                    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
                
                output = model(img_tensor)
                
                output = output[..., :h_old * 2, :w_old * 2].clamp_(0, 1)
                output = (output * 255.0).round().to(torch.uint8).cpu().squeeze(0)
                
                img_np = output.permute(1, 2, 0).numpy()
                
                # Nén OpenCV trực tiếp sang buffer JPEG (rất nhanh)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                success, buffer = cv2.imencode('.jpg', img_bgr)
                if success:
                    zipf.writestr(rel_path, buffer.tobytes())

TARGET_DIR = "/kaggle/input/datasets/cpkimhianh/whu-mars/train/RGB" #train/IR, test/IR,... thay thế đường link để làm nét các phần khác trong dataset
OUTPUT_ZIP_PATH = "/kaggle/working/whu-mars-s3-clip-train-rgb.zip" #đổi lại tên thư mục tương ứng với các phần cần làm nét

# ==============================================================================
# [CELL 6]: MAIN EXECUTION VÀ TẠO DATASET MỚI
# ==============================================================================

def main():
    print("\n--- BƯỚC 1: KHỞI TẠO DỮ LIỆU ---")
    dataset = WHUMarsDataset(TARGET_DIR)
    if len(dataset) == 0:
        return
                            
    print("\n--- BƯỚC 2: KHỞI TẠO MÔ HÌNH SWINIR-S (LIGHTWEIGHT) ---")
    model, device = load_s3clip_sr_model(weights_path)
    
    # [YÊU CẦU]: TEST 5 ẢNH BẤT KỲ VÀ IN RA CELL
    test_5_random_images(model, device, dataset)
    
    print("\n--- BƯỚC 3: INFERENCE BẰNG DATALOADER & TẠO FILE NÉN ZIP ---")
    process_and_zip_dataset(model, device, dataset, OUTPUT_ZIP_PATH)

if __name__ == '__main__':
    main()