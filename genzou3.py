

import sys
import os
import glob # globは直接使わなくなりますが、将来的な拡張のために残しても良いでしょう
import numpy as np
from PIL import Image, ImageOps # Pillow (画像保存、回転に使用)
import rawpy # rawpy (RAWファイル読み込みに使用)
import torch # PyTorch (モデル定義、推論に使用)
import torch.nn as nn
import torch.nn.functional as F # Padのために必要
from datetime import datetime # タイムスタンプ取得のために必要
# import argparse # コマンドライン引数解析は不要になるため削除
import traceback # エラー時のトレースバック表示用
import warnings # torchvisionのUserWarningを抑制するために追加

# --- 設定 (ハードコード) ---
# 入力RAWファイルのパス (実際のファイルパスに置き換えてください)
RAW_FILEPATH = "R0023151.DNG"  #例: "raw/sample.cr2", "raw/image.dng"など

# 学習済みモデルファイルのパス (実際のファイルパスに置き換えてください)
MODEL_FILEPATH = "main.pth" # 例: "model_13/20231027_100000_demosaic_model_epoch_1_patch_10000.pth"

# 現像済み画像の出力先ディレクトリとファイル名接頭辞
OUTPUT_DIR = "developed_standalone"
OUTPUT_FILENAME_PREFIX = "developed_" # この接頭辞の後ろにRAWファイル名とタイムスタンプが付きます

# --- モデル・推論関連設定 ---
PATCH_SIZE = 128
OVERLAP_SIZE = 32
PATCH_STEP = PATCH_SIZE - OVERLAP_SIZE

# 推論時後処理設定
INFERENCE_CONTRAST_FACTOR = 1.015
INFERENCE_SATURATION_FACTOR = 1.5
DISPLAY_GAMMA = 2.2 # 現像後の保存画像に適用するガンマ

# デバッグモード (Trueにすると詳細情報が出力されます)
# DEBUG_MODE = False
DEBUG_MODE = True

# torchvisionのUserWarningを抑制
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

# --- モデル定義 (SimpleCNNのみ) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x

# --- ヘルパー関数 ---
def read_raw_process_linear(raw_filepath: str) -> tuple[np.ndarray, dict, int | None] | tuple[None, None, None]:
    """
    RAWファイルを読み込み、ブラックレベル補正とホワイトレベル正規化を行った
    高ビットリニアなNumPy配列 (float32), Bayerパターンマップ, および raw.sizes.flip 値を返す。
    エラー時は (None, None, None) を返す。
    """
    try:
        with rawpy.imread(raw_filepath) as raw:
            # 元の raw_image (センサー全体のデータである可能性)
            raw_image_full_sensor = raw.raw_image.astype(np.float32)
            
            if DEBUG_MODE:
                print(f"Debug (rawpy.sizes): raw_height={raw.sizes.raw_height}, raw_width={raw.sizes.raw_width}")
                print(f"Debug (rawpy.sizes): active_height={raw.sizes.height}, active_width={raw.sizes.width}")
                print(f"Debug (rawpy.sizes): top_margin={raw.sizes.top_margin}, left_margin={raw.sizes.left_margin}")
                print(f"Debug: Initial raw_image_full_sensor shape: {raw_image_full_sensor.shape}")

            top_m = raw.sizes.top_margin
            left_m = raw.sizes.left_margin
            active_h = raw.sizes.height
            active_w = raw.sizes.width
            
            if top_m < 0 or left_m < 0 or \
               top_m + active_h > raw_image_full_sensor.shape[0] or \
               left_m + active_w > raw_image_full_sensor.shape[1]:
                if DEBUG_MODE:
                    print(f"Warning: Calculated active area [{top_m}:{top_m+active_h}, {left_m}:{left_m+active_w}] "
                          f"is outside the bounds of raw_image_full_sensor {raw_image_full_sensor.shape}. "
                          "Using raw_image_full_sensor without explicit cropping based on these margins.")
                raw_image_np = raw_image_full_sensor
            else:
                if DEBUG_MODE:
                    print(f"Debug: Cropping to active area: raw_image_full_sensor[{top_m}:{top_m + active_h}, {left_m}:{left_m + active_w}]")
                raw_image_np = raw_image_full_sensor[top_m : top_m + active_h, left_m : left_m + active_w]

            h, w = raw_image_np.shape
            
            if DEBUG_MODE:
                 print(f"Debug: Raw file: {os.path.basename(raw_filepath)}")
                 print(f"Debug: Cropped raw_image_np dimensions (h, w): {h}x{w}")

            bayer_pattern_indices = raw.raw_pattern
            color_description = raw.color_desc
            black_level_per_channel = raw.black_level_per_channel
            white_level = raw.white_level

            if DEBUG_MODE:
                 print(f"Debug: Raw file: {os.path.basename(raw_filepath)}")
                 # print(f"Debug: Raw dimensions: {h}x{w}") # Redundant with above
                 print(f"Debug: Bayer pattern indices: {bayer_pattern_indices}")
                 print(f"Debug: Color description: {color_description}")
                 print(f"Debug: Black level per channel: {black_level_per_channel}")
                 print(f"Debug: White level: {white_level}")

            slice_map_for_color_role = {}
            # (Bayer pattern mapping logic - unchanged)
            color_lookup = []
            try:
                if isinstance(color_description, bytes): color_lookup = [chr(b) for b in color_description]
                elif isinstance(color_description, tuple): color_lookup = list(color_description)
                elif isinstance(color_description, str): color_lookup = list(color_description)
            except Exception as e:
                 if DEBUG_MODE: print(f"Warning: Could not parse color_description '{color_description}': {e}. Falling back.")
                 color_lookup = ['R', 'G', 'B', 'G']

            pattern_color_initials = [['', ''], ['', '']]
            try:
                if len(color_lookup) > 0:
                    for r_idx in range(2):
                        for c_idx in range(2):
                            if r_idx < bayer_pattern_indices.shape[0] and c_idx < bayer_pattern_indices.shape[1]:
                                idx = bayer_pattern_indices[r_idx, c_idx]
                                if 0 <= idx < len(color_lookup):
                                    color_str = color_lookup[idx]
                                    if len(color_str) > 0: pattern_color_initials[r_idx][c_idx] = color_str[0].upper()
                                    else: pattern_color_initials[r_idx][c_idx] = '?'
                                else: pattern_color_initials[r_idx][c_idx] = '?'
                            else: pattern_color_initials[r_idx][c_idx] = '?'
                else: pattern_color_initials = [['?', '?'], ['?', '?']]
            except Exception as e:
                 if DEBUG_MODE: print(f"Warning: Error processing raw.raw_pattern or raw.color_desc: {e}.")
                 pattern_color_initials = [['?', '?'], ['?', '?']]
            if DEBUG_MODE: print(f"Debug: Pattern color initials: {pattern_color_initials}")

            pattern_str = "".join([pattern_color_initials[r_idx][c_idx] for r_idx in range(2) for c_idx in range(2)])
            if pattern_str == "RGGB":
                slice_map_for_color_role = {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)}
            elif pattern_str == "GRBG":
                slice_map_for_color_role = {'R': (0, 1), 'Gr': (0, 0), 'Gb': (1, 1), 'B': (1, 0)}
            elif pattern_str == "GBRG":
                 slice_map_for_color_role = {'R': (1, 0), 'Gr': (1, 1), 'Gb': (0, 0), 'B': (0, 1)}
            elif pattern_str == "BGGR":
                 slice_map_for_color_role = {'R': (1, 1), 'Gr': (1, 0), 'Gb': (0, 1), 'B': (0, 0)}
            else:
                 if DEBUG_MODE: print(f"Warning: Unknown Bayer pattern: '{pattern_str}'. Attempting fallback.")
                 for r_off in range(2):
                     for c_off in range(2):
                         initial = pattern_color_initials[r_off][c_off]
                         if initial == 'R' and 'R' not in slice_map_for_color_role: slice_map_for_color_role['R'] = (r_off, c_off)
                         elif initial == 'B' and 'B' not in slice_map_for_color_role: slice_map_for_color_role['B'] = (r_off, c_off)
                         elif initial == 'G':
                              if (r_off, c_off) == (0, 1) and 'Gr' not in slice_map_for_color_role: slice_map_for_color_role['Gr'] = (r_off, c_off)
                              elif (r_off, c_off) == (1, 0) and 'Gb' not in slice_map_for_color_role: slice_map_for_color_role['Gb'] = (r_off, c_off)
                              elif 'Gr' not in slice_map_for_color_role: slice_map_for_color_role['Gr'] = (r_off, c_off)
                              elif 'Gb' not in slice_map_for_color_role: slice_map_for_color_role['Gb'] = (r_off, c_off)
            if len(slice_map_for_color_role) != 4:
                 if DEBUG_MODE: print(f"Warning: Only {len(slice_map_for_color_role)}/4 color roles mapped.")
            if DEBUG_MODE: print(f"Debug: Final slice_map_for_color_role: {slice_map_for_color_role}")

            # (Black level correction and normalization - unchanged)
            raw_image_corrected = np.copy(raw_image_np)
            slices = { (0, 0): (slice(0, None, 2), slice(0, None, 2)), (0, 1): (slice(0, None, 2), slice(1, None, 2)),
                       (1, 0): (slice(1, None, 2), slice(0, None, 2)), (1, 1): (slice(1, None, 2), slice(1, None, 2))}
            black_level_applied = False
            if len(black_level_per_channel) >= 4 and not np.all(np.abs(black_level_per_channel) < 1e-6):
                 black_level_applied = True
                 for r_off in range(2):
                     for c_off in range(2):
                         row_slice, col_slice = slices[(r_off, c_off)]
                         pattern_index = bayer_pattern_indices[r_off, c_off]
                         if pattern_index < len(black_level_per_channel):
                             bl_value = black_level_per_channel[pattern_index]
                             raw_image_corrected[row_slice, col_slice] = np.maximum(raw_image_np[row_slice, col_slice] - bl_value, 0)
                         elif DEBUG_MODE: print(f"Warning: Pattern index {pattern_index} out of bounds for black_level_per_channel.")
            elif DEBUG_MODE: print("Warning: Skipping black level correction (all zeros or not enough channels).")

            raw_image_normalized = np.zeros_like(raw_image_corrected)
            if black_level_applied and len(black_level_per_channel) >= 4:
                if DEBUG_MODE: print("Debug: Using per-channel black level corrected white level for normalization.")
                for r_off in range(2):
                    for c_off in range(2):
                        row_slice, col_slice = slices[(r_off, c_off)]
                        pattern_index = bayer_pattern_indices[r_off, c_off]
                        if pattern_index < len(black_level_per_channel):
                            bl_value = black_level_per_channel[pattern_index]
                            normalization_denom = white_level - bl_value
                            if normalization_denom > 1e-5:
                                raw_image_normalized[row_slice, col_slice] = raw_image_corrected[row_slice, col_slice] / normalization_denom
                            elif DEBUG_MODE: print(f"Warning: Normalization denominator zero for pattern ({r_off},{c_off}).")
                        elif DEBUG_MODE: print(f"Warning: Pattern index {pattern_index} OOB for black_level_per_channel during normalization.")
            else:
                 if DEBUG_MODE: print("Debug: Using global white level for normalization (fallback).")
                 normalization_denom_global = white_level
                 if normalization_denom_global > 1e-5:
                     raw_image_normalized = raw_image_corrected / normalization_denom_global
                 else:
                     print(f"Error: white_level is zero or too small ({white_level}). Cannot normalize.")
                     return None, None, None

            raw_image_normalized = np.clip(raw_image_normalized, 0, 1)
            if raw_image_normalized.ndim == 3 and raw_image_normalized.shape[2] == 1: raw_image_normalized = raw_image_normalized[:,:,0]
            elif raw_image_normalized.ndim != 2:
                 print(f"Error: Unexpected shape for normalized RAW data ({raw_image_normalized.shape}).")
                 return None, None, None
            
            flip_orientation = raw.sizes.flip # 回転情報を取得
            if DEBUG_MODE: 
                print(f"Debug: raw.sizes.flip read as: {flip_orientation}")

            if DEBUG_MODE: print(f"Debug: Finished read_raw_process_linear. Normalized data shape: {raw_image_normalized.shape}, min: {np.min(raw_image_normalized):.4f}, max: {np.max(raw_image_normalized):.4f}, flip: {flip_orientation}")
            return raw_image_normalized, slice_map_for_color_role, flip_orientation
    except rawpy.LibRawFileUnsupportedError:
        print(f"Error: {os.path.basename(raw_filepath)} is not a supported RAW file or is corrupted. Skipping.")
        return None, None, None
    except FileNotFoundError:
        print(f"Error: RAW file not found at {raw_filepath}. Skipping.")
        return None, None, None
    except Exception as e:
        print(f"Error processing RAW file {os.path.basename(raw_filepath)}: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return None, None, None

def create_4ch_input_from_raw(raw_linear_data: np.ndarray, slice_map_for_color_role: dict) -> torch.Tensor | None:
    # (Function unchanged)
    raw_h, raw_w = raw_linear_data.shape
    if DEBUG_MODE:
         print(f"Debug: create_4ch_input_from_raw - Input shape: {raw_linear_data.shape}, map: {slice_map_for_color_role}")
    required_roles = ['R', 'Gr', 'Gb', 'B']
    if not all(role in slice_map_for_color_role for role in required_roles):
         print(f"Error: Incomplete slice_map for 4ch input. Missing: {[r for r in required_roles if r not in slice_map_for_color_role]}.")
         return None
    input_4ch_tensor = torch.zeros(1, 4, raw_h, raw_w, dtype=torch.float32)
    raw_tensor = torch.from_numpy(raw_linear_data).float()
    slices_pattern = {
        (0, 0): (slice(0, None, 2), slice(0, None, 2)), (0, 1): (slice(0, None, 2), slice(1, None, 2)),
        (1, 0): (slice(1, None, 2), slice(0, None, 2)), (1, 1): (slice(1, None, 2), slice(1, None, 2)) }
    
    populated_mask = [False] * 4
    non_zero_mask = [False] * 4

    role_to_channel_idx = {'R': 0, 'Gr': 1, 'Gb': 2, 'B': 3}
    for role, std_ch_idx in role_to_channel_idx.items():
        pos = slice_map_for_color_role.get(role)
        if pos and pos in slices_pattern:
            row_slice, col_slice = slices_pattern[pos]
            channel_data = raw_tensor[row_slice, col_slice]
            input_4ch_tensor[0, std_ch_idx, row_slice, col_slice] = channel_data
            populated_mask[std_ch_idx] = True
            if torch.max(torch.abs(channel_data)).item() > 1e-9: non_zero_mask[std_ch_idx] = True
            if DEBUG_MODE: print(f"Debug: Mapped '{role}' from pattern {pos} to channel {std_ch_idx}. Data shape: {channel_data.shape}")
    
    if DEBUG_MODE:
        for i in range(4): print(f"Debug: Channel {i} populated: {populated_mask[i]}, non-zero: {non_zero_mask[i]}")
    if not all(populated_mask) or not all(non_zero_mask):
         print(f"Error: Not all channels populated or non-zero. Populated: {populated_mask}, Non-zero: {non_zero_mask}")
         return None
    if torch.max(torch.abs(input_4ch_tensor)).item() < 1e-9:
         print("Error: Generated 4ch input tensor is all zeros.")
         return None
    return input_4ch_tensor

# --- 現像処理関数 ---
def process_real_raw(raw_filepath: str, model: nn.Module, device: torch.device) -> tuple[np.ndarray | None, int | None]:
    """
    実際のRAWファイルを読み込み、フル解像度の4チャンネルデータを生成し、
    モデルでデモザイク処理（パッチ処理とスティッチング）を行い、後処理を適用する。
    現像済み画像 (NumPy配列) と回転情報 (flip) を返す。
    """
    flip_orientation_from_raw = None # 初期化
    try:
        raw_linear_data_full, slice_map_for_color_role, flip_orientation_from_raw = read_raw_process_linear(raw_filepath)
        if raw_linear_data_full is None or slice_map_for_color_role is None or len(slice_map_for_color_role) != 4:
            print(f"Error: Failed to read or map channels from {os.path.basename(raw_filepath)}. Skipping inference.")
            return None, flip_orientation_from_raw

        raw_h, raw_w = raw_linear_data_full.shape
        if DEBUG_MODE: print(f"Debug: Started process_real_raw for {os.path.basename(raw_filepath)}. Full RAW linear data shape: {raw_linear_data_full.shape}, flip: {flip_orientation_from_raw}")

        input_4ch_tensor = create_4ch_input_from_raw(raw_linear_data_full, slice_map_for_color_role)
        if input_4ch_tensor is None:
            print(f"Error: Failed to create 4-channel input from {os.path.basename(raw_filepath)}. Skipping inference.")
            return None, flip_orientation_from_raw
        
        # (推論処理 - パッチ処理とスティッチング - 変更なし)
        model.eval()
        with torch.no_grad():
            num_steps_h = (raw_h + PATCH_STEP - 1) // PATCH_STEP if raw_h > 0 else 0
            num_steps_w = (raw_w + PATCH_STEP - 1) // PATCH_STEP if raw_w > 0 else 0
            padded_h = (num_steps_h - 1) * PATCH_STEP + PATCH_SIZE if num_steps_h > 0 else 0
            padded_w = (num_steps_w - 1) * PATCH_STEP + PATCH_SIZE if num_steps_w > 0 else 0
            if raw_h > 0 and padded_h < PATCH_SIZE: padded_h = PATCH_SIZE
            if raw_w > 0 and padded_w < PATCH_SIZE: padded_w = PATCH_SIZE
            if raw_h == 0: padded_h = 0
            if raw_w == 0: padded_w = 0

            pad_right = max(0, padded_w - raw_w)
            pad_bottom = max(0, padded_h - raw_h)
            padded_input_tensor = F.pad(input_4ch_tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
            if DEBUG_MODE: print(f"Debug: Padded input tensor shape: {padded_input_tensor.shape} from raw ({raw_h}, {raw_w}) with padding ({pad_bottom}, {pad_right})")

            if padded_h == 0 or padded_w == 0:
                if DEBUG_MODE: print("Debug: Padded dimensions are zero. Returning black image.")
                # ゼロ次元の場合はflip情報に関わらずそのまま返す
                return np.zeros((raw_h, raw_w, 3), dtype=np.uint8), flip_orientation_from_raw

            output_canvas = torch.zeros(1, 3, padded_h, padded_w, dtype=torch.float32, device=device)
            weight_canvas = torch.zeros(1, 1, padded_h, padded_w, dtype=torch.float32, device=device)
            
            patch_weight_mask_2d = torch.ones(1, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32, device=device)
            if OVERLAP_SIZE > 0:
                fade_size = OVERLAP_SIZE // 2
                if fade_size > 0:
                    fade_mask_1d = torch.linspace(0, 1, fade_size)
                    weight_1d = torch.ones(PATCH_SIZE, dtype=torch.float32)
                    weight_1d[:fade_size] = fade_mask_1d
                    weight_1d[PATCH_SIZE - fade_size:] = torch.linspace(1, 0, fade_size)
                    patch_weight_mask_2d_temp = torch.min(torch.outer(weight_1d, torch.ones(PATCH_SIZE, dtype=torch.float32)),
                                                          torch.outer(torch.ones(PATCH_SIZE, dtype=torch.float32), weight_1d))
                    patch_weight_mask_2d = torch.clamp(patch_weight_mask_2d_temp.unsqueeze(0).unsqueeze(0), 0.0, 1.0).to(device)

            if DEBUG_MODE: print(f"Debug: patch_weight_mask_2d shape: {patch_weight_mask_2d.shape}")
            num_patches_processed = 0

            for i in range(num_steps_h):
                r_start = padded_h - PATCH_SIZE if i == num_steps_h - 1 and padded_h > PATCH_SIZE else i * PATCH_STEP
                for j in range(num_steps_w):
                    c_start = padded_w - PATCH_SIZE if j == num_steps_w - 1 and padded_w > PATCH_SIZE else j * PATCH_STEP
                    r_start = max(0, r_start)
                    c_start = max(0, c_start)
                    num_patches_processed += 1
                    if DEBUG_MODE and (num_patches_processed % max(1, (num_steps_h * num_steps_w) // 10) == 0 or num_patches_processed == 1):
                        print(f"Debug: Processing patch {num_patches_processed}/{num_steps_h * num_steps_w} (r_start={r_start}, c_start={c_start})")

                    input_patch = padded_input_tensor[:, :, r_start : min(r_start + PATCH_SIZE, padded_h), c_start : min(c_start + PATCH_SIZE, padded_w)].to(device)
                    if input_patch.shape[2] != PATCH_SIZE or input_patch.shape[3] != PATCH_SIZE:
                        if DEBUG_MODE: print(f"Warning: Extracted patch size mismatch: {input_patch.shape}. Skipping.")
                        continue
                    output_patch = model(input_patch)
                    if output_patch.shape[2] == PATCH_SIZE and output_patch.shape[3] == PATCH_SIZE:
                        weighted_output_patch = output_patch * patch_weight_mask_2d.expand_as(output_patch)
                        output_canvas[:, :, r_start : r_start + PATCH_SIZE, c_start : c_start + PATCH_SIZE] += weighted_output_patch
                        weight_canvas[:, :, r_start : r_start + PATCH_SIZE, c_start : c_start + PATCH_SIZE] += patch_weight_mask_2d
                    else:
                        if DEBUG_MODE: print(f"Warning: Model output patch size mismatch: {output_patch.shape}. Skipping stitch.")
            if DEBUG_MODE: print("Debug: Patch inference loop finished.")

            epsilon = 1e-6
            output_tensor = output_canvas / (weight_canvas + epsilon)
            if DEBUG_MODE: print(f"Debug: Normalized output canvas. Output tensor shape: {output_tensor.shape}")

            output_tensor_cropped = output_tensor[:, :, :raw_h, :raw_w]
            output_np = output_tensor_cropped.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
        if DEBUG_MODE:
            print(f"Debug: Model inference done. Output numpy shape: {output_np.shape}, dtype: {output_np.dtype}")
            if output_np.size > 0: print(f"Debug: Output numpy min: {np.min(output_np):.4f}, max: {np.max(output_np):.4f}, mean: {np.mean(output_np):.4f}")

        # --- 後処理 (WB, Saturation, Contrast, Gamma) ---
        # (カラーマトリクス処理は廃止済み)
        current_image_linear = np.maximum(output_np, 0.0)

        wb_gains = None
        try:
            with rawpy.imread(raw_filepath) as raw: # WBのためだけに再度開く (最適化の余地あり)
                 wb_gains = raw.camera_whitebalance
            if DEBUG_MODE:
                print(f"Debug: Camera WB gains: {wb_gains}")
        except Exception as e:
             if DEBUG_MODE: print(f"Warning: Could not read rawpy metadata for post-processing (WB): {e}")

        if wb_gains is not None and len(wb_gains) >= 4:
            try:
                r_g, g1_g, b_g, g2_g = float(wb_gains[0]), float(wb_gains[1]), float(wb_gains[2]), float(wb_gains[3])
                valid_gg = [g for g in [g1_g, g2_g] if g > 0]
                if valid_gg:
                    avg_g_gain = sum(valid_gg) / len(valid_gg)
                    if avg_g_gain > 1e-6:
                        r_gain_app = r_g / avg_g_gain
                        b_gain_app = b_g / avg_g_gain
                        if DEBUG_MODE: print(f"Debug: Applying WB Gains (R,G,B): [{r_gain_app:.3f}, 1.000, {b_gain_app:.3f}]")
                        current_image_linear_wb = np.copy(current_image_linear)
                        current_image_linear_wb[:, :, 0] *= r_gain_app
                        current_image_linear_wb[:, :, 2] *= b_gain_app
                        current_image_linear = np.maximum(current_image_linear_wb, 0.0)
                        if DEBUG_MODE: print("Debug: White balance applied.")
                    elif DEBUG_MODE: print("Warning: Avg green WB gain too small. Skipping WB.")
                elif DEBUG_MODE: print("Warning: No valid green WB gains. Skipping WB.")
            except Exception as e:
                 if DEBUG_MODE: print(f"Warning: Error applying WB: {e}. Skipping.")
        
        if DEBUG_MODE:
            print(f"Debug: After WB - R min/max/mean: {np.min(current_image_linear[:,:,0]):.3f}/{np.max(current_image_linear[:,:,0]):.3f}/{np.mean(current_image_linear[:,:,0]):.3f}")
            # G, B のmin/max/meanも同様に出力 (省略)
        
        if DEBUG_MODE:
            print("Debug: Skipping Camera Color Matrix application as per user request (or it was removed).")

        image_for_sat_cont = np.clip(current_image_linear, 0, 1)
        # (Saturation, Contrast, Gamma - 変更なし)
        if abs(INFERENCE_SATURATION_FACTOR - 1.0) > 1e-6:
            if DEBUG_MODE: print(f"Debug: Applying saturation: {INFERENCE_SATURATION_FACTOR:.3f}")
            luminance = np.dot(image_for_sat_cont[..., :3], [0.2126, 0.7152, 0.0722])
            luminance = np.maximum(luminance, 0)
            grayscale = np.stack([luminance]*3, axis=-1)
            image_for_sat_cont = grayscale + INFERENCE_SATURATION_FACTOR * (image_for_sat_cont - grayscale)
            image_for_sat_cont = np.clip(image_for_sat_cont, 0, 1)
            if DEBUG_MODE: print("Debug: Saturation applied.")

        if abs(INFERENCE_CONTRAST_FACTOR - 1.0) > 1e-6:
            if DEBUG_MODE: print(f"Debug: Applying contrast: {INFERENCE_CONTRAST_FACTOR:.3f}")
            image_for_sat_cont = 0.5 + INFERENCE_CONTRAST_FACTOR * (image_for_sat_cont - 0.5)
            image_for_sat_cont = np.clip(image_for_sat_cont, 0, 1)
            if DEBUG_MODE: print("Debug: Contrast applied.")
        
        output_np_clipped_for_gamma = image_for_sat_cont

        gamma_val = DISPLAY_GAMMA
        if abs(gamma_val - 1.0) > 1e-6:
            if DEBUG_MODE: print(f"Debug: Applying gamma: {gamma_val}")
            output_np_gamma_corrected = np.power(output_np_clipped_for_gamma, 1.0 / gamma_val)
            output_np_gamma_corrected = np.clip(output_np_gamma_corrected, 0.0, 1.0)
            if DEBUG_MODE: print("Debug: Gamma correction applied.")
        else:
            output_np_gamma_corrected = output_np_clipped_for_gamma

        output_np_final = (output_np_gamma_corrected * 255.0).astype(np.uint8)
        if DEBUG_MODE: print(f"Debug: Final conversion to uint8. Shape: {output_np_final.shape}, dtype: {output_np_final.dtype}")
        return output_np_final, flip_orientation_from_raw

    except Exception as e:
        print(f"Error processing RAW file {os.path.basename(raw_filepath)} during inference: {e}")
        if DEBUG_MODE: traceback.print_exc()
        # flip_orientation_from_raw は try の最初で設定されるため、エラー時もその値を返す
        return None, flip_orientation_from_raw

def main():
    # ハードコードされた設定値を使用
    raw_filepath = RAW_FILEPATH
    model_filepath = MODEL_FILEPATH
    output_dir = OUTPUT_DIR

    # 出力ファイル名の生成
    raw_basename = os.path.splitext(os.path.basename(raw_filepath))[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{OUTPUT_FILENAME_PREFIX}{raw_basename}_{timestamp}.png"
    output_filepath = os.path.join(output_dir, output_filename)

    print(f"--- Standalone Demosaic/Develop ---")
    print(f"Input RAW: {raw_filepath}")
    print(f"Model:     {model_filepath}")
    print(f"Output to: {output_filepath}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print(f"----------------------------------")


    if not os.path.exists(raw_filepath):
        print(f"Error: Input RAW file not found: {raw_filepath}")
        sys.exit(1)
    if not os.path.exists(model_filepath):
        print(f"Error: Model file not found: {model_filepath}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN().to(device) # SimpleCNN を使用
    print("Using SimpleCNN model.")

    try:
        model.load_state_dict(torch.load(model_filepath, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_filepath}")
    except Exception as e:
        print(f"Error loading model: {e}")
        if DEBUG_MODE: traceback.print_exc()
        sys.exit(1)

    print(f"Processing {raw_filepath}...")
    developed_image_np, flip_orientation = process_real_raw(raw_filepath, model, device)

    if developed_image_np is not None:
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")

            img_to_save = Image.fromarray(developed_image_np, 'RGB')

            # 回転処理を再有効化し、90度回転の方向を逆にしてみる
            if flip_orientation is not None:
                if DEBUG_MODE: print(f"Debug: Attempting rotation based on flip_orientation: {flip_orientation} (with 90-deg swapped)")
                
                applied_rotation = False
                if flip_orientation == 3: # 180 deg - これは変わらないはず
                    img_to_save = img_to_save.transpose(Image.ROTATE_180)
                    if DEBUG_MODE: print("Debug: Applied 180 degree rotation.")
                    applied_rotation = True
                elif flip_orientation == 5: # rawpy "90 deg CW" -> Pillow "90 deg CCW"
                    img_to_save = img_to_save.transpose(Image.ROTATE_90) # 以前はROTATE_270
                    if DEBUG_MODE: print("Debug: Applied 90 degree CCW rotation (for flip=5).")
                    applied_rotation = True
                elif flip_orientation == 6: # rawpy "90 deg CCW" -> Pillow "90 deg CW"
                    img_to_save = img_to_save.transpose(Image.ROTATE_270) # 以前はROTATE_90
                    if DEBUG_MODE: print("Debug: Applied 90 degree CW rotation (for flip=6).")
                    applied_rotation = True
                elif flip_orientation == 0:
                    if DEBUG_MODE: print("Debug: No rotation needed (flip_orientation is 0).")
                    # applied_rotation = True # 明示的に回転なし
                else:
                    if DEBUG_MODE: print(f"Warning: Unknown or unhandled flip_orientation value {flip_orientation}. No specific rotation applied for this value.")
                
                if not applied_rotation and flip_orientation != 0 and DEBUG_MODE:
                    print(f"Debug: No rotation applied for flip_orientation {flip_orientation}, but it was not 0.")
            else:
                if DEBUG_MODE: print("Warning: flip_orientation is None. No rotation applied.")
            
            img_to_save.save(output_filepath)
            print(f"Developed image saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving or rotating developed image: {e}")
            if DEBUG_MODE: traceback.print_exc()
    else:
        print(f"Failed to develop {raw_filepath}.")

if __name__ == "__main__":
    main()