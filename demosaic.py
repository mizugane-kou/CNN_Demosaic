import sys
import os
import glob
import numpy as np
from PIL import Image # Pillow (画像保存に使用)
import rawpy # rawpy (RAWファイル読み込みに使用)
import torch # PyTorch (モデル定義、学習、推論に使用)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Padのために必要
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QLabel, QProgressBar, QSizePolicy,
                               QSpacerItem, QMessageBox)
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSize
from datetime import datetime # タイムスタンプ取得のために必要
import argparse # コマンドライン引数解析のために追加 (Standaloneモード用)
import torchvision.models as models
import torch.optim.lr_scheduler
import traceback # エラー時のトレースバック表示用
import warnings # torchvisionのUserWarningを抑制するために追加


# --- 設定 ---
# ディレクトリ設定
RAW_DIR = "raw" # 実際のRAWファイル推論用
MODEL_DIR = "model_15"
DEVELOPED_DIR = "developed_output_15" # 現像済み画像保存先ディレクトリを追加
DATA_ROOT_DIR = "training_data" # dataset.pyのDEFAULT_OUTPUT_DIRに相当
INPUT_SUBDIR = "input_pseudo_bayer_linear" # dataset.pyのOUTPUT_PSEUDO_BAYER_SUBDIRに相当
TARGET_SUBDIR = "target_rgb_linear" # dataset.pyのOUTPUT_TARGET_SUBDIRに相当

# 学習データ設定
PATCH_SIZE = 128 # dataset.pyのOUTPUT_PATCH_SIZE (TILE_SIZE // 4) に合わせる

# 学習ハイパーパラメータ
NUM_EPOCHS = 15
# LEARNING_RATE = 0.00002
LEARNING_RATE = 0.00015
OPTIMIZER_WEIGHT_DECAY = 1e-2 # AdamWのweight_decay

# 学習率スケジューラ (StepLR) 設定
STEP_LR_STEP_SIZE = 1  # 学習率更新頻度
STEP_LR_GAMMA = 0.3     # 学習率更新係数


# 損失関数設定 (CombinedLoss)
L1_LOSS_WEIGHT = 1.0
PERCEPTUAL_LOSS_WEIGHT = 0.005 #知覚損失
GRADIENT_LOSS_WEIGHT = 0.1 #勾配損失
CHROMA_LOSS_WEIGHT = 0.2 # クロマ損失の重み
DARK_WEIGHT_FACTOR = 0.2 # 暗部強調のためのWeighted Lossの重み係数 (0.0で無効)
LOCAL_COLOR_CONSISTENCY_LOSS_WEIGHT = 0.2 # 局所色一貫性損失の重み (例: 0.1)
LOCAL_CONSISTENCY_PATCH_SIZE = 7          # 局所一貫性損失で使うパッチサイズ (例: 5x5)
LOCAL_CONSISTENCY_STRIDE = 1              # 局所一貫性損失で使うストライド (例: 1)


# GUI表示関連設定
DISPLAY_TRAINING_PATCHES_PROCESSED_INTERVAL = 100 # 例: 100パッチ処理ごとに更新
DISPLAY_REAL_RAW_PATCHES_PROCESSED_INTERVAL = 5000 # 例: 5000パッチ処理ごとに更新
DISPLAY_GAMMA = 2.2 # GUI表示用のガンマ補正値

# モデル保存関連設定
SAVE_MODEL_PATCHES_PROCESSED_INTERVAL = 10000 # 例: 10000パッチ処理ごとにモデルを保存

# 推論時パッチ処理設定
OVERLAP_SIZE = 32 # Overlap for patching during full-resolution inference
PATCH_STEP = PATCH_SIZE - OVERLAP_SIZE # Step size for patching

# 推論時後処理設定 (process_real_rawで使用)
INFERENCE_CONTRAST_FACTOR = 1.015
INFERENCE_SATURATION_FACTOR = 1.5

# デバッグモード
DEBUG_MODE = False
# DEBUG_MODE = True

# torchvisionのUserWarning (deprecated 'pretrained' and 'weights' parameters) を抑制
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

# 残差ブロック実装 (過剰性能のため廃止)
class ResidualBlock(nn.Module):
    """
    畳み込み層とReLUを含む基本的なResidual Block
    入力と出力のチャンネル数は同じである必要があります。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # ショートカット接続は恒等写像 (identity) なので、追加の層は不要

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # Residual Blockの出力は入力との和
        out += residual
        out = self.relu(out) # Residual Blockの最後にもReLUを適用することが多い
        return out


class ResidualDemosaicNet(nn.Module):
    """
    Residual Block を使用したデモザイク用CNNモデル
    入力: 4チャンネル (R, Gr, Gb, B) - 空間サイズは任意 (学習はPATCH_SIZE x PATCH_SIZE)
    出力: 3チャンネル (RGB) - 入力と同じ空間サイズ
    """
    def __init__(self):
        super(ResidualDemosaicNet, self).__init__()

        # 入力層: 4ch -> 64ch (Padding=1で空間サイズを維持)
        self.input_layer = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual Blocks
        # ここでは例としていくつかのResidual Blockを直列に接続
        # チャンネル数は64で固定
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64) # 必要に応じてブロック数を増減
        )

        # 出力層: 64ch -> 3ch (Padding=1で空間サイズを維持)
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 入力層
        out = self.input_layer(x)

        # Residual Blocks
        out = self.residual_blocks(out)

        # 出力層
        out = self.output_layer(out)

        return out


# CNN
class SimpleCNN(nn.Module):
    """
    単純な5層のCNNモデル
    入力: 4チャンネル (R, Gr, Gb, B) - 空間サイズは任意 (学習はPATCH_SIZE x PATCH_SIZE)
    出力: 3チャンネル (RGB) - 入力と同じ空間サイズ
    """
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


# --- モデル定義ここまで ---

# --- dataset.pyから拝借・改変したヘルパー関数 ---
# get_channel_index関数はcreate_4ch_input_from_rawの修正により不要になりますが、
# 他の場所で使用されていないか確認し、もし使用されていなければ削除しても構いません。
# 今回の修正案ではcreate_4ch_input_from_rawの引数とロジックを変更するため、get_channel_indexは直接使用しなくなります。
# 必要に応じて残すか判断してください。（今回の修正案では削除せずそのまま置いておく前提で進めます）
def get_channel_index(y, x, pos_to_channel_map):
    """
    Helper to get the standard channel index (0=R, 1=Gr, 2=Gb, 3=B) for a pixel at (y, x)
    based on the 2x2 pattern mapping derived from rawpy metadata.
    Args:
        y, x: Global coordinates of the pixel in the original RAW image.
        pos_to_channel_map: Dictionary mapping (r_off, c_off) -> standard channel_index (0-3).
                            This map is based on the 2x2 pattern relative to the top-left of the image.
                            e.g., for RGGB, {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
    Returns:
        Standard channel index (0-3) or -1 if not mapped.
    """
    # The color at (y, x) is determined by its position within the repeating 2x2 Bayer pattern
    r_off = y % 2
    c_off = x % 2
    return pos_to_channel_map.get((r_off, c_off), -1) # -1 if not mapped


def read_raw_process_linear(raw_filepath: str) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """
    RAWファイルを読み込み、ブラックレベル補正とホワイトレベル正規化を行った
    高ビットリニアなNumPy配列 (float32) と、Bayerパターン内のカラーロール
    (R, Gr, Gb, B) とそのRAWデータ内での位置 (r_off, c_off) のマッピング
    (slice_map_for_color_role) を返す。
    エラー時は (None, None) を返す。
    """
    try:
        with rawpy.imread(raw_filepath) as raw:
            # 生画像データとメタデータの取得 (uint16 -> float32)
            raw_image_np = raw.raw_image.astype(np.float32)
            h, w = raw_image_np.shape # RAW画像の高さと幅

            # RAWメタデータの取得
            bayer_pattern_indices = raw.raw_pattern # 2x2 の数値配列 (rawpy内部での色インデックス)
            color_description = raw.color_desc # バイト文字列またはタプル (rawpy内部の色インデックスに対応する色名)
            black_level_per_channel = raw.black_level_per_channel # [R, G1, G2, B] or similar
            white_level = raw.white_level # ホワイトレベル

            if DEBUG_MODE:
                 print(f"Debug: Raw file: {os.path.basename(raw_filepath)}")
                 print(f"Debug: Raw dimensions: {h}x{w}")
                 print(f"Debug: Bayer pattern indices: {bayer_pattern_indices}")
                 print(f"Debug: Color description: {color_description}")
                 print(f"Debug: Black level per channel: {black_level_per_channel}")
                 print(f"Debug: White level: {white_level}")


            # --- ベイヤーパターンの色文字列とカラーロール->位置マッピングを取得 ---
            # rawpy.raw_pattern と raw.color_desc を使ってマッピングを作成 (genzou.py を参考に改善)
            slice_map_for_color_role = {} # Maps color_role_string (R, Gr, Gb, B) -> (r_off, c_off) in RAW data

            # rawpy 内部色インデックスから色文字列を取得するルックアップテーブルを作成
            color_lookup = []
            try:
                if isinstance(color_description, bytes):
                    color_lookup = [chr(b) for b in color_description]
                elif isinstance(color_description, tuple):
                    color_lookup = list(color_description)
                elif isinstance(color_description, str):
                     color_lookup = list(color_description)

            except Exception as e:
                 if DEBUG_MODE: print(f"Warning: Could not parse color_description '{color_description}': {e}. Falling back to common RGGB color lookup.")
                 # Fallback to common assumption if description parsing fails
                 color_lookup = ['R', 'G', 'B', 'G'] # Common assumption for RGGB pattern indices

            # Determine the color string at each pattern position using raw.raw_pattern indices and color_lookup
            pattern_color_strings_full = [['', ''], ['', '']] # e.g., [['Red', 'Green'], ['Green', 'Blue']] or [['R', 'G', 'B', 'G']]
            pattern_color_initials = [['', ''], ['', '']] # e.g., [['R', 'G'], ['G', 'B']]

            try:
                # Ensure color_lookup has enough elements for the pattern indices
                if len(color_lookup) > 0:
                    for r in range(2):
                        for c in range(2):
                            # Ensure the pattern index is within the bounds of color_lookup
                            if r < bayer_pattern_indices.shape[0] and c < bayer_pattern_indices.shape[1]:
                                idx = bayer_pattern_indices[r, c] # rawpy internal color index
                                if 0 <= idx < len(color_lookup):
                                    color_str = color_lookup[idx]
                                    pattern_color_strings_full[r][c] = color_str
                                    if len(color_str) > 0:
                                        pattern_color_initials[r][c] = color_str[0].upper()
                                    else:
                                         pattern_color_initials[r][c] = '?'
                                else:
                                     if DEBUG_MODE: print(f"Warning: RawPy color index {idx} out of bounds for color_lookup ({len(color_lookup)}) at pattern position ({r},{c}) for {os.path.basename(raw_filepath)}. Pattern initial set to '?'.")
                                     pattern_color_initials[r][c] = '?'
                            else:
                                if DEBUG_MODE: print(f"Warning: Pattern position ({r},{c}) out of bounds for raw_pattern shape {bayer_pattern_indices.shape} for {os.path.basename(raw_filepath)}. Pattern initial set to '?'.")
                                pattern_color_initials[r][c] = '?'

                else:
                     if DEBUG_MODE: print(f"Warning: color_lookup is empty for {os.path.basename(raw_filepath)}. Cannot determine pattern colors accurately. Pattern initials set to '?'.")
                     pattern_color_initials = [['?', '?'], ['?', '?']] # Cannot determine pattern

            except Exception as e:
                 if DEBUG_MODE: print(f"Warning: Error processing raw.raw_pattern or raw.color_desc for {os.path.basename(raw_filepath)}: {e}. Pattern initials set to '?'.")
                 pattern_color_initials = [['?', '?'], ['?', '?']] # Error in processing

            if DEBUG_MODE:
                 print(f"Debug: Pattern color initials: {pattern_color_initials}")


            # Determine the pattern type based on the initials string (e.g., "RGGB")
            pattern_str = "".join([pattern_color_initials[r][c] for r in range(2) for c in range(2)])

            # Map color roles to slice positions based on the detected pattern type
            # Use the most common patterns first, then fallback
            if pattern_str == "RGGB":
                slice_map_for_color_role['R'] = (0, 0)
                slice_map_for_color_role['Gr'] = (0, 1)
                slice_map_for_color_role['Gb'] = (1, 0)
                slice_map_for_color_role['B'] = (1, 1)
            elif pattern_str == "GRBG":
                slice_map_for_color_role['R'] = (0, 1)
                slice_map_for_color_role['Gr'] = (0, 0)
                slice_map_for_color_role['Gb'] = (1, 1)
                slice_map_for_color_role['B'] = (1, 0)
            elif pattern_str == "GBRG":
                 slice_map_for_color_role['R'] = (1, 0)
                 slice_map_for_color_role['Gr'] = (1, 1)
                 slice_map_for_color_role['Gb'] = (0, 0)
                 slice_map_for_color_role['B'] = (0, 1)
            elif pattern_str == "BGGR":
                 slice_map_for_color_role['R'] = (1, 1)
                 slice_map_for_color_role['Gr'] = (1, 0)
                 slice_map_for_color_role['Gb'] = (0, 1)
                 slice_map_for_color_role['B'] = (0, 0)
            else:
                 # Fallback logic if pattern is not a standard type or detection failed
                 if DEBUG_MODE: print(f"Warning: Unknown or non-standard Bayer pattern detected: '{pattern_str}' for {os.path.basename(raw_filepath)}. Attempting fallback mapping based on initials.")
                 # Try mapping based on initials, prioritizing standard Gr/Gb positions
                 for r_off in range(2):
                     for c_off in range(2):
                         initial = pattern_color_initials[r_off][c_off]
                         if initial == 'R':
                              slice_map_for_color_role['R'] = (r_off, c_off)
                         elif initial == 'B':
                              slice_map_for_color_role['B'] = (r_off, c_off)
                         elif initial == 'G':
                              # Assign G based on standard Gr/Gb positions (0,1) and (1,0) relative to pattern start
                              if (r_off, c_off) == (0, 1) and 'Gr' not in slice_map_for_color_role:
                                  slice_map_for_color_role['Gr'] = (r_off, c_off)
                              elif (r_off, c_off) == (1, 0) and 'Gb' not in slice_map_for_color_role:
                                   slice_map_for_color_role['Gb'] = (r_off, c_off)
                              else:
                                   # If G is at a non-standard position OR Gr/Gb are already mapped,
                                   # try to map to the remaining G channel if any.
                                   if 'Gr' not in slice_map_for_color_role:
                                        slice_map_for_color_role['Gr'] = (r_off, c_off) # Assign to Gr if not yet mapped
                                   elif 'Gb' not in slice_map_for_color_role:
                                        slice_map_for_color_role['Gb'] = (r_off, c_off) # Assign to Gb if not yet mapped
                                   elif DEBUG_MODE:
                                         print(f"Warning: G pixel at pattern position ({r_off},{c_off}) could not be confidently mapped to Gr or Gb role for {os.path.basename(raw_filepath)}. Its data might not be used correctly in 4ch input.")


            # Check if all 4 color roles (R, Gr, Gb, B) were mapped
            if len(slice_map_for_color_role) != 4:
                 if DEBUG_MODE:
                      print(f"Warning: Only {len(slice_map_for_color_role)}/4 color roles mapped for {os.path.basename(raw_filepath)}. Data generation might be incorrect.")
                 # If mapping is incomplete, data generation might fail or produce incorrect results.
                 # We'll proceed but expect potential issues.
                 # It's also possible the raw file is non-standard (e.g., 3-color filter array)

            if DEBUG_MODE:
                 print(f"Debug: Final slice_map_for_color_role: {slice_map_for_color_role}")


            # --- ブラックレベル補正 ---
            raw_image_corrected = np.copy(raw_image_np) # 補正用にコピーを作成
            # 2x2 パターン内の各位置に対応するグローバルなスライス
            slices = {
                (0, 0): (slice(0, None, 2), slice(0, None, 2)), # ::2, ::2
                (0, 1): (slice(0, None, 2), slice(1, None, 2)), # ::2, 1::2
                (1, 0): (slice(1, None, 2), slice(0, None, 2)), # 1::2, ::2
                (1, 1): (slice(1, None, 2), slice(1, None, 2)), # 1::2, 1::2
            }

            # Check if black_level_per_channel is all zeros, which might indicate an issue
            # Add a small tolerance for floating point comparisons
            if len(black_level_per_channel) >= 4 and np.all(np.abs(black_level_per_channel) < 1e-6):
                 if DEBUG_MODE:
                      print(f"Warning: Black level per channel is all zeros or near-zero for {os.path.basename(raw_filepath)}. This is unusual and might indicate incorrect metadata or a problem with black level reading. Skipping black level correction based on per-channel values.")
                 # Fallback to assuming no black level correction is needed or a global black level if available/known.
                 # For now, we'll proceed without per-channel black level correction if they are all zero.
                 # The normalization step will still happen using white_level.
                 black_level_applied = False # Flag to indicate if black level correction was applied
            elif len(black_level_per_channel) >= 4:
                 black_level_applied = True
                 # Apply black level correction based on raw.raw_pattern index mapping to black_level_per_channel
                 # This assumes rawpy's black_level_per_channel ordering corresponds to raw.raw_pattern positions' color indices
                 for r_off in range(2):
                     for c_off in range(2):
                         row_slice, col_slice = slices[(r_off, c_off)]
                         # raw.raw_pattern で示される、この位置の色に対応するrawpy内部インデックス
                         pattern_index = bayer_pattern_indices[r_off, c_off] # black_level_per_channel へのインデックス (RawPyの慣例を想定)

                         if pattern_index < len(black_level_per_channel):
                             bl_value = black_level_per_channel[pattern_index]
                             raw_image_corrected[row_slice, col_slice] = np.maximum(raw_image_np[row_slice, col_slice] - bl_value, 0)
                         elif DEBUG_MODE:
                             print(f"Warning: Pattern index {pattern_index} out of bounds for black_level_per_channel ({len(black_level_per_channel)}) at position ({r_off},{c_off}) for {os.path.basename(raw_filepath)}. Pixels not black level corrected.")
            else:
                 if DEBUG_MODE:
                      print("Warning: black_level_per_channel is not available or too short (<4 elements). Skipping black level correction.")
                 black_level_applied = False


            # --- 正規化 (0-1 スケール) ---
            raw_image_normalized = np.zeros_like(raw_image_corrected) # 正規化後の画像を格納する配列 (float32)

            # Normalization denominator calculation needs care, especially if black levels were skipped.
            # If black_level_per_channel was used for correction: (white_level - bl_value)
            # If black levels were skipped (all zero or missing): use white_level
            # If black_level_per_channel was available and used, normalize per-channel based on WL - BL.
            # If black_level_per_channel was not available or all zeros, normalize globally by WL (after any skipped black level subtraction).
            if black_level_applied and len(black_level_per_channel) >= 4:
                if DEBUG_MODE: print("Debug: Using per-channel black level corrected white level for normalization.")
                # 各ピクセル位置に対応するブラックレベル差を用いて正規化
                for r_off in range(2):
                    for c_off in range(2):
                        row_slice, col_slice = slices[(r_off, c_off)]
                        # このピクセル位置に対応する Bayer パターン内のインデックス (raw.raw_pattern)
                        pattern_index = bayer_pattern_indices[r_off, c_off]

                        if pattern_index < len(black_level_per_channel):
                            # このピクセル位置に対応するブラックレベル値
                            bl_value = black_level_per_channel[pattern_index]
                            # このピクセルタイプにおけるブラックレベル補正後の有効な値の範囲 (0からの開始を想定)
                            # したがって、正規化のための分母は white_level - bl_value となる
                            normalization_denom = white_level - bl_value

                            if normalization_denom > 1e-5: # ゼロ除算を防ぐ
                                # このスライスの正規化を実行: V_corrected / (WL - BL)
                                # raw_image_corrected は既に V_raw - BL となっている
                                raw_image_normalized[row_slice, col_slice] = raw_image_corrected[row_slice, col_slice] / normalization_denom
                            elif DEBUG_MODE:
                                 print(f"Warning: Normalization denominator zero or negative for pattern position ({r_off},{c_off}) for {os.path.basename(raw_filepath)}. Setting slice to 0.")
                                 raw_image_normalized[row_slice, col_slice] = 0.0
                        elif DEBUG_MODE:
                             # black_level_per_channel が4つ以上ある場合は、通常ここには来ないはず
                             print(f"Warning: Pattern index {pattern_index} out of bounds for black_level_per_channel ({len(black_level_per_channel)}) at position ({r_off},{c_off}) for {os.path.basename(raw_filepath)}. Setting slice to 0.")
                             raw_image_normalized[row_slice, col_slice] = 0.0
            else:
                 # black_level_applied が False の場合、または black_level_per_channel が不足している場合
                 if DEBUG_MODE: print("Debug: Using global white level for normalization (fallback).")
                 # black level補正がスキップされている可能性があるので、white_level - 0 として扱う
                 normalization_denom_global = white_level # assumption: black level was ~0 or already handled differently by rawpy internal processing leading to raw_image_corrected

                 if normalization_denom_global > 1e-5:
                     raw_image_normalized = raw_image_corrected / normalization_denom_global
                 else:
                     print(f"Error: white_level is zero or too small ({white_level}) for {os.path.basename(raw_filepath)}. Cannot normalize. Skipping.")
                     return None, None


            # 正規化後のデータが確実に [0, 1] の範囲に収まるようにクリップ
            # 計算誤差や特殊なケースでわずかに範囲外に出る可能性に備える
            raw_image_normalized = np.clip(raw_image_normalized, 0, 1)

            # Ensure raw_image_normalized is float32 HxW before returning
            if raw_image_normalized.ndim == 3 and raw_image_normalized.shape[2] == 1:
                 raw_image_normalized = raw_image_normalized[:,:,0]
            elif raw_image_normalized.ndim != 2:
                 print(f"Error: Unexpected shape for normalized RAW data ({raw_image_normalized.shape}) for {os.path.basename(raw_filepath)}. Expecting HxW after processing. Skipping.")
                 return None, None

            if DEBUG_MODE:
                 print(f"Debug: Finished read_raw_process_linear. Normalized data shape: {raw_image_normalized.shape}, min: {np.min(raw_image_normalized):.4f}, max: {np.max(raw_image_normalized):.4f}")


            # raw_image_normalized と slice_map_for_color_role を返す
            return raw_image_normalized, slice_map_for_color_role

    except rawpy.LibRawFileUnsupportedError:
        print(f"Error: {os.path.basename(raw_filepath)} is not a supported RAW file or is corrupted. Skipping.")
        return None, None
    except FileNotFoundError:
        print(f"Error: RAW file not found at {raw_filepath}. Skipping.")
        return None, None
    except Exception as e:
        print(f"Error processing RAW file {os.path.basename(raw_filepath)}: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return None, None


def create_4ch_input_from_raw(raw_linear_data: np.ndarray, slice_map_for_color_role: dict) -> torch.Tensor | None:
    """
    高ビットリニアなフル解像度RAWデータ (HxW float32) とカラーロール->位置マッピングから、
    フル解像度の4チャンネル入力テンソル (1, 4, H, W float32) を生成する。
    チャンネルの順序は R, Gr, Gb, B (標準順序) となる。
    """
    raw_h, raw_w = raw_linear_data.shape

    if DEBUG_MODE:
         print(f"Debug: create_4ch_input_from_raw - Input raw_linear_data shape: {raw_linear_data.shape}")
         print(f"Debug: create_4ch_input_from_raw - slice_map_for_color_role: {slice_map_for_color_role}")


    # 必要なカラーロール (R, Gr, Gb, B) が全てマッピングに含まれているか確認
    required_roles = ['R', 'Gr', 'Gb', 'B']
    if not all(role in slice_map_for_color_role for role in required_roles):
         print(f"Error: Incomplete slice_map_for_color_role ({len(slice_map_for_color_role)}/4) in create_4ch_input_from_raw. Missing roles: {[role for role in required_roles if role not in slice_map_for_color_role]}. Cannot create 4-channel input.")
         return None

    # 4チャンネル入力テンソルを初期化 (B=1, C=4, H, W). チャンネルは R, Gr, Gb, B の標準順序
    input_4ch_tensor = torch.zeros(1, 4, raw_h, raw_w, dtype=torch.float32)

    # RAWデータ (HxW) をPyTorch Tensorに変換
    raw_tensor = torch.from_numpy(raw_linear_data).float() # HxW float32

    # 2x2 パターン内の各位置に対応するグローバルなスライス
    slices = {
        (0, 0): (slice(0, None, 2), slice(0, None, 2)), # ::2, ::2
        (0, 1): (slice(0, None, 2), slice(1, None, 2)), # ::2, 1::2
        (1, 0): (slice(1, None, 2), slice(0, None, 2)), # 1::2, ::2
        (1, 1): (slice(1, None, 2), slice(1, None, 2)), # 1::2, 1::2
    }

    # カラーロールマップを使って、RAWデータの対応する位置からデータを抽出し、
    # 4チャンネル入力テンソルの適切なチャンネルに配置 (R=0, Gr=1, Gb=2, B=3)
    channels_populated_mask = [False] * 4 # 各標準チャンネルにデータが入ったか追跡
    channels_with_non_zero_data = [False] * 4 # 各標準チャンネルに非ゼロデータが入ったか追跡


    # Process R channel (standard channel 0)
    r_pos = slice_map_for_color_role.get('R')
    if r_pos and r_pos in slices:
        row_slice, col_slice = slices[r_pos]
        channel_data = raw_tensor[row_slice, col_slice]
        input_4ch_tensor[0, 0, row_slice, col_slice] = channel_data
        channels_populated_mask[0] = True
        if torch.max(torch.abs(channel_data)).item() > 1e-9: channels_with_non_zero_data[0] = True
        if DEBUG_MODE: print(f"Debug: Mapped color role 'R' from pattern position {r_pos} to channel 0. Data shape: {channel_data.shape}")

    # Process Gr channel (standard channel 1)
    gr_pos = slice_map_for_color_role.get('Gr')
    if gr_pos and gr_pos in slices:
        row_slice, col_slice = slices[gr_pos]
        channel_data = raw_tensor[row_slice, col_slice]
        input_4ch_tensor[0, 1, row_slice, col_slice] = channel_data
        channels_populated_mask[1] = True
        if torch.max(torch.abs(channel_data)).item() > 1e-9: channels_with_non_zero_data[1] = True
        if DEBUG_MODE: print(f"Debug: Mapped color role 'Gr' from pattern position {gr_pos} to channel 1. Data shape: {channel_data.shape}")

    # Process Gb channel (standard channel 2)
    gb_pos = slice_map_for_color_role.get('Gb')
    if gb_pos and gb_pos in slices:
        row_slice, col_slice = slices[gb_pos]
        channel_data = raw_tensor[row_slice, col_slice]
        input_4ch_tensor[0, 2, row_slice, col_slice] = channel_data
        channels_populated_mask[2] = True
        if torch.max(torch.abs(channel_data)).item() > 1e-9: channels_with_non_zero_data[2] = True
        if DEBUG_MODE: print(f"Debug: Mapped color role 'Gb' from pattern position {gb_pos} to channel 2. Data shape: {channel_data.shape}")

    # Process B channel (standard channel 3)
    b_pos = slice_map_for_color_role.get('B')
    if b_pos and b_pos in slices:
        row_slice, col_slice = slices[b_pos]
        channel_data = raw_tensor[row_slice, col_slice]
        input_4ch_tensor[0, 3, row_slice, col_slice] = channel_data
        channels_populated_mask[3] = True
        if torch.max(torch.abs(channel_data)).item() > 1e-9: channels_with_non_zero_data[3] = True
        if DEBUG_MODE: print(f"Debug: Mapped color role 'B' from pattern position {b_pos} to channel 3. Data shape: {channel_data.shape}")


    if DEBUG_MODE:
         print(f"Debug: create_4ch_input_from_raw - Check if all standard channels (0-3) were populated and contain non-zero data:")
         for i in range(4):
             print(f"Debug: Channel {i} populated: {channels_populated_mask[i]}, contains non-zero data: {channels_with_non_zero_data[i]}")

         # Check the final tensor content
         print(f"Debug: Final input_4ch_tensor shape: {input_4ch_tensor.shape}")
         print(f"Debug: Final input_4ch_tensor min: {torch.min(input_4ch_tensor).item():.4f}")
         print(f"Debug: Final input_4ch_tensor max: {torch.max(input_4ch_tensor).item():.4f}")
         print(f"Debug: Final input_4ch_tensor mean: {torch.mean(input_4ch_tensor).item():.4f}")
         # Check if the final tensor contains any non-zero data across all channels
         if torch.max(torch.abs(input_4ch_tensor)).item() > 1e-9:
              print("Debug: Final input_4ch_tensor contains non-zero values.")
         else:
              print("Warning: Final input_4ch_tensor appears to be all zeros or very close to zero.")


    # Check if the tensor is effectively all zeros (indicates a major mapping issue or zero input data)
    # また、全ての必須カラーロールからデータが正常にマップされたかを確認する
    if not all(channels_populated_mask):
         print("Error: Not all standard channels (0-3) were populated from the RAW image based on the color role mapping.")
         return None

    if not all(channels_with_non_zero_data):
         print("Error: Not all standard channels (0-3) received non-zero data from the RAW image based on the color role mapping.")
         return None

    if torch.max(torch.abs(input_4ch_tensor)).item() < 1e-9: # 非常に小さい値もしきい値と比較
         print("Error: Generated 4-channel input tensor is all zeros or very close to zero even after populating channels.")
         return None


    return input_4ch_tensor

# --- 実際のRAWファイル処理関数 (推論用) ---
def process_real_raw(raw_filepath: str, model: nn.Module, device: torch.device) -> np.ndarray | None:
    """
    実際のRAWファイル（.NEFなど）を読み込み、フル解像度の4チャンネルデータを生成し、
    モデルでデモザイク処理（パッチ処理とスティッチング）を行い、後処理を適用する。
    GUI表示と保存のための関数として使用される。
    """
    # CONTRAST_FACTOR と SATURATION_FACTOR はファイル冒頭のグローバル設定から取得
    # CONTRAST_FACTOR = 1.015 (INFERENCE_CONTRAST_FACTOR)
    # SATURATION_FACTOR = 1.5 (INFERENCE_SATURATION_FACTOR)

    try:
        # RAWファイルを読み込み、ブラックレベル補正・正規化済みのリニアデータとチャンネルマッピングを取得
        raw_linear_data_full, slice_map_for_color_role = read_raw_process_linear(raw_filepath)

        if raw_linear_data_full is None or slice_map_for_color_role is None or len(slice_map_for_color_role) != 4:
             print(f"Error: Failed to read or map channels from {os.path.basename(raw_filepath)}. Skipping inference.")
             return None

        raw_h, raw_w = raw_linear_data_full.shape
        if DEBUG_MODE:
             print(f"Debug: Started process_real_raw for {os.path.basename(raw_filepath)}. Full RAW linear data shape: {raw_linear_data_full.shape}")


        # フル解像度の4チャンネル入力テンソルを生成
        # This tensor is now the input for patching
        input_4ch_tensor = create_4ch_input_from_raw(raw_linear_data_full, slice_map_for_color_role)

        if input_4ch_tensor is None:
             print(f"Error: Failed to create 4-channel input from {os.path.basename(raw_filepath)}. Skipping inference.")
             return None

        # input_4ch_tensor is already on the correct device from create_4ch_input_from_raw

        # --- モデルによる推論 (パッチ処理とスティッチング) ---
        model.eval() # モデルを評価モードに設定
        with torch.no_grad(): # 推論時は勾配計算は不要

            # Calculate number of steps needed to cover the image with overlap
            # num_steps = ceil(raw_dim / PATCH_STEP)
            # Using integer division and ceil: ceil(a/b) = (a + b - 1) // b for a > 0
            # If raw_dim is 0, num_steps is 0.
            # This calculation ensures that even the last pixel is covered by at least one patch.
            num_steps_h = (raw_h + PATCH_STEP - 1) // PATCH_STEP if raw_h > 0 else 0
            num_steps_w = (raw_w + PATCH_STEP - 1) // PATCH_STEP if raw_w > 0 else 0

            # Calculate required padded dimensions based on the number of steps
            # The padded dimension should be large enough to contain the last patch, whose starting position is (num_steps - 1) * PATCH_STEP.
            # The last patch ends at (num_steps - 1) * PATCH_STEP + PATCH_SIZE - 1.
            # So, padded_dim must be at least (num_steps - 1) * PATCH_STEP + PATCH_SIZE.
            # If num_steps is 0, padded_dim is 0.
            padded_h = (num_steps_h - 1) * PATCH_STEP + PATCH_SIZE if num_steps_h > 0 else 0
            padded_w = (num_steps_w - 1) * PATCH_STEP + PATCH_SIZE if num_steps_w > 0 else 0

            # Sanity check: if original size is small, padded size must be at least PATCH_SIZE to accommodate one patch
            if raw_h > 0 and padded_h < PATCH_SIZE: padded_h = PATCH_SIZE
            if raw_w > 0 and padded_w < PATCH_SIZE: padded_w = PATCH_SIZE
            # If original size is 0, padded size must be 0.
            if raw_h == 0: padded_h = 0
            if raw_w == 0: padded_w = 0


            # Calculate padding amounts for F.pad (left, right, top, bottom)
            pad_right = padded_w - raw_w
            pad_bottom = padded_h - raw_h
            # Ensure padding is non-negative
            pad_right = max(0, pad_right)
            pad_bottom = max(0, pad_bottom)

            # Pad the input tensor
            # Use reflection padding for potentially better edge handling
            # pad = (pad_left, pad_right, pad_top, pad_bottom)
            padded_input_tensor = F.pad(input_4ch_tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
            if DEBUG_MODE: print(f"Debug: Padded input tensor shape: {padded_input_tensor.shape} from raw shape ({raw_h}, {raw_w}) with padding ({pad_bottom}, {pad_right}) bottom/right.")

            # Handle the case of zero dimensions after padding calculation (e.g., if raw_h or raw_w was 0)
            if padded_h == 0 or padded_w == 0:
                 if DEBUG_MODE: print("Debug: Padded dimensions are zero. Skipping patch processing.")
                 # Return a black image of the original size
                 output_np = np.zeros((raw_h, raw_w, 3), dtype=np.uint8)
                 # Skip the rest of the patching and post-processing
                 return output_np


            # Initialize output canvas and weight canvas based on the padded dimensions
            output_canvas = torch.zeros(1, 3, padded_h, padded_w, dtype=torch.float32, device=device)
            weight_canvas = torch.zeros(1, 1, padded_h, padded_w, dtype=torch.float32, device=device) # Use 1 channel for weight


            # Generate a simple 2D weight mask for blending
            # Linear fade from 0 at edge to 1 in center of the overlap region
            # Total fade distance is OVERLAP_SIZE. Fade-in over first half, fade-out over second half.
            # Weight should be 1 in the non-overlapping center region.
            patch_weight_mask = torch.ones(PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)
            fade_size = OVERLAP_SIZE // 2 # Fade region extent on one side

            # Create 1D fade mask for edges
            # Ensure fade_size is at least 1 if OVERLAP_SIZE > 0 to avoid empty linspace
            if OVERLAP_SIZE > 0: # Only apply fading if there is overlap
                if fade_size > 0:
                    fade_mask_1d = torch.linspace(0, 1, fade_size) # Goes from 0 to 1
                    weight_1d = torch.ones(PATCH_SIZE, dtype=torch.float32)
                    # Apply fade-in to the start
                    weight_1d[:fade_size] = fade_mask_1d
                    # Apply fade-out to the end
                    weight_1d[PATCH_SIZE - fade_size:] = torch.linspace(1, 0, fade_size) # Goes from 1 to 0

                    # Create 2D weight mask by outer product
                    patch_weight_mask_2d = torch.min(torch.outer(weight_1d, torch.ones(PATCH_SIZE, dtype=torch.float32)),
                                                     torch.outer(torch.ones(PATCH_SIZE, dtype=torch.float32), weight_1d)).unsqueeze(0).unsqueeze(0).to(device)
                    # Clamp any potential small negative values due to float precision
                    patch_weight_mask_2d = torch.clamp(patch_weight_mask_2d, 0.0, 1.0)
                else:
                     # If OVERLAP_SIZE > 0 but fade_size is 0 (e.g. OVERLAP_SIZE = 1), use a simple box filter (no fade)
                     patch_weight_mask_2d = torch.ones(1, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32, device=device)
            else:
                # No overlap, weight mask is all ones (no blending)
                patch_weight_mask_2d = torch.ones(1, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32, device=device)


            if DEBUG_MODE: print(f"Debug: Generated patch_weight_mask_2d shape: {patch_weight_mask_2d.shape}")


            # Iterate over patches and perform inference
            if DEBUG_MODE: print("Debug: Starting patch inference loop...")
            num_patches_processed = 0

            # Number of steps in height and width based on padded dimensions and step size
            # This was already calculated as num_steps_h and num_steps_w

            if DEBUG_MODE: print(f"Debug: Patching: Padded H={padded_h}, W={padded_w}. Patch Size={PATCH_SIZE}, Overlap={OVERLAP_SIZE}, Step={PATCH_STEP}")
            if DEBUG_MODE: print(f"Debug: Number of steps H={num_steps_h}, W={num_steps_w}. Total patches = {num_steps_h * num_steps_w}")

            # Handle case where num_steps is 0 (e.g., raw_h or raw_w was 0) - already handled by padded_h/w check and return
            if num_steps_h == 0 or num_steps_w == 0:
                 if DEBUG_MODE: print("Debug: Number of steps is zero. Skipping patch processing loop.")
                 # Return a black image of the original size (handled before the loop)
                 pass # Continue to post-processing if somehow reached here with zero steps


            for i in range(num_steps_h):
                # Calculate the starting row for the patch
                # If it's the last step, adjust the start position to ensure the patch ends exactly at padded_h
                # This logic is crucial to cover the very end of the padded image.
                if i == num_steps_h - 1 and padded_h > PATCH_SIZE:
                    r_start = padded_h - PATCH_SIZE
                else:
                    r_start = i * PATCH_STEP

                for j in range(num_steps_w):
                    # Calculate the starting column for the patch
                    # If it's the last step, adjust the start position to ensure the patch ends exactly at padded_w
                    if j == num_steps_w - 1 and padded_w > PATCH_SIZE:
                        c_start = padded_w - PATCH_SIZE
                    else:
                        c_start = j * PATCH_STEP


                    # Ensure calculated start positions are within valid range just in case
                    r_start = max(0, r_start)
                    c_start = max(0, c_start)


                    num_patches_processed += 1
                    # Reduced print frequency during patch processing
                    if DEBUG_MODE and (num_patches_processed <= min(5, num_steps_h * num_steps_w) or num_patches_processed % max(1, (num_steps_h * num_steps_w) // 20) == 0 or num_patches_processed == num_steps_h * num_steps_w):
                         # Print first few, last one, and ~20 evenly spaced prints
                         print(f"Debug: Processing patch {num_patches_processed}/{num_steps_h * num_steps_w} (r_start={r_start}, c_start={c_start})...")


                    # Extract patch from the padded input tensor
                    # Ensure the slice does not exceed padded dimensions (should be guaranteed by padded_h/w and r_start/c_start logic, but defensive slicing)
                    input_patch = padded_input_tensor[:, :, r_start : min(r_start + PATCH_SIZE, padded_h), c_start : min(c_start + PATCH_SIZE, padded_w)].to(device)

                    # If for some reason the extracted patch is not the expected size (e.g., at boundaries if logic is slightly off), handle this
                    if input_patch.shape[2] != PATCH_SIZE or input_patch.shape[3] != PATCH_SIZE:
                        if DEBUG_MODE: print(f"Warning: Extracted patch size mismatch: {input_patch.shape}. Expected: (1, 4, {PATCH_SIZE}, {PATCH_SIZE}). Skipping patch.")
                        continue # Skip this patch


                    # Inference on patch
                    output_patch = model(input_patch) # Shape [1, 3, PATCH_SIZE, PATCH_SIZE]

                    # Apply weight mask to the output patch and add to canvas
                    # Ensure weight mask matches output patch size (should be true if input patch was correct size)
                    if output_patch.shape[2] == PATCH_SIZE and output_patch.shape[3] == PATCH_SIZE:
                         weighted_output_patch = output_patch * patch_weight_mask_2d.expand_as(output_patch) # Expand mask to 3 channels
                         output_canvas[:, :, r_start : r_start + PATCH_SIZE, c_start : c_start + PATCH_SIZE] += weighted_output_patch
                         # Add weight mask to the weight canvas
                         weight_canvas[:, :, r_start : r_start + PATCH_SIZE, c_start : c_start + PATCH_SIZE] += patch_weight_mask_2d # Mask is [1, 1, H, W]
                    else:
                         if DEBUG_MODE: print(f"Warning: Model output patch size mismatch: {output_patch.shape}. Expected: (1, 3, {PATCH_SIZE}, {PATCH_SIZE}). Skipping stitching for this patch.")


            if DEBUG_MODE: print("Debug: Patch inference loop finished.")

            # Normalize the output canvas by the weight canvas
            # Add a small epsilon to weight_canvas to prevent division by zero
            epsilon = 1e-6
            output_tensor = output_canvas / (weight_canvas + epsilon)
            if DEBUG_MODE: print(f"Debug: Normalized output canvas. Output tensor shape: {output_tensor.shape}")


            # Tensor to NumPy (HWC, float32)
            # Crop back to original dimensions (raw_h, raw_w) before converting to numpy for post-processing
            # Ensure slicing doesn't go out of bounds of output_tensor
            # The padded tensor covers at least the original dimensions, so cropping should be safe.
            output_tensor_cropped = output_tensor[:, :, :raw_h, :raw_w]
            output_np = output_tensor_cropped.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() # Shape (raw_h, raw_w, 3)


        if DEBUG_MODE:
             print(f"Debug: Model inference (patching/stitching) finished. Output numpy shape: {output_np.shape}, dtype: {output_np.dtype}")
             # Check if the cropped numpy array seems valid
             if output_np.size > 0:
                  print(f"Debug: Output numpy min: {np.min(output_np):.4f}, max: {np.max(output_np):.4f}, mean: {np.mean(output_np):.4f}")
             else:
                  print("Debug: Output numpy array is empty.")


        # --- 後処理 (WB, Color Matrix, Saturation, Contrast, Gamma) ---
        # Apply post-processing to the stitched numpy array (output_np)
        # The rest of the post-processing logic remains the same, operating on output_np

        # RawPy instance already used above to get WB and Color Matrix
        # Re-open rawpy instance just to be sure metadata is available here if needed (redundant but safe)
        try:
            with rawpy.imread(raw_filepath) as raw:
                 wb_gains = raw.camera_whitebalance
                 color_matrix_cam_to_xyz = raw.color_matrix # Get color matrix for post-processing
            if DEBUG_MODE: print(f"Debug: Camera WB gains (re-read): {wb_gains}")
            if DEBUG_MODE:
                 if color_matrix_cam_to_xyz is not None:
                      print(f"Debug: Camera Color Matrix shape (re-read): {color_matrix_cam_to_xyz.shape}")
                 else:
                      print("Debug: Camera Color Matrix not available (re-read).")
        except Exception as e:
             if DEBUG_MODE: print(f"Warning: Could not re-read rawpy metadata for post-processing: {e}")
             wb_gains = None # Use None if metadata re-reading fails
             color_matrix_cam_to_xyz = None


        # White Balance Adjustment (applied to output_np)
        output_np_wb = output_np # Fallback if WB fails
        if wb_gains is not None and len(wb_gains) >= 4:
            try:
                r_gain_raw = float(wb_gains[0])
                g1_gain_raw = float(wb_gains[1])
                b_gain_raw = float(wb_gains[2])
                g2_gain_raw = float(wb_gains[3])

                # Calculate average green gain, ensuring no division by zero or invalid values
                valid_green_gains = [g for g in [g1_gain_raw, g2_gain_raw] if g > 0]
                if valid_green_gains:
                     avg_g_gain_raw = sum(valid_green_gains) / len(valid_green_gains)
                else:
                     avg_g_gain_raw = 0 # Indicate invalid green gains


                if avg_g_gain_raw > 1e-6:
                    r_gain_applied = r_gain_raw / avg_g_gain_raw
                    g_gain_applied = 1.0 # Green is the reference
                    b_gain_applied = b_gain_raw / avg_g_gain_raw

                    if DEBUG_MODE: print(f"Debug: Applying Camera WB Gains (R, G, B): [{r_gain_applied:.4f}, {g_gain_applied:.4f}, {b_gain_applied:.4f}]")
                    output_np_wb = np.copy(output_np)
                    output_np_wb[:, :, 0] *= r_gain_applied
                    output_np_wb[:, :, 1] *= g_gain_applied
                    output_np_wb[:, :, 2] *= b_gain_applied
                    output_np_wb = np.maximum(output_np_wb, 0)
                    if DEBUG_MODE: print("Debug: White balance applied.")
                elif DEBUG_MODE:
                    print("Warning: Average green WB gain is zero or too small. Skipping white balance.")
            except Exception as e: # Catch any other errors during WB application
                 if DEBUG_MODE: print(f"Warning: Error applying white balance: {e}. Skipping white balance.")


        # Color Matrix Application (applied to output_np_wb)
        output_np_after_color_step = output_np_wb # Fallback if Color Matrix fails
        # Re-implement color matrix application here similar to genzou.py
        if color_matrix_cam_to_xyz is not None and color_matrix_cam_to_xyz.shape == (3, 3):
             if DEBUG_MODE: print("Debug: Applying Camera Color Matrix (Camera RGB -> CIE XYZ -> Linear sRGB).")
             # Standard CIE XYZ -> Linear sRGB conversion matrix (D65 white point, 2 degree observer)
             xyz_to_srgb_matrix = np.array([
                 [ 3.2404562, -1.5371385, -0.4985314],
                 [-0.9692660,  1.8760108,  0.0415560],
                 [ 0.0556434, -0.2040259,  1.0572252]
             ], dtype=np.float32)

             h_wb, w_wb, c_wb = output_np_wb.shape
             reshaped_rgb = output_np_wb.reshape(-1, 3) # (Height * Width, 3)

             try:
                 # Camera RGB -> CIE XYZ
                 # V_xyz = M_cam_to_xyz * V_cam (vector V_cam = [R, G, B].T)
                 # numpy: (N, 3) @ (3, 3).T -> (N, 3)
                 xyz_linear = np.dot(reshaped_rgb, color_matrix_cam_to_xyz.T)
                 # CIE XYZ -> Linear sRGB
                 # V_srgb = M_xyz_to_srgb * V_xyz
                 srgb_linear = np.dot(xyz_linear, xyz_to_srgb_matrix.T)

                 output_np_colorspace = srgb_linear.reshape(h_wb, w_wb, 3)
                 output_np_after_color_step = np.maximum(output_np_colorspace, 0) # Ensure non-negative

                 if DEBUG_MODE: print("Debug: Color matrix conversion applied.")

             except Exception as e:
                 if DEBUG_MODE: print(f"Warning: Error applying color matrix transformation: {e}. Skipping color matrix conversion.")
                 # output_np_after_color_step remains output_np_wb


        elif DEBUG_MODE:
             print("Warning: Camera color matrix not available or not 3x3. Skipping color matrix conversion.")
             # output_np_after_color_step remains output_np_wb


        # Saturation Adjustment (applied to output_np_after_color_step)
        output_np_saturated = output_np_after_color_step # Fallback if Saturation fails
        if abs(INFERENCE_SATURATION_FACTOR - 1.0) > 1e-6: # Use global setting
            if DEBUG_MODE: print(f"Debug: Applying saturation factor: {INFERENCE_SATURATION_FACTOR:.4f}")
            # Clip to [0, 1] before luminance calculation (input to saturation should be in valid range)
            clipped_input_for_saturation = np.clip(output_np_after_color_step, 0, 1)
            # Rec.709 Luminance
            luminance = np.dot(clipped_input_for_saturation[..., :3], [0.2126, 0.7152, 0.0722])
            # Ensure luminance is >= 0
            luminance = np.maximum(luminance, 0)
            grayscale_image = np.stack([luminance, luminance, luminance], axis=-1)
            output_np_saturated = grayscale_image + INFERENCE_SATURATION_FACTOR * (clipped_input_for_saturation - grayscale_image) # Use global setting
            output_np_saturated = np.clip(output_np_saturated, 0, 1) # Clip results back to [0, 1]
            if DEBUG_MODE: print("Debug: Saturation adjustment applied.")


        output_np_saturated_clipped = output_np_saturated # Result after saturation (and clipping)

        # Data Range Analysis and Remapping / Clipping (applied to output_np_saturated_clipped)
        # Rely on clipping to 0-1 range
        output_np_clipped_remap = np.clip(output_np_saturated_clipped, 0, 1)
        if DEBUG_MODE: print("Debug: Clipping after saturation and color matrix steps.")


        # Contrast Adjustment (applied to output_np_clipped_remap)
        if abs(INFERENCE_CONTRAST_FACTOR - 1.0) > 1e-6: # Use global setting
             if DEBUG_MODE: print(f"Debug: Applying contrast factor: {INFERENCE_CONTRAST_FACTOR:.4f}")
             # Adjust contrast around the midpoint 0.5
             output_np_contrast = 0.5 + INFERENCE_CONTRAST_FACTOR * (output_np_clipped_remap - 0.5) # Use global setting
             output_np_clipped = np.clip(output_np_contrast, 0, 1) # Clip result back to [0, 1]
             if DEBUG_MODE: print("Debug: Contrast adjustment applied.")
        else:
             if DEBUG_MODE: print("Debug: Contrast factor is 1.0, skipping contrast adjustment.")
             output_np_clipped = output_np_clipped_remap # No contrast adjustment


        # Gamma Correction (applied to output_np_clipped)
        # The 'gamma' variable here is DISPLAY_GAMMA, used to prepare the image for uint8 conversion
        # This means the output uint8 image is gamma-corrected.
        gamma_correction_val = DISPLAY_GAMMA # Use global setting for display gamma
        if abs(gamma_correction_val - 1.0) > 1e-6:
            if DEBUG_MODE: print(f"Debug: Applying gamma correction: {gamma_correction_val}")
            # Apply gamma correction to the 0-1 linear data
            output_np_gamma_corrected = np.power(output_np_clipped, 1.0 / gamma_correction_val)
            output_np_gamma_corrected = np.clip(output_np_gamma_corrected, 0.0, 1.0) # Clip result back to [0, 1]
            if DEBUG_MODE: print("Debug: Gamma correction applied.")
        else:
            if DEBUG_MODE: print("Debug: Gamma is 1.0, skipping gamma correction.")
            output_np_gamma_corrected = output_np_clipped # No gamma correction


        # Convert float32 (0-1) to 0-255 uint8
        # This conversion inherently clips to 0-255, but the previous clips ensure the range is 0-1
        output_np_final = (output_np_gamma_corrected * 255.0).astype(np.uint8)

        # Ensure the final output is exactly the original size (raw_h, raw_w, 3)
        # This should already be the case due to the cropping before numpy conversion,
        # but a final safety check can be added if needed, though slicing numpy is efficient.
        # The cropping [:raw_h, :raw_w, :] is already done on the numpy array just before returning.

        if DEBUG_MODE: print(f"Debug: Final conversion to uint8. Output shape: {output_np_final.shape}, dtype: {output_np_final.dtype}")


        return output_np_final # Return final result (uint8 HWC)

    except Exception as e:
        print(f"Error processing RAW file {os.path.basename(raw_filepath)} during inference: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return None






# --- PySide6 GUIと学習スレッド ---
def convert_np_to_qpixmap_display(np_image: np.ndarray | None, is_pseudo_bayer: bool = False, gamma: float = 1.0) -> QPixmap:
    """
    numpy array (HWC, float or uint8) を表示用に uint8 に変換し、QPixmapとして返す。
    is_pseudo_bayer=True の場合、1チャンネル入力を擬似カラー表示する。
    gamma > 1.0 の場合、float入力に対してガンマ補正を適用する。
    uint8入力の場合、gammaは通常1.0として扱われ、再度のガンマ補正は行わない。
    """
    if DEBUG_MODE: print(f"Debug: convert_np_to_qpixmap_display called with shape {np_image.shape if np_image is not None else 'None'}, dtype {np_image.dtype if np_image is not None else 'None'}, is_pseudo_bayer={is_pseudo_bayer}, gamma={gamma:.2f}.")

    if np_image is None or np_image.size == 0:
        if DEBUG_MODE: print("Debug: convert_np_to_qpixmap_display received empty numpy array.")
        return QPixmap()

    temp_np_float = None

    if np_image.dtype == np.uint8:
        if DEBUG_MODE: print(f"Debug: Input dtype is uint8. Converting to float32 (0-1).")
        temp_np_float = np_image.astype(np.float32) / 255.0
    elif np_image.dtype == np.uint16:
        if DEBUG_MODE: print(f"Debug: Input dtype is uint16. Converting to float32 (0-1).")
        temp_np_float = np_image.astype(np.float32) / np.iinfo(np.uint16).max
    elif np_image.dtype in [np.float32, np.float64]:
        if DEBUG_MODE: print(f"Debug: Input dtype is {np_image.dtype}. Ensuring float32.")
        temp_np_float = np_image.astype(np.float32)
    else:
        print(f"Warning: convert_np_to_qpixmap_display received unsupported dtype: {np_image.dtype}. Expected uint8, uint16, float32/64.")
        return QPixmap()

    # NaNやinfをゼロに置き換え、0-1の範囲にクリップ (ガンマ補正前)
    temp_np_float = np.nan_to_num(temp_np_float, nan=0.0, posinf=1.0, neginf=0.0)
    temp_np_float = np.clip(temp_np_float, 0.0, 1.0)
    if DEBUG_MODE: print(f"Debug: Clipped and handled NaN/inf. Min: {np.min(temp_np_float):.4f}, Max: {np.max(temp_np_float):.4f}")

    display_np_uint8 = None
    h, w = temp_np_float.shape[:2]

    if is_pseudo_bayer and temp_np_float.ndim == 3 and temp_np_float.shape[2] == 1:
        if DEBUG_MODE: print(f"Debug: Processing 1-channel pseudo-Bayer for display (pseudo-color + gamma).")
        pseudo_color_rgb = np.zeros((h, w, 3), dtype=np.float32)
        display_np_float_1ch = temp_np_float[:, :, 0]

        pseudo_color_rgb[0::2, 0::2, 0] = display_np_float_1ch[0::2, 0::2]
        pseudo_color_rgb[0::2, 1::2, 1] = display_np_float_1ch[0::2, 1::2]
        pseudo_color_rgb[1::2, 0::2, 1] = display_np_float_1ch[1::2, 0::2]
        pseudo_color_rgb[1::2, 1::2, 2] = display_np_float_1ch[1::2, 1::2]

        processed_float_data = pseudo_color_rgb
    elif temp_np_float.ndim == 3 and temp_np_float.shape[2] == 3:
        if DEBUG_MODE: print(f"Debug: Processing 3-channel RGB for display (gamma).")
        processed_float_data = temp_np_float
    elif temp_np_float.ndim == 2:
        if DEBUG_MODE: print("Debug: Processing 2D grayscale for display (gamma).")
        processed_float_data = temp_np_float # Will be converted to Grayscale8
    elif temp_np_float.ndim == 3 and temp_np_float.shape[2] == 4:
        if DEBUG_MODE: print(f"Debug: Received 4-channel data. Displaying first 3 as RGB (gamma).")
        processed_float_data = temp_np_float[:, :, :3]
    else:
        print(f"Warning: convert_np_to_qpixmap_display received unsupported numpy array shape: {temp_np_float.shape}.")
        return QPixmap()

    # Apply gamma correction if gamma is not 1.0
    # Note: If original input was uint8, 'gamma' passed to this function should typically be 1.0.
    if abs(gamma - 1.0) > 1e-6:
        data_after_gamma = np.power(processed_float_data, 1.0 / gamma)
        data_after_gamma = np.clip(data_after_gamma, 0.0, 1.0)
        if DEBUG_MODE: print(f"Debug: Applied gamma correction ({gamma:.2f}). Clipped.")
    else:
        data_after_gamma = processed_float_data # No gamma correction needed

    # Convert float32 (0-1 range) to uint8 (0-255 range)
    display_np_uint8 = (data_after_gamma * 255.0).astype(np.uint8)
    if DEBUG_MODE: print(f"Debug: Converted to uint8. Shape: {display_np_uint8.shape}")

    display_np_contiguous = np.ascontiguousarray(display_np_uint8)
    img = QImage()

    if display_np_contiguous.ndim == 3 and display_np_contiguous.shape[2] == 3:
        h_img, w_img, c_img = display_np_contiguous.shape
        bytes_per_line = w_img * c_img
        img = QImage(display_np_contiguous.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)
        if DEBUG_MODE: print(f"Debug: Created QImage Format_RGB888.")
    elif display_np_contiguous.ndim == 2: # Grayscale
        h_img, w_img = display_np_contiguous.shape
        bytes_per_line = w_img
        img = QImage(display_np_contiguous.data, w_img, h_img, bytes_per_line, QImage.Format_Grayscale8)
        if DEBUG_MODE: print(f"Debug: Created QImage Format_Grayscale8.")
    else:
        print(f"Error: Final uint8 numpy array has unsupported shape for QImage: {display_np_contiguous.shape}.")
        return QPixmap()

    if img.isNull():
        if DEBUG_MODE: print("Error: QImage creation failed.")
        return QPixmap()

    return QPixmap.fromImage(img)



class SquareImageLabel(QLabel):
    """
    QLabelを継承し、表示される画像を常に正方形領域にアスペクト比を保って描画するカスタムウィジェット
    ピクセルがぼやけないようにニアレストネイバー法でスケーリングする。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setScaledContents(False) # QLabelの自動スケーリングを無効化
        self.setAlignment(Qt.AlignCenter) # 描画位置を中央寄せ
        self._pixmap: QPixmap = QPixmap() # 表示するQPixmapを保持

        # サイズポリシーを設定
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        # 最小サイズを設定 (GUIのレイアウトを考慮)
        self.setMinimumSize(150, 150)


    def setPixmap(self, pixmap: QPixmap | None):
        """
        表示するQPixmapを設定

        Args:
            pixmap: 設定するQPixmapオブジェクト、またはNone
        """
        if isinstance(pixmap, QPixmap):
            self._pixmap = pixmap
        elif pixmap is None:
            self._pixmap = QPixmap() # Clear the pixmap
        else:
            if DEBUG_MODE: print(f"Warning: SquareImageLabel.setPixmap received invalid object type: {type(pixmap)}. Expected QPixmap or None.")
            self._pixmap = QPixmap() # Invalid input, clear pixmap

        self.update() # paintEventをトリガーして再描画

    def pixmap(self) -> QPixmap:
        """
        現在設定されているQPixmapを取得
        """
        return self._pixmap

    def paintEvent(self, event):
        """ウィジェットの描画イベント"""
        painter = QPainter(self)
        # スムーズなスケーリングではなく、ピクセルをシャープに保つため、
        # SmoothPixmapTransformは設定しない（または明示的に無効化）
        # painter.setRenderHint(QPainter.SmoothPixmapTransform, False) # これでも良いが、デフォルトで無効なことが多い

        if not self._pixmap.isNull():
            # ウィジェットのサイズ内で、描画可能な最大のサイズを計算
            # アスペクト比を保って表示するため、短い辺に合わせる
            widget_width = self.width()
            widget_height = self.height()
            pixmap_width = self._pixmap.width()
            pixmap_height = self._pixmap.height()

            if pixmap_width <= 0 or pixmap_height <= 0 or widget_width <= 0 or widget_height <= 0:
                 super().paintEvent(event) # 描画サイズが不正ならデフォルト描画
                 return

            # ウィジェット内にアスペクト比を保って収まるようにスケーリングサイズを計算
            scale_factor = min(widget_width / pixmap_width, widget_height / pixmap_height)
            scaled_w = int(pixmap_width * scale_factor)
            scaled_h = int(pixmap_height * scale_factor)

            if scaled_w <= 0 or scaled_h <= 0:
                 super().paintEvent(event) # スケーリングサイズが不正ならデフォルト描画
                 return

            # 元のPixmapを計算したサイズに、アスペクト比を保ってスケーリング
            # Qt.FastTransformation を使用してピクセルのぼやけを防ぐ (ニアレストネイバー法に近い)
            scaled_pixmap = self._pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.FastTransformation)


            # スケーリングされた画像をウィジェットの中央に描画するための位置を計算
            x = (widget_width - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2

            # 画像を描画
            painter.drawPixmap(x, y, scaled_pixmap)
        else:
            # Pixmapが設定されていない場合、基底クラスのpaintEventを呼び出す
            # これにより背景などが適切に描画される
            super().paintEvent(event)



class TrainingWorker(QThread):
    """
    学習処理を行うスレッド
    """
    # シグナル定義: (current_epoch, total_epochs, current_epoch_avg_loss, total_patches_in_epoch, processed_patches_in_current_epoch)
    # total_patches_in_epoch: 現在のエポックで処理するパッチ総数
    # processed_patches_in_current_epoch: 現在のエポックで処理済みのパッチ数
    progress_updated = Signal(int, int, float, int, int, float)

    # シグナル定義: (label_key, image_data_np, is_pseudo_bayer) - image_data_np は numpy array (float32 or uint8), is_pseudo_bayerはbool
    image_updated = Signal(str, np.ndarray, bool)

    # シグナル定義: ()
    training_finished = Signal()

    # シグナル定義: (message)
    error_occurred = Signal(str)


    def __init__(self, input_data_dir: str, target_data_dir: str, raw_paths: list[str],
                 model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                 device: torch.device, patch_size: int, scheduler):
        super().__init__()
        self.input_data_dir = input_data_dir
        self.target_data_dir = target_data_dir
        self.raw_paths = raw_paths
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patch_size = patch_size
        self.scheduler = scheduler
        self._is_running = True # スレッド停止フラグ

        # 学習データのパスリストを取得
        # input_files は 128x128x1 の疑似ベイヤーデータ (.npy)
        self.input_files = sorted(glob.glob(os.path.join(self.input_data_dir, "*.npy")))
        # 対応する教師データのパスリストを取得 (ファイル名をベースに検索)
        self.training_data_pairs = []
        for input_file in self.input_files:
            base_name = os.path.basename(input_file)
            target_file = os.path.join(self.target_data_dir, base_name) # 教師データも同じファイル名
            if os.path.exists(target_file):
                self.training_data_pairs.append((input_file, target_file))
            else:
                print(f"Warning: Corresponding target file not found for {input_file}. Skipping pair.")
                if DEBUG_MODE: print(f"Debug: Missing target file: {target_file}")

        if not self.training_data_pairs:
             # MainWindow側でこのケースをチェックし、ユーザーに通知する
             print(f"Info: No matching .npy data pairs found in '{self.input_data_dir}' and '{self.target_data_dir}'.")
             self.total_patches_in_each_epoch = 0
        else:
            # パッチ総数はデータペアの数
            self.total_patches_in_each_epoch = len(self.training_data_pairs)
            print(f"Found {self.total_patches_in_each_epoch} training data pairs.")


    def stop(self):
        """スレッドの停止を要求する"""
        self._is_running = False
        print("Training stop requested...")









    def run(self):
        """スレッドのメイン実行ループ"""
        if self.total_patches_in_each_epoch == 0:
             print("No training data available. Training skipped.")
             self.training_finished.emit()
             return

        self.model.train()
        total_epochs = NUM_EPOCHS
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(DEVELOPED_DIR, exist_ok=True)
        display_raw_path = self.raw_paths[0] if self.raw_paths else None
        if display_raw_path:
            print(f"Using {os.path.basename(display_raw_path)} for Real RAW Demosaic display and saving.")
        else:
            print("No RAW file found for Real RAW Demosaic display and saving.")
        print(f"Total patches per epoch: {self.total_patches_in_each_epoch}")

        for epoch in range(1, total_epochs + 1):
            if not self._is_running:
                print(f"Epoch {epoch}/{total_epochs} finished. Training stopped by request.")
                break

            epoch_loss = 0.0
            processed_patches_in_current_epoch = 0
            np.random.shuffle(self.training_data_pairs)
            print(f"Starting Epoch {epoch}/{total_epochs}...")

            # エポック開始時の進捗バーをリセット
            current_lr = self.optimizer.param_groups[0]['lr'] # <<< 学習率を取得
            self.progress_updated.emit(epoch, total_epochs, 0.0, self.total_patches_in_each_epoch, processed_patches_in_current_epoch, current_lr) # <<< current_lr を渡す

            for input_file, target_file in self.training_data_pairs:
                if not self._is_running:
                    print(f"Epoch {epoch}: Patch processing loop stopped by request.")
                    break
                processed_patches_in_current_epoch += 1
                try:
                    # (データロードとテンソル変換処理はそのまま)
                    input_data_np_1ch = np.load(input_file)
                    target_data_np = np.load(target_file)
                    h, w, _ = input_data_np_1ch.shape
                    bayer_input_tensor_4ch = torch.zeros(1, 4, h, w, dtype=torch.float32)
                    bayer_input_tensor_4ch[0, 0, 0::2, 0::2] = torch.from_numpy(input_data_np_1ch[0::2, 0::2, 0])
                    bayer_input_tensor_4ch[0, 1, 0::2, 1::2] = torch.from_numpy(input_data_np_1ch[0::2, 1::2, 0])
                    bayer_input_tensor_4ch[0, 2, 1::2, 0::2] = torch.from_numpy(input_data_np_1ch[1::2, 0::2, 0])
                    bayer_input_tensor_4ch[0, 3, 1::2, 1::2] = torch.from_numpy(input_data_np_1ch[1::2, 1::2, 0])
                    bayer_input_tensor_4ch = bayer_input_tensor_4ch.to(self.device)
                    target_output_tensor = torch.from_numpy(target_data_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
                except Exception as e:
                    print(f"Error loading or processing data pair ({input_file}, {target_file}): {e}. Skipping patch.")
                    if DEBUG_MODE: traceback.print_exc()
                    self.error_occurred.emit(f"Error processing data pair {os.path.basename(input_file)}: {e}")
                    continue

                self.optimizer.zero_grad()
                outputs = self.model(bayer_input_tensor_4ch)
                loss = self.criterion(outputs, target_output_tensor)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                # (モデル保存処理はそのまま)
                is_model_save_interval = (processed_patches_in_current_epoch % SAVE_MODEL_PATCHES_PROCESSED_INTERVAL == 0)
                is_end_of_epoch_patches = (processed_patches_in_current_epoch == self.total_patches_in_each_epoch)
                if is_model_save_interval or is_end_of_epoch_patches:
                    # (モデル保存のコード)
                    print(f"Saving model at epoch {epoch}, patch {processed_patches_in_current_epoch} / {self.total_patches_in_each_epoch}...")
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"{timestamp_str}_demosaic_model_epoch_{epoch}_patch_{processed_patches_in_current_epoch}.pth"
                    model_save_path = os.path.join(MODEL_DIR, model_filename)
                    try:
                        torch.save(self.model.state_dict(), model_save_path)
                        print(f"Model saved to {model_save_path}")
                    except Exception as e:
                        print(f"Error saving model at epoch {epoch}, patch {processed_patches_in_current_epoch}: {e}")
                        if DEBUG_MODE: traceback.print_exc()
                        self.error_occurred.emit(f"Error saving model: {e}")


                is_training_patch_update_interval = (processed_patches_in_current_epoch % DISPLAY_TRAINING_PATCHES_PROCESSED_INTERVAL == 0)
                is_real_raw_update_interval = (processed_patches_in_current_epoch % DISPLAY_REAL_RAW_PATCHES_PROCESSED_INTERVAL == 0)
                is_last_patch_of_epoch = (processed_patches_in_current_epoch == self.total_patches_in_each_epoch)

                if is_training_patch_update_interval or is_last_patch_of_epoch:
                    avg_epoch_loss = epoch_loss / processed_patches_in_current_epoch if processed_patches_in_current_epoch > 0 else 0.0
                    current_lr = self.optimizer.param_groups[0]['lr'] # <<< 学習率を取得
                    self.progress_updated.emit(epoch, total_epochs, avg_epoch_loss, self.total_patches_in_each_epoch, processed_patches_in_current_epoch, current_lr) # <<< current_lr を渡す

                    # (画像更新シグナル発行部分はそのまま)
                    self.image_updated.emit("Loaded Target RGB (Linear)", target_data_np, False)
                    self.image_updated.emit("Loaded Input RGBG (Linear Pseudo-color)", input_data_np_1ch, True)
                    output_display_np = outputs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    self.image_updated.emit("Model Prediction (Training Patch)", output_display_np, False)


                if (is_real_raw_update_interval or is_last_patch_of_epoch) and display_raw_path:
                     # (実際のRAWデモザイク処理と保存、GUI更新部分はそのまま)
                     print(f"Updating Real RAW Demosaic display for epoch {epoch}, patch {processed_patches_in_current_epoch} / {self.total_patches_in_each_epoch} and saving...")
                     raw_result_np = process_real_raw(display_raw_path, self.model, self.device)
                     if raw_result_np is not None:
                         self.image_updated.emit("Real RAW Demosaic (Post-processed)", raw_result_np, False)
                         try:
                             os.makedirs(DEVELOPED_DIR, exist_ok=True)
                             timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                             raw_file_base = os.path.splitext(os.path.basename(display_raw_path))[0]
                             save_filename = f"{timestamp_str}_epoch{epoch}_patch{processed_patches_in_current_epoch}_{raw_file_base}_developed.png"
                             save_path = os.path.join(DEVELOPED_DIR, save_filename)
                             img_to_save = Image.fromarray(raw_result_np, 'RGB')
                             img_to_save.save(save_path)
                             print(f"Saved developed image to {save_path}")
                         except Exception as e:
                             print(f"Error saving developed image for epoch {epoch}, patch {processed_patches_in_current_epoch}: {e}")
                             if DEBUG_MODE: traceback.print_exc()
                             self.error_occurred.emit(f"Error saving developed image: {e}")
                     else:
                         print(f"Failed to process Real RAW file {os.path.basename(display_raw_path)} for display/saving.")
                         self.error_occurred.emit(f"Failed to process Real RAW file {os.path.basename(display_raw_path)} for inference.")


            avg_loss = epoch_loss / processed_patches_in_current_epoch if processed_patches_in_current_epoch > 0 else 0.0
            print(f"Epoch {epoch}/{total_epochs} finished. Average Loss: {avg_loss:.4f}")

            # StepLRはエポックの終わりに引数なしでstep()を呼び出す
            self.scheduler.step() # ReduceLROnPlateauでは avg_loss を渡していたが、StepLRでは不要
            current_lr = self.optimizer.param_groups[0]['lr'] # <<< 学習率をスケジューラ更新後に再度取得
            # ↓↓↓ スケジューラ更新後のLRも表示するため、学習率を取得してemitする
            print(f"Current learning rate after scheduler step: {current_lr:.2e}") # コンソールにも表示
            self.progress_updated.emit(epoch, total_epochs, avg_loss, self.total_patches_in_each_epoch, processed_patches_in_current_epoch, current_lr) # <<< current_lr を渡す

            if not self._is_running:
                 print(f"Epoch {epoch}: Stopped after display/save.")
                 break

        print("Training loop finished.")
        self.training_finished.emit()








# ImageNetの平均・標準偏差（VGGなどの事前学習モデルの入力正規化に使用）
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class VGGPerceptualLoss(nn.Module):
    """
    事前学習済みVGG19モデルを利用した知覚損失 (Perceptual Loss)
    """
    def __init__(self, requires_grad=False, layer_weights=None, device='cpu'):
        super(VGGPerceptualLoss, self).__init__()
        # VGG19のfeatures部分をロード (勾配計算はオフ)
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_layers = vgg19.eval()
        # 勾配計算を無効化
        for param in self.vgg_layers.parameters():
            param.requires_grad = requires_grad

        # 特徴を抽出するVGGの層インデックス (ReLUの直後)
        self.layer_indices = [2, 7, 12, 21, 30] # Corresponding to relu1_1, relu2_1, relu3_1, relu4_1, relu5_1

        # 各層の特徴マップに対する損失の重み
        if layer_weights is None:
            self.layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            if len(layer_weights) != len(self.layer_indices):
                raise ValueError("Number of layer_weights must match number of layer_indices")
            self.layer_weights = layer_weights

        self.criterion = nn.L1Loss() # 特徴マップ間のロスはL1が一般的
        self.device = device
        self.vgg_layers.to(device)
        self.mean = IMAGENET_MEAN.clone().to(device)
        self.std = IMAGENET_STD.clone().to(device)

    def normalize_input(self, x):
        """ImageNetの統計情報で入力画像を正規化"""
        return (x - self.mean) / self.std


    def forward(self, input_img, target_img):
        """
        入力画像とターゲット画像の特徴マップ間の損失を計算
        input_img, target_img は [B, 3, H, W] float32 を想定
        """
        normalized_input = self.normalize_input(input_img)
        normalized_target = self.normalize_input(target_img)

        input_features = []
        target_features = []
        current_input = normalized_input
        current_target = normalized_target

        i = 0
        for layer in self.vgg_layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                 current_input = layer(current_input)
                 current_target = layer(current_target)
            elif isinstance(layer, nn.ReLU):
                 current_input = F.relu(current_input, inplace=False)
                 current_target = F.relu(current_target, inplace=False)
            else:
                 current_input = layer(current_input)
                 current_target = layer(current_target)

            if i in self.layer_indices:
                input_features.append(current_input)
                target_features.append(current_target)
            i += 1

        loss = 0
        for j in range(len(self.layer_indices)):
             loss += self.layer_weights[j] * self.criterion(input_features[j], target_features[j].detach())
        return loss

class GradientLoss(nn.Module):
    """
    画像の勾配に対する損失 (Gradient Loss)
    エッジやディテールの再現に寄与
    """
    def __init__(self, device='cpu'):
        super(GradientLoss, self).__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0., 0., 0.],
                                [ 1., 2., 1.]])

        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)

        self.sobel_x.requires_grad_(False)
        self.sobel_y.requires_grad_(False)

        self.criterion = nn.L1Loss()

    def forward(self, img):
        """
        画像の勾配を計算し、その値を返す（Helperとして使用）
        img は [B, 3, H, W] float32 を想定
        """
        padded_img = F.pad(img, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(padded_img, self.sobel_x, groups=3)
        grad_y = F.conv2d(padded_img, self.sobel_y, groups=3)
        return grad_x, grad_y

class ChromaLoss(nn.Module):
    """
    画像のクロマ (色差) 成分に対する損失 (Chroma Loss)
    RGB画像をYCbCrに変換し、CbCr成分のL1損失を計算
    """
    def __init__(self, device='cpu'):
        super(ChromaLoss, self).__init__()
        self.device = device
        # RGB to YCbCr (BT.709) conversion matrix components
        # Y  =  0.2126 * R + 0.7152 * G + 0.0722 * B
        # Cb = -0.114572 * R - 0.385428 * G + 0.500000 * B (scaled to [-0.5, 0.5])
        # Cr =  0.500000 * R - 0.454153 * G - 0.045847 * B (scaled to [-0.5, 0.5])
        # Matrix for direct application [Y, Cb, Cr].T = Matrix @ [R, G, B].T
        # We will apply this by direct multiplication for clarity with [B,C,H,W] tensors
        self.r_coeffs = torch.tensor([0.2126, -0.114572, 0.500000], device=self.device).view(3, 1, 1)
        self.g_coeffs = torch.tensor([0.7152, -0.385428, -0.454153], device=self.device).view(3, 1, 1)
        self.b_coeffs = torch.tensor([0.0722, 0.500000, -0.045847], device=self.device).view(3, 1, 1)

        self.r_coeffs.requires_grad_(False)
        self.g_coeffs.requires_grad_(False)
        self.b_coeffs.requires_grad_(False)

        self.criterion = nn.L1Loss()

    def rgb_to_ycbcr(self, rgb_image):
        """
        Converts an RGB image tensor to YCbCr components.
        rgb_image: [B, 3, H, W] tensor, values in [0, 1]
        Returns:
            y_channel: [B, 1, H, W]
            cb_channel_shifted: [B, 1, H, W] (Cb values shifted from [-0.5, 0.5] to [0, 1])
            cr_channel_shifted: [B, 1, H, W] (Cr values shifted from [-0.5, 0.5] to [0, 1])
        """
        r_comp = rgb_image[:, 0:1, :, :]
        g_comp = rgb_image[:, 1:2, :, :]
        b_comp = rgb_image[:, 2:3, :, :]

        y_channel  = self.r_coeffs[0] * r_comp + self.g_coeffs[0] * g_comp + self.b_coeffs[0] * b_comp
        cb_channel = self.r_coeffs[1] * r_comp + self.g_coeffs[1] * g_comp + self.b_coeffs[1] * b_comp
        cr_channel = self.r_coeffs[2] * r_comp + self.g_coeffs[2] * g_comp + self.b_coeffs[2] * b_comp
        
        # Shift Cb, Cr from [-0.5, 0.5] to [0, 1] for L1 loss consistency
        cb_channel_shifted = cb_channel + 0.5
        cr_channel_shifted = cr_channel + 0.5
        
        return y_channel, cb_channel_shifted, cr_channel_shifted

    def forward(self, outputs_rgb, targets_rgb):
        """
        Calculates L1 loss on Cb and Cr channels.
        outputs_rgb, targets_rgb: [B, 3, H, W] tensors, values in [0, 1]
        """
        _, cb_outputs, cr_outputs = self.rgb_to_ycbcr(outputs_rgb)
        _, cb_targets, cr_targets = self.rgb_to_ycbcr(targets_rgb.detach()) # detach targets

        loss_cb = self.criterion(cb_outputs, cb_targets)
        loss_cr = self.criterion(cr_outputs, cr_targets)

        return (loss_cb + loss_cr) / 2.0

class LocalColorConsistencyLoss(nn.Module):
    """
    局所的な色の一貫性に関する損失 (Local Color Consistency Loss)
    モデル出力とターゲット画像の各パッチにおける色の分散を比較する。
    """
    def __init__(self, patch_size: int, stride: int, device='cpu'):
        super(LocalColorConsistencyLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.device = device
        # unfoldを使ってパッチを効率的に抽出
        self.unfolder = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.stride)
        self.criterion = nn.L1Loss() # 分散間の差はL1損失で評価

        if DEBUG_MODE:
            print(f"Debug: LocalColorConsistencyLoss initialized with patch_size={self.patch_size}, stride={self.stride}")

    def _calculate_patch_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソル x (B, C, H, W) からパッチを抽出し、各パッチ・各チャンネルの分散を計算する。
        Returns:
            Tensor of shape (B, C, NumPatches) containing variances.
        """
        # 1. パッチ抽出
        # x: (B, C, H, W)
        # patches: (B, C * patch_size * patch_size, NumPatches)
        patches = self.unfolder(x)

        # 2. パッチの形状を整える
        # (B, C, patch_size * patch_size, NumPatches)
        # これにより、チャンネルごとにパッチ内のピクセルを扱える
        patches = patches.view(x.size(0), x.size(1), self.patch_size * self.patch_size, -1)

        # 3. 各パッチ・各チャンネルの分散を計算
        # dim=2 は patch_size * patch_size の次元 (パッチ内のピクセル)
        # unbiased=True で不偏分散 (N-1で割る)
        variance = torch.var(patches, dim=2, unbiased=True) # Shape: (B, C, NumPatches)
        return variance

    def forward(self, outputs_rgb: torch.Tensor, targets_rgb: torch.Tensor) -> torch.Tensor:
        """
        outputs_rgb, targets_rgb: [B, 3, H, W] tensors, values in [0, 1]
        """
        if outputs_rgb.shape[2] < self.patch_size or outputs_rgb.shape[3] < self.patch_size:
            if DEBUG_MODE:
                print(f"Warning: LocalColorConsistencyLoss - Input H/W ({outputs_rgb.shape[2]}/{outputs_rgb.shape[3]}) "
                      f"is smaller than patch_size ({self.patch_size}). Returning 0 loss.")
            return torch.tensor(0.0, device=self.device)

        outputs_variance = self._calculate_patch_variance(outputs_rgb)
        # ターゲットの分散計算では勾配は不要
        with torch.no_grad():
            targets_variance = self._calculate_patch_variance(targets_rgb.detach())

        # 分散の差に対するL1損失を計算
        loss = self.criterion(outputs_variance, targets_variance)
        return loss







class CombinedLoss(nn.Module):
    """
    L1 Loss, Perceptual Loss, Gradient Loss, Chroma Loss, Local Color Consistency Loss を組み合わせた損失関数
    暗部により大きな重みをつける Weighted Loss に対応
    """
    def __init__(self, l1_weight, perceptual_weight, gradient_weight, chroma_weight,
                 local_color_consistency_weight, # <<< 追加
                 dark_weight_factor,
                 local_consistency_patch_size, local_consistency_stride, # <<< 追加 (LCCLossの初期化用)
                 device='cpu'):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.gradient_weight = gradient_weight
        self.chroma_weight = chroma_weight
        self.local_color_consistency_weight = local_color_consistency_weight # <<< 追加
        self.dark_weight_factor = dark_weight_factor

        # L1 Lossは手動で計算するため、nn.L1Lossインスタンスは不要

        if self.perceptual_weight > 0:
            self.perceptual_criterion = VGGPerceptualLoss(requires_grad=False, device=device).to(device)
        else:
            self.perceptual_criterion = None
            
        if self.gradient_weight > 0:
             self.gradient_helper = GradientLoss(device=device).to(device)
        else:
             self.gradient_helper = None
        
        if self.chroma_weight > 0:
            self.chroma_criterion = ChromaLoss(device=device).to(device)
        else:
            self.chroma_criterion = None

        if self.local_color_consistency_weight > 0:
            self.local_color_consistency_criterion = LocalColorConsistencyLoss(
                patch_size=local_consistency_patch_size,
                stride=local_consistency_stride,
                device=device
            ).to(device)
        else:
            self.local_color_consistency_criterion = None

    def forward(self, outputs, targets):
        """
        予測結果とターゲット画像に対する総合的な損失を計算
        outputs, targets は [B, 3, H, W] float32 を想定
        """
        total_loss = 0

        # ターゲット画像から輝度（またはチャンネル平均）を計算 [B, 1, H, W]
        # targets は [0, 1] のリニアデータであることを想定
        luminance = torch.mean(targets, dim=1, keepdim=True)

        # 輝度に基づいて重みマップを計算
        # 暗いピクセルほど重みが大きくなるようにする (輝度0で最大重み、輝度1で最小重み1.0)
        weight_map = (1.0 + self.dark_weight_factor * (1.0 - luminance)).expand_as(outputs) # outputs/targetsと同じ形状にexpand

        # 1. Weighted L1 Loss
        if self.l1_weight > 0:
            # ピクセルごとの絶対誤差を計算 [B, 3, H, W]
            l1_error = torch.abs(outputs - targets)

            # 絶対誤差に重みマップを適用
            weighted_l1_error = l1_error * weight_map

            # 重み付けされた誤差の平均を計算
            weighted_l1_loss = torch.mean(weighted_l1_error)

            total_loss += self.l1_weight * weighted_l1_loss
            if DEBUG_MODE: print(f"Debug: Weighted L1 Loss: {weighted_l1_loss.item():.6f}")

        # 2. Perceptual Loss (重みなし、または内部で重み付けされている場合はそれに従う)
        if self.perceptual_weight > 0 and self.perceptual_criterion is not None:
            # Perceptual Loss は [B, 3, H, W] に対して計算
            perceptual_loss_val = self.perceptual_criterion(outputs, targets)
            total_loss += self.perceptual_weight * perceptual_loss_val
            if DEBUG_MODE: print(f"Debug: Perceptual Loss: {perceptual_loss_val.item():.6f}")

        # 3. Weighted Gradient Loss
        if self.gradient_weight > 0 and self.gradient_helper is not None:
            # 予測結果とターゲット画像の勾配を計算 [B, 3, H', W'] (x方向, y方向それぞれ)
            outputs_grad_x, outputs_grad_y = self.gradient_helper(outputs)
            targets_grad_x, targets_grad_y = self.gradient_helper(targets)

            # 勾配間の絶対誤差を計算
            grad_error_x = torch.abs(outputs_grad_x - targets_grad_x.detach())
            grad_error_y = torch.abs(outputs_grad_y - targets_grad_y.detach())

            # L1 Lossと同様の重みマップを適用 (GradientLossの出力サイズに合わせる)
            # GradientLossはpadding='reflect', kernel=3, stride=1 なので、出力サイズ H', W' は H, W と同じ
            weighted_grad_error_x = grad_error_x * weight_map
            weighted_grad_error_y = grad_error_y * weight_map

            # 重み付けされた勾配誤差の平均を計算し合計
            weighted_grad_loss_x = torch.mean(weighted_grad_error_x)
            weighted_grad_loss_y = torch.mean(weighted_grad_error_y)

            weighted_gradient_loss_val = weighted_grad_loss_x + weighted_grad_loss_y

            total_loss += self.gradient_weight * weighted_gradient_loss_val
            if DEBUG_MODE: print(f"Debug: Weighted Gradient Loss: {weighted_gradient_loss_val.item():.6f}")
        
        # 4. Chroma Loss
        if self.chroma_weight > 0 and self.chroma_criterion is not None:
            chroma_loss_val = self.chroma_criterion(outputs, targets)
            total_loss += self.chroma_weight * chroma_loss_val
            if DEBUG_MODE: print(f"Debug: Chroma Loss: {chroma_loss_val.item():.6f}")

        # 5. Local Color Consistency Loss
        # この損失には現状、暗部強調の重み付け(weight_map_for_l1_grad)は適用していません。
        # 適用する場合は、LocalColorConsistencyLossクラス内でパッチごとの輝度に応じた重み付けを考慮する必要があります。
        if self.local_color_consistency_weight > 0 and self.local_color_consistency_criterion is not None:
            local_consistency_loss_val = self.local_color_consistency_criterion(outputs, targets)
            total_loss += self.local_color_consistency_weight * local_consistency_loss_val
            if DEBUG_MODE: print(f"Debug: Local Color Consistency Loss: {local_consistency_loss_val.item():.6f}")

        return total_loss








class MainWindow(QMainWindow):
    """
    デモザイク学習のGUIメインウィンドウ
    """
    @Slot(str, np.ndarray, bool)
    def update_image_display(self, label_key: str, image_data_np: np.ndarray, is_pseudo_bayer: bool):
        """
        学習スレッドから送られてきた画像をGUIに表示。
        convert_np_to_qpixmap_display を使用して QPixmap を生成し、適切なラベルに設定する。
        Args:
            label_key: 画像の種類を示す文字列 (どのラベルに表示するかを決定)
            image_data_np: 表示する numpy 配列 (float32 or uint8)
            is_pseudo_bayer: 疑似ベイヤーデータの場合はTrue
        """
        if DEBUG_MODE: 
            print(f"Debug: MainWindow.update_image_display received for '{label_key}' "
                  f"with shape {image_data_np.shape if image_data_np is not None else 'None'}, "
                  f"dtype {image_data_np.dtype if image_data_np is not None else 'None'}, "
                  f"is_pseudo_bayer={is_pseudo_bayer}.")

        if image_data_np is None:
            if DEBUG_MODE: print(f"Debug: image_data_np is None for '{label_key}'. Clearing pixmap.")
            # Clear the specific pixmap
            if label_key == "Loaded Target RGB (Linear)": self.image_original.setPixmap(None)
            elif label_key == "Loaded Input RGBG (Linear Pseudo-color)": self.image_simulated_bayer.setPixmap(None)
            elif label_key == "Model Prediction (Training Patch)": self.image_model_prediction.setPixmap(None)
            elif label_key == "Real RAW Demosaic (Post-processed)": self.image_real_raw_demosaic.setPixmap(None)
            return

        # Determine the gamma value to pass to the conversion function
        gamma_for_display = DISPLAY_GAMMA  # Default for linear float data

        if is_pseudo_bayer:
            # Pseudo-bayer input is linear, apply display gamma for visualization
            gamma_for_display = DISPLAY_GAMMA
        elif image_data_np.dtype == np.uint8:
            # If input is uint8 (e.g., from process_real_raw, which is already gamma-corrected),
            # pass gamma=1.0 to avoid re-applying gamma.
            gamma_for_display = 1.0
            if DEBUG_MODE: print(f"Debug: Input for '{label_key}' is uint8, setting gamma_for_display=1.0 for conversion.")
        
        # Call the centralized conversion function
        pixmap = convert_np_to_qpixmap_display(image_data_np, is_pseudo_bayer, gamma_for_display)

        if pixmap.isNull():
            if DEBUG_MODE: print(f"Warning: Failed to create pixmap for '{label_key}'.")
            # Optionally, clear the label if pixmap creation fails
            if label_key == "Loaded Target RGB (Linear)": self.image_original.setPixmap(None)
            # ... similar for other labels
            return

        # Set the pixmap to the appropriate image label based on label_key
        if label_key == "Loaded Target RGB (Linear)":
            self.image_original.setPixmap(pixmap)
        elif label_key == "Loaded Input RGBG (Linear Pseudo-color)":
            self.image_simulated_bayer.setPixmap(pixmap)
        elif label_key == "Model Prediction (Training Patch)":
            self.image_model_prediction.setPixmap(pixmap)
        elif label_key == "Real RAW Demosaic (Post-processed)":
            self.image_real_raw_demosaic.setPixmap(pixmap)
        else:
             if DEBUG_MODE: print(f"Warning: No matching image label found in MainWindow for '{label_key}'.")


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demosaic Training GUI")
        self.setGeometry(100, 100, 1200, 400) # ウィンドウサイズ調整

        # --- GUIコンポーネント ---
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setFormat("Initializing...") # 初期表示

        # 画像表示用のラベル
        self.label_original = QLabel(f"Loaded Target RGB (Linear - {PATCH_SIZE}x{PATCH_SIZE})", alignment=Qt.AlignCenter) # ラベル変更
        self.label_simulated_bayer = QLabel(f"Loaded Input Pseudo-Bayer (Linear - {PATCH_SIZE}x{PATCH_SIZE})", alignment=Qt.AlignCenter) # ラベル変更
        # Real RAW 推論結果のサイズは RAW によって変わる可能性があるためサイズ表示は省略
        self.label_model_prediction = QLabel(f"Model Prediction (Training Patch - {PATCH_SIZE}x{PATCH_SIZE})", alignment=Qt.AlignCenter) # ラベル変更
        self.label_real_raw_demosaic = QLabel("Real RAW Demosaic (Post-processed)", alignment=Qt.AlignCenter) # ラベル名調整

        # 画像表示領域 (カスタムクラスを使用)
        self.image_original = SquareImageLabel()
        self.image_simulated_bayer = SquareImageLabel()
        self.image_model_prediction = SquareImageLabel()
        self.image_real_raw_demosaic = SquareImageLabel()

        # 各画像表示領域にラベルと画像をまとめるウィジェットを作成するヘルパー関数
        def create_image_widget(label: QLabel, image_label: SquareImageLabel) -> QWidget:
            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(label) # ラベルは中央寄せで作成済み
            layout.addWidget(image_label)
            layout.setContentsMargins(5, 5, 5, 5) # 各画像の周囲にマージン
            layout.setSpacing(5)
            widget.setLayout(layout)
            # 画像表示領域が均等にスペースを占めるようにサイズポリシーを設定
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # 各画像ウィジェットの最小サイズを設定 (SquareImageLabelで設定済みだが念のため)
            widget.setMinimumSize(160, 180) # ラベルと余白を考慮して少し大きめに

            return widget

        # 画像表示を横に並べるレイアウト
        image_grid_layout = QHBoxLayout()
        image_grid_layout.addWidget(create_image_widget(self.label_original, self.image_original))
        image_grid_layout.addWidget(create_image_widget(self.label_simulated_bayer, self.image_simulated_bayer))
        image_grid_layout.addWidget(create_image_widget(self.label_model_prediction, self.image_model_prediction))
        image_grid_layout.addWidget(create_image_widget(self.label_real_raw_demosaic, self.image_real_raw_demosaic))
        image_grid_layout.setSpacing(10) # 画像間の間隔

        # メインレイアウト
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_grid_layout)
        # 画像グリッドとプログレスバーの間に固定のスペースを追加
        main_layout.addSpacing(20) # 20ピクセルの固定スペース
        main_layout.addWidget(self.progress_bar)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)


        # --- モデルとデバイス ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = SimpleCNN().to(self.device)
        # self.model = ResidualDemosaicNet().to(self.device)

        # --- 損失関数の定義 (Combined Loss) ---
        # ファイル冒頭の設定値を使用
        self.criterion = CombinedLoss(
            l1_weight=L1_LOSS_WEIGHT,
            perceptual_weight=PERCEPTUAL_LOSS_WEIGHT,
            gradient_weight=GRADIENT_LOSS_WEIGHT,
            chroma_weight=CHROMA_LOSS_WEIGHT,
            local_color_consistency_weight=LOCAL_COLOR_CONSISTENCY_LOSS_WEIGHT, # <<< 追加
            dark_weight_factor=DARK_WEIGHT_FACTOR,
            local_consistency_patch_size=LOCAL_CONSISTENCY_PATCH_SIZE,     # <<< 追加
            local_consistency_stride=LOCAL_CONSISTENCY_STRIDE,           # <<< 追加
            device=self.device
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=OPTIMIZER_WEIGHT_DECAY
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=STEP_LR_STEP_SIZE,
            gamma=STEP_LR_GAMMA       
        )


        # --- データパス取得 ---
        self.input_data_dir = os.path.join(DATA_ROOT_DIR, INPUT_SUBDIR)
        self.target_data_dir = os.path.join(DATA_ROOT_DIR, TARGET_SUBDIR)
        self.raw_paths = self._get_raw_file_paths()

        self.training_thread = None # Initialize to None

        if not os.path.isdir(self.input_data_dir) or not os.path.isdir(self.target_data_dir):
            msg = f"Error: Training data directories not found. Please run dataset.py first.\nExpected:\n- {self.input_data_dir}\n- {self.target_data_dir}"
            print(msg)
            QMessageBox.critical(self, "Data Directory Error", msg)
            self.progress_bar.setFormat("Error: Training data directories not found.")
            self.progress_bar.setValue(0)
            return # Do not proceed to create TrainingWorker

        # --- 学習スレッドの準備と開始 ---
        # TrainingWorkerインスタンスを作成してデータペアの数をチェック
        temp_training_thread = TrainingWorker(self.input_data_dir, self.target_data_dir,
                                              self.raw_paths, self.model, self.criterion,
                                              self.optimizer, self.device, PATCH_SIZE,
                                              self.scheduler)

        if temp_training_thread.total_patches_in_each_epoch == 0:
            msg = f"Error: No matching .npy data pairs found in '{self.input_data_dir}' and '{self.target_data_dir}'. Please generate data or check paths."
            print(msg)
            QMessageBox.critical(self, "Data Error", msg)
            self.progress_bar.setFormat("Error: No training data pairs found.")
            self.progress_bar.setValue(0)
            # self.training_thread remains None, training will not start
        else:
            self.training_thread = temp_training_thread # Assign valid thread

            # シグナルとスロットの接続
            self.training_thread.progress_updated.connect(self.update_progress)
            self.training_thread.image_updated.connect(self.update_image_display)
            self.training_thread.training_finished.connect(self.on_training_finished)
            self.training_thread.error_occurred.connect(self.show_error_message)
            self.training_thread.training_finished.connect(QApplication.instance().quit)

            self.progress_bar.setMaximum(self.training_thread.total_patches_in_each_epoch)
            self.progress_bar.setValue(0)
            initial_lr = self.optimizer.param_groups[0]['lr'] # <<< 初期学習率を取得

            self.progress_bar.setFormat(f"Epoch 1/{NUM_EPOCHS} | LR: {initial_lr:.2e} | Loss: Calculating... | Patch: 0/{self.training_thread.total_patches_in_each_epoch}")
            self.training_thread.start()


    def _get_raw_file_paths(self) -> list[str]:
        """rawディレクトリ以下の全RAWファイルのパスを取得"""
        raw_extensions = ['*.arw', '*.cf2', '*.cr2', '*.crw', '*.dng', '*.erf', '*.3fr', '*.fff', '*.hdr', '*.k25', '*.kdc', '*.mdc', '*.mos', '*.mrw', '*.nef', '*.orf', '*.pef', '*.raf', '*.raw', '*.rdc', '*.sr2', '*.srf', '*.x3f']
        all_paths = []
        raw_base = os.path.abspath(RAW_DIR)
        if not os.path.exists(raw_base):
             print(f"Warning: RAW directory '{RAW_DIR}' not found. Creating it.")
             os.makedirs(RAW_DIR, exist_ok=True)
             return []

        for ext in raw_extensions:
            all_paths.extend(glob.glob(os.path.join(raw_base, '**', ext), recursive=True))
        all_paths.sort()
        print(f"Found {len(all_paths)} RAW files for live demosaic.")
        return all_paths



    @Slot(int, int, float, int, int, float) # <<< float型 (current_lr) を追加
    def update_progress(self, current_epoch: int, total_epochs: int, avg_epoch_loss: float, total_patches: int, processed_patches: int, current_lr: float): # <<< current_lr を引数に追加
        """学習進捗に応じてプログレスバーとテキストを更新"""
        if total_patches > 0:
            self.progress_bar.setValue(processed_patches)
            # プログレスバーの書式文字列に学習率を追加 (例: 指数表記 .2e)
            self.progress_bar.setFormat(f"Epoch {current_epoch}/{total_epochs} | LR: {current_lr:.2e} | Loss: {avg_epoch_loss:.4f} | Patch: {processed_patches}/{total_patches} (%p%)")
        else:
             self.progress_bar.setValue(0)
             self.progress_bar.setFormat(f"Epoch {current_epoch}/{total_epochs} | LR: {current_lr:.2e} | Loss: {avg_epoch_loss:.4f} | No patches to process.")


    @Slot()
    def on_training_finished(self):
        """学習完了時に呼び出される"""
        print("Training finished signal received.")
        self.progress_bar.setFormat("Training Finished.")
        if self.training_thread and self.training_thread.total_patches_in_each_epoch > 0:
             self.progress_bar.setValue(self.training_thread.total_patches_in_each_epoch)
        else:
             self.progress_bar.setValue(self.progress_bar.maximum())

        if self.training_thread and self.training_thread.isRunning():
             self.training_thread.quit()
             # self.training_thread.wait() # Avoid blocking GUI

        QMessageBox.information(self, "Training Finished", "The training process has finished.")
        print("GUI application is ready after training.")

    @Slot(str)
    def show_error_message(self, message: str):
        """エラーメッセージをGUIで表示"""
        print(f"Error from worker thread: {message}")
        QMessageBox.critical(self, "Training Error", message)
        self.progress_bar.setFormat(f"Error: {message[:50]}...")

    def closeEvent(self, event):
        """ウィンドウを閉じるときのイベントハンドラ"""
        if self.training_thread and self.training_thread.isRunning():
            print("Attempting to stop training thread on window close...")
            self.training_thread.stop()
            self.hide() 
            event.ignore()
            print("Window hidden, waiting for thread to signal completion to quit application...")
        else:
            print("Training thread not running or already finished, closing window immediately.")
            super().closeEvent(event)

def ensure_directories_exist(dir_list: list[str]):
    """指定されたディレクトリのリストが存在することを確認し、なければ作成する。"""
    for d_path in dir_list:
        os.makedirs(d_path, exist_ok=True)
        if DEBUG_MODE: print(f"Ensured directory exists: {d_path}")

if __name__ == "__main__":
    # 必要なディレクトリを作成
    dirs_to_create = [
        DATA_ROOT_DIR,
        os.path.join(DATA_ROOT_DIR, INPUT_SUBDIR),
        os.path.join(DATA_ROOT_DIR, TARGET_SUBDIR),
        RAW_DIR,
        MODEL_DIR,
        DEVELOPED_DIR
    ]
    ensure_directories_exist(dirs_to_create)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())