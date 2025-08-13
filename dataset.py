import sys
import os
import glob
import numpy as np
from PIL import Image # Pillow (GUI表示変換に使用)
import rawpy # rawpy (RAWファイル読み込みに使用)
# import torch # 疑似カラー化表示のために利用 - Not directly used in this version, can be removed if not needed elsewhere
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QLabel, QProgressBar, QPushButton, QSizePolicy,
                               QFileDialog, QMessageBox)
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSize
import argparse
import traceback # エラー時のトレースバック表示用

# --- Added for Parallelization ---
import concurrent.futures
import multiprocessing
# ---------------------------------

# --- 設定 ---
# 入力RAWファイルディレクトリ (コマンドライン引数で指定可能にするが、デフォルト値も設定)
DEFAULT_RAW_INPUT_DIR = "raw_input"
# 出力データディレクトリ (コマンドライン引数で指定可能にするが、デフォルト値も設定)
DEFAULT_OUTPUT_DIR = "training_data"
# OUTPUT_INPUT_SUBDIR = "input_rgbg_linear" # 学習データ（疑似RGBG）保存先サブディレクトリ (旧)
OUTPUT_PSEUDO_BAYER_SUBDIR = "input_pseudo_bayer_linear" # 学習データ（疑似ベイヤー）保存先サブディレクトリ (新)
OUTPUT_TARGET_SUBDIR = "target_rgb_linear" # 教師データ（デモザイク済みRGB）保存先サブディレクトリ

# RAW画像から切り出すタイルのサイズ
# 256px -> 512px に変更
TILE_SIZE = 512 # 512x512

# 生成する学習データ・教師データのサイズ
# 512px -> 128px (圧縮率4倍) なので、TILE_SIZE / 4 になる
OUTPUT_PATCH_SIZE = TILE_SIZE // 4 # 128x128

# GUI表示用の画像リサイズサイズ
GUI_DISPLAY_SIZE = 200 # 200x200 にリサイズして表示

# デバッグモード (Trueにすると詳細なログが出力されます)
DEBUG_MODE = True
DEBUG_LOG_PATCHES = 10 # デバッグログを出力するパッチ数 (各ファイルの最初のN個の8x8 blocks)

# --- データ処理ヘルパー関数 ---

def get_channel_index(y, x, pos_to_channel_map):
    """Helper to get the channel index (0=R, 1=Gr, 2=Gb, 3=B) for a pixel at (y, x) based on the 2x2 pattern mapping."""
    # The color at (y, x) is determined by its position within the repeating 2x2 Bayer pattern
    r_off = y % 2
    c_off = x % 2
    # The pos_to_channel_map is based on the global tile's top-left (0,0) pattern.
    # We use the global tile coordinates (y, x) to determine the position within the 2x2 pattern.
    return pos_to_channel_map.get((r_off, c_off), -1) # -1 if not mapped


def convert_np_to_qpixmap_display(np_image: np.ndarray | None, display_size: int, is_pseudo_bayer: bool = False) -> QPixmap:
    """
    numpy array (HWC, float or uint8) を表示用に uint8 に変換し、指定サイズにリサイズしてQPixmapとして返す。
    RAWや疑似ベイヤーなど、可視化のために正規化・変換を行う。
    is_pseudo_bayer=True の場合、128x128x1 の疑似ベイヤー入力を擬似カラー表示する。
    """
    if np_image is None or np_image.size == 0:
        if DEBUG_MODE: # Debug log even if not in is_pseudo_bayer path
             print("Debug: convert_np_to_qpixmap_display received empty numpy array.")
        return QPixmap() # 空のPixampを返す

    # 入力がfloatの場合は0-1にクリップして0-255 uint8に変換
    # 入力がuint8の場合はそのまま使用
    # Pseudo-Bayer (float) もここで一時的に float -> uint8 スケーリングを行うが、
    # 擬似カラー化のために float に戻す必要がある。
    # 擬似カラー化は float (0-1) で行うのが自然。
    # なので、一度 float32 に統一してから処理を分岐する。

    if np_image.dtype in [np.uint8, np.uint16]: # Handle common input types
         if np_image.dtype == np.uint16:
              # Assume uint16 is raw data, scale down to approximate 0-1 (crude)
              # Better to use rawpy processing which returns float32 0-1
              temp_np_float = np_image.astype(np.float32) / np.iinfo(np.uint16).max
              if DEBUG_MODE: print(f"Debug: Converting uint16 to float32 for display.")
         else: # uint8
             temp_np_float = np_image.astype(np.float32) / 255.0
             if DEBUG_MODE: print(f"Debug: Converting uint8 to float32 for display.")
    elif np_image.dtype in [np.float32, np.float64]:
        temp_np_float = np_image.astype(np.float32) # Keep float32
    else:
        print(f"Warning: convert_np_to_qpixmap_display received unsupported dtype: {np_image.dtype}. Expected uint8, uint16, float32/64.")
        return QPixmap()


    img = QImage() # QImageオブジェクトを先に宣言

    # Handle different scenarios based on shape and is_pseudo_bayer flag
    if temp_np_float.ndim == 3:
        h, w, c = temp_np_float.shape
        # FIX: Handle 1-channel 3D array first before checking for 3 channels
        if c == 1: # Handle 1-channel input (like pseudo-Bayer or HxWx1 grayscale)
            if is_pseudo_bayer and h == OUTPUT_PATCH_SIZE and w == OUTPUT_PATCH_SIZE:
                 # Existing pseudo-color logic for 128x128x1 Pseudo-Bayer
                 if DEBUG_MODE:
                      print(f"Debug: Processing 1-channel pseudo-Bayer for display (ndim=3, c=1). Shape: ({h}, {w}, {c}), Range: min={np.min(temp_np_float):.4f}, max={np.max(temp_np_float):.4f}, mean={np.mean(temp_np_float):.4f}")

                 # Create a 3-channel RGB image for pseudo-color display
                 pseudo_color_rgb = np.zeros((h, w, 3), dtype=np.float32)
                 display_np_float_1ch = temp_np_float[:, :, 0] # Squeeze for easier slicing

                 # Populate the RGB channels based on the pseudo-bayer pattern (RGGB)
                 # Assuming output 2x2 pseudo-Bayer maps to display channels:
                 # (0,0) -> R (pseudo_color_rgb[:,:,0])
                 # (0,1) -> Gr (pseudo_color_rgb[:,:,1])
                 # (1,0) -> Gb (pseudo_color_rgb[:,:,1])
                 # (1,1) -> B (pseudo_color_rgb[:,:,2])

                 # Note: The pseudo-bayer input is 128x128x1 where each pixel's value corresponds
                 # to the average of a 4x4 region in the 512 tile, grouped by channel (R, Gr, Gb, B)
                 # according to the 8x8 input block.
                 # The display pseudo-color needs to map these 128x128 values to the correct display
                 # channel based on their position within the 128x128 grid, assuming a *visual* RGGB pattern
                 # on the 128x128 grid itself.

                 # This part of the logic in the original code seems to apply the pseudo-bayer value
                 # to display channels based on the *display grid* position (0::2, 0::2 etc.),
                 # which corresponds to the logical position in the 128x128 pseudo-bayer pattern.
                 # This is correct for visualizing the pseudo-bayer input itself.

                 pseudo_color_rgb[0::2, 0::2, 0] = display_np_float_1ch[0::2, 0::2] # R positions get R value (Red channel)
                 pseudo_color_rgb[0::2, 1::2, 1] = display_np_float_1ch[0::2, 1::2] # Gr positions get G value (Green channel)
                 pseudo_color_rgb[1::2, 0::2, 1] = display_np_float_1ch[1::2, 0::2] # Gb positions get G value (Green channel)
                 pseudo_color_rgb[1::2, 1::2, 2] = display_np_float_1ch[1::2, 1::2] # B positions get B value (Blue channel)


                 if DEBUG_MODE:
                      print(f"Debug: Pseudo-colored RGB image created (float32) from 1-channel input. Shape: {pseudo_color_rgb.shape}, Range: min={np.min(pseudo_color_rgb):.4f}, max={np.max(pseudo_color_rgb):.4f}, mean={np.mean(pseudo_color_rgb):.4f}")


                 # Convert float32 (0-1 range) to uint8 (0-255 range) for QImage display
                 display_np_uint8_rgb = np.clip(pseudo_color_rgb * 255.0, 0, 255).astype(np.uint8)

                 if DEBUG_MODE:
                      print(f"Debug: Converting pseudo-colored float32 to uint8 for QImage. Output shape: {display_np_uint8_rgb.shape}, Dtype: {display_np_uint8_rgb.dtype}. Range: min={np.min(display_np_uint8_rgb):.4f}, max={np.max(display_np_uint8_rgb):.4f}")
                      print(f"Debug: display_np_uint8_rgb is C-contiguous: {display_np_uint8_rgb.flags.c_contiguous}")


                 bytes_per_line = w * 3
                 display_np_contiguous = np.ascontiguousarray(display_np_uint8_rgb)
                 if DEBUG_MODE and not display_np_contiguous.flags.c_contiguous:
                      print("Warning: np.ascontiguousarray did not result in C_CONTIGUOUS array for pseudo-color display before QImage creation.")

                 img = QImage(display_np_contiguous.data, w, h, bytes_per_line, QImage.Format_RGB888)
                 if img.isNull():
                      if DEBUG_MODE: print("Error: QImage creation failed for pseudo-color display.")
                      return QPixmap()
                 if DEBUG_MODE: print("Debug: Created QImage with Format_RGB888 from pseudo-colored uint8 data.")

            else:
                # Not a pseudo-bayer 1-channel input, treat as grayscale HxWx1
                if DEBUG_MODE:
                    print(f"Debug: Processing 1-channel grayscale (ndim=3, c=1) for display. Shape: ({h}, {w}), Range: min={np.min(temp_np_float):.4f}, max={np.max(temp_np_float):.4f}, mean={np.mean(temp_np_float):.4f}")
                display_np_float_1ch = temp_np_float[:, :, 0] # Squeeze for grayscale handling below

                # Convert float32 (0-1 range) to uint8 (0-255 range) for QImage display
                display_np_uint8 = np.clip(display_np_float_1ch * 255.0, 0, 255).astype(np.uint8)
                if DEBUG_MODE:
                     print(f"Debug: Converted grayscale float32 to uint8 (ndim=3, c=1). Output shape: {display_np_uint8.shape}, Dtype: {display_np_uint8.dtype}. Range: min={np.min(display_np_uint8):.4f}, max={np.max(display_np_uint8):.4f}")
                     print(f"Debug: display_np_uint8 is C-contiguous: {display_np_uint8.flags.c_contiguous}")

                bytes_per_line = w # For Grayscale8, stride is just width
                display_np_contiguous = np.ascontiguousarray(display_np_uint8)
                if DEBUG_MODE and not display_np_contiguous.flags.c_contiguous:
                     print("Warning: np.ascontiguousarray did not result in C_CONTIGUOUS array for grayscale display before QImage creation.")

                img = QImage(display_np_contiguous.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                if img.isNull():
                     if DEBUG_MODE: print("Error: QImage creation failed for grayscale display (ndim=3, c=1).")
                     return QPixmap()
                if DEBUG_MODE: print("Debug: Created QImage with Format_Grayscale8 from uint8 data (ndim=3, c=1).")


        elif c == 3: # Handle 3-channel input (like Demosaiced RGB)
            if DEBUG_MODE:
                 print(f"Debug: Processing 3-channel RGB for display (ndim=3, c=3). Shape: ({h}, {w}, {c}), Range: min={np.min(temp_np_float):.4f}, max={np.max(temp_np_float):.4f}, mean={np.mean(temp_np_float):.4f}")
            # Convert float32 (0-1 range) to uint8 (0-255 range) for QImage display
            display_np_uint8 = np.clip(temp_np_float * 255.0, 0, 255).astype(np.uint8)
            if DEBUG_MODE:
                 print(f"Debug: Converted RGB float32 to uint8. Output shape: {display_np_uint8.shape}, Dtype: {display_np_uint8.dtype}. Range: min={np.min(display_np_uint8):.4f}, max={np.max(display_np_uint8):.4f}")
                 print(f"Debug: display_np_uint8 is C-contiguous: {display_np_uint8.flags.c_contiguous}")


            bytes_per_line = w * c
            display_np_contiguous = np.ascontiguousarray(display_np_uint8)
            if DEBUG_MODE and not display_np_contiguous.flags.c_contiguous:
                 print("Warning: np.ascontiguousarray did not result in C_CONTIGUOUS array for 3ch display before QImage creation.")

            img = QImage(display_np_contiguous.data, w, h, bytes_per_line, QImage.Format_RGB888)
            if img.isNull():
                 if DEBUG_MODE: print("Error: QImage creation failed for 3-channel RGB display.")
                 return QPixmap()
            if DEBUG_MODE: print("Debug: Created QImage with Format_RGB888 from 3-channel uint8 data.")

        else: # Unsupported number of channels for ndim == 3
            print(f"Warning: convert_np_to_qpixmap_display received unsupported number of channels ({c}) for 3D array display. Expected 1 or 3.")
            return QPixmap()


    elif temp_np_float.ndim == 2: # Handle 2D grayscale array (H, W) - This is where Original Tile goes
         h, w = temp_np_float.shape
         if DEBUG_MODE:
              print(f"Debug: Processing 2D grayscale for display (ndim=2). Shape: ({h}, {w}), Range: min={np.min(temp_np_float):.4f}, max={np.max(temp_np_float):.4f}, mean={np.mean(temp_np_float):.4f}")

         # Convert float32 (0-1 range) to uint8 (0-255 range) for QImage display
         display_np_uint8 = np.clip(temp_np_float * 255.0, 0, 255).astype(np.uint8)
         if DEBUG_MODE:
              print(f"Debug: Converted grayscale float32 to uint8 (ndim=2). Output shape: {display_np_uint8.shape}, Dtype: {display_np_uint8.dtype}. Range: min={np.min(display_np_uint8):.4f}, max={np.max(display_np_uint8):.4f}")
              print(f"Debug: display_np_uint8 is C-contiguous: {display_np_uint8.flags.c_contiguous}")


         bytes_per_line = w # For Grayscale8, stride is just width
         display_np_contiguous = np.ascontiguousarray(display_np_uint8)
         if DEBUG_MODE and not display_np_contiguous.flags.c_contiguous:
              print("Warning: np.ascontiguousarray did not result in C_CONTIGUOUS array for grayscale display before QImage creation.")

         img = QImage(display_np_contiguous.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
         if img.isNull():
              if DEBUG_MODE: print("Error: QImage creation failed for grayscale display (ndim=2).")
              return QPixmap()
         if DEBUG_MODE: print("Debug: Created QImage with Format_Grayscale8 from uint8 data (ndim=2).")


    else:
        print(f"Warning: convert_np_to_qpixmap_display received unsupported number of dimensions: {temp_np_float.ndim}. Expected 2 or 3.")
        return QPixmap()


    if img.isNull():
        # Should have been caught earlier, but double check
        if DEBUG_MODE:
             print(f"Failed to create QImage from numpy array (after contiguity check) - Fallback check with shape {np_image.shape} and dtype {np_image.dtype}")
        return QPixmap()

    # リサイズ
    # imgがnullでないことを確認してからscaledを呼び出す
    img = img.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # QImage to QPixmap
    return QPixmap.fromImage(img)


def read_raw_process_linear(raw_filepath: str) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict] | tuple[None, None, None, None, None]:
    """
    RAWファイルを読み込み、ブラックレベル補正とホワイトレベル正規化を行った
    高ビットリニアなNumPy配列 (float32) を返す。
    ベイヤーパターンインデックス、ホワイトレベル、ブラックレベル、チャンネルマッピングも返す。
    エラー時は (None, None, None, None, None) を返す。
    """
    # このtryブロックには、rawpyの読み込みエラー、ファイルが見つからないエラー、その他の予期せぬエラーを
    # 捕捉するためのexceptブロックが後続しています。
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

            # --- ベイヤーパターンの色文字列とチャンネルマッピングを取得 ---
            # rawpy.raw_pattern と raw.color_desc を使ってマッピングを作成
            pos_to_channel_map = {}
            bayer_pattern_str = "UNKNOWN" # Default

            try:
                # rawpy 内部色インデックスから色文字列を取得するルックアップテーブルを作成
                color_lookup = []
                if isinstance(color_description, bytes):
                    color_lookup = [chr(b) for b in color_description]
                elif isinstance(color_description, tuple):
                    color_lookup = list(color_description)
                elif isinstance(color_description, str):
                     color_lookup = list(color_description)
                # else: color_lookup remains empty

                # Determine the color string at each pattern position using raw.raw_pattern indices and color_lookup
                pattern_color_strings = [['', ''], ['', '']]
                if len(color_lookup) >= 4: # Ensure enough color descriptions are available
                    for r in range(2):
                        for c in range(2):
                            idx = bayer_pattern_indices[r, c] # rawpy internal color index
                            if idx < len(color_lookup):
                                pattern_color_strings[r][c] = color_lookup[idx].upper()
                            else:
                                if DEBUG_MODE:
                                     print(f"Warning: RawPy color index {idx} out of bounds for color_lookup ({len(color_lookup)}) at pattern position ({r},{c}) for {os.path.basename(raw_filepath)}.")
                                pattern_color_strings[r][c] = 'UNKNOWN'
                else:
                    if DEBUG_MODE:
                         print(f"Warning: raw.color_desc has less than 4 elements ({len(color_lookup)}) for {os.path.basename(raw_filepath)}. Cannot determine color strings reliably from indices.")


                # Construct the 4-char pattern string (e.g., "RGGB") for logging
                pattern_str_list = [pattern_color_strings[r][c][0] if pattern_color_strings[r][c] not in ['', 'UNKNOWN'] else '?' for r in range(2) for c in range(2)]
                bayer_pattern_str = "".join(pattern_str_list)


                # Standard channels mapping (R:0, Gr:1, Gb:2, B:3)
                # We map the rawpy pattern position (r_off, c_off) to our channel index (0-3)
                # based on the color string determined for that position and standard Bayer conventions.
                mapped_channels_count = 0
                # Keep track of which of our target channels (0,1,2,3) have been mapped from a pattern position
                mapped_channel_indices = set()

                for r_off in range(2):
                    for c_off in range(2):
                        current_pos = (r_off, c_off)
                        # The color string at this position according to rawpy's mapping
                        color_str_at_pos = pattern_color_strings[r_off][c_off]

                        # Assign to our standard channels based on color and typical Gr/Gb positions
                        if 'R' in color_str_at_pos and 0 not in mapped_channel_indices:
                             pos_to_channel_map[current_pos] = 0 # R -> Channel 0
                             mapped_channel_indices.add(0)
                             mapped_channels_count += 1
                        elif 'B' in color_str_at_pos and 3 not in mapped_channel_indices:
                             pos_to_channel_map[current_pos] = 3 # B -> Channel 3
                             mapped_channel_indices.add(3)
                             mapped_channels_count += 1
                        elif 'G' in color_str_at_pos:
                             # Assign G based on standard Gr/Gb positions (0,1) and (1,0) relative to pattern start
                             if (r_off, c_off) == (0, 1) and 1 not in mapped_channel_indices:
                                 pos_to_channel_map[current_pos] = 1 # Gr -> Channel 1
                                 mapped_channel_indices.add(1)
                                 mapped_channels_count += 1
                             elif (r_off, c_off) == (1, 0) and 2 not in mapped_channel_indices:
                                 pos_to_channel_map[current_pos] = 2 # Gb -> Channel 2
                                 mapped_channel_indices.add(2)
                                 mapped_channels_count += 1
                             else:
                                 # If G is at a non-standard position OR Gr/Gb are already mapped,
                                 # try to map to the remaining G channel if any.
                                 if 1 not in mapped_channel_indices:
                                      pos_to_channel_map[current_pos] = 1 # Assign to Gr if not yet mapped
                                      mapped_channel_indices.add(1)
                                      mapped_channels_count += 1
                                 elif 2 not in mapped_channel_indices:
                                      pos_to_channel_map[current_pos] = 2 # Assign to Gb if not yet mapped
                                      mapped_channel_indices.add(2)
                                      mapped_channels_count += 1
                                 else:
                                       if DEBUG_MODE:
                                            print(f"Warning: G pixel at raw_pattern position ({r_off},{c_off}) could not be confidently mapped to Gr or Gb channel (1 or 2) for pattern '{bayer_pattern_str}' for {os.path.basename(raw_filepath)}. Its value might not be used correctly in 4ch input.")

            except Exception as e:
                if DEBUG_MODE:
                    print(f"Warning: Could not fully determine Bayer pattern string and channel mapping from metadata for {os.path.basename(raw_filepath)}: {e}. Proceeding with potentially incomplete mapping.")
                # pos_to_channel_map might be incomplete here.

            # Check if all 4 pattern positions (0,0)...(1,1) were mapped
            if len(pos_to_channel_map) != 4:
                 if DEBUG_MODE:
                      print(f"Warning: Only {len(pos_to_channel_map)}/4 pattern positions mapped to channels for {os.path.basename(raw_filepath)}. Data generation might be incorrect.")
                 # If mapping is incomplete, data generation might fail or produce incorrect results.
                 # We'll proceed but expect potential issues.


            # --- ブラックレベル補正 ---
            raw_image_corrected = np.copy(raw_image_np) # 補正用にコピーを作成
            # 2x2 パターン内の各位置に対応するグローバルなスライス
            slices = {
                (0, 0): (slice(0, None, 2), slice(0, None, 2)), # ::2, ::2
                (0, 1): (slice(0, None, 2), slice(1, None, 2)), # ::2, 1::2
                (1, 0): (slice(1, None, 2), slice(0, None, 2)), # 1::2, ::2
                (1, 1): (slice(1, None, 2), slice(1, None, 2)), # 1::2, 1::2
            }

            if len(black_level_per_channel) >= 4:
                 for r_off in range(2):
                     for c_off in range(2):
                         row_slice, col_slice = slices[(r_off, c_off)]
                         # raw.camera_whitebalance と同様に raw.color_desc の順序に依存すると仮定
                         # raw.raw_pattern で示される、この位置の色に対応するrawpy内部インデックス
                         pattern_index = bayer_pattern_indices[r_off, c_off] # black_level_per_channel へのインデックス (RawPyの慣例を想定)

                         if pattern_index < len(black_level_per_channel):
                             bl_value = black_level_per_channel[pattern_index]
                             raw_image_corrected[row_slice, col_slice] = np.maximum(raw_image_np[row_slice, col_slice] - bl_value, 0)
                         else:
                             if DEBUG_MODE:
                                  print(f"Warning: Pattern index {pattern_index} out of bounds for black_level_per_channel ({len(black_level_per_channel)}) at position ({r_off},{c_off}) for {os.path.basename(raw_filepath)}. Pixels not black level corrected.")
            else:
                 if DEBUG_MODE:
                      print("Warning: black_level_per_channel is not available or too short (<4 elements). Skipping black level correction.")


            # --- 正規化 (0-1 スケール) ---
            raw_image_normalized = np.zeros_like(raw_image_corrected) # 正規化後の画像を格納する配列 (float32)

            # black_level_per_channel の要素数が4未満の場合は、正確な個別正規化ができない
            if len(black_level_per_channel) < 4:
                 if DEBUG_MODE:
                      print("Warning: black_level_per_channel has less than 4 elements. Cannot perform accurate per-channel normalization. Falling back to global white level normalization.")
                 # フォールバックとして、単純にホワイトレベルで割る（ブラックレベルを引いた後なので）
                 if white_level > 1e-5:
                     raw_image_normalized = raw_image_corrected / white_level
                 else:
                     print(f"Error: white_level is zero or too small ({white_level}) for {os.path.basename(raw_filepath)}. Cannot normalize. Skipping.")
                     return None, None, None, None, None
            else:
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
                            else:
                                if DEBUG_MODE:
                                     print(f"Warning: Normalization denominator zero or negative for pattern position ({r_off},{c_off}) for {os.path.basename(raw_filepath)}. Setting slice to 0.")
                                raw_image_normalized[row_slice, col_slice] = 0.0
                        else:
                             # black_level_per_channel が4つ以上ある場合は、通常ここには来ないはず
                             if DEBUG_MODE:
                                  print(f"Warning: Pattern index {pattern_index} out of bounds for black_level_per_channel ({len(black_level_per_channel)}) at position ({r_off},{c_off}) for {os.path.basename(raw_filepath)}. Setting slice to 0.")
                             raw_image_normalized[row_slice, col_slice] = 0.0


            # 正規化後のデータが確実に [0, 1] の範囲に収まるようにクリップ
            # 計算誤差や特殊なケースでわずかに範囲外に出る可能性に備える
            raw_image_normalized = np.clip(raw_image_normalized, 0, 1)

            # Ensure raw_image_normalized is float32 HxW before returning
            if raw_image_normalized.ndim == 3 and raw_image_normalized.shape[2] == 1:
                 raw_image_normalized = raw_image_normalized[:,:,0]
            elif raw_image_normalized.ndim != 2:
                 print(f"Error: Unexpected shape for normalized RAW data ({raw_image_normalized.shape}) for {os.path.basename(raw_filepath)}. Expecting HxW after processing. Skipping.")
                 return None, None, None, None, None

            # ベイヤーパターンインデックスはそのまま返す (create_data_pair_from_tileでは直接使わないが情報として)
            # チャンネルマッピングは create_data_pair_from_tile でピクセル抽出に使う
            return raw_image_normalized, bayer_pattern_indices, white_level, np.array(black_level_per_channel), pos_to_channel_map

    except rawpy.LibRawFileUnsupportedError:
        print(f"Error: {os.path.basename(raw_filepath)} is not a supported RAW file or is corrupted. Skipping.")
        return None, None, None, None, None
    except FileNotFoundError:
        print(f"Error: RAW file not found at {raw_filepath}. Skipping.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error processing RAW file {os.path.basename(raw_filepath)}: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return None, None, None, None, None

# Keep create_data_pair_from_tile as is - this function will be executed in parallel processes
def create_data_pair_from_tile(raw_tile_512: np.ndarray, tile_origin_yx: tuple[int, int], global_bayer_pattern_indices: np.ndarray, pos_to_channel_map: dict, file_name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str, np.ndarray] | tuple[None, None, tuple[int, int], str, None]:
    """
    512x512のRAWタイルから128x128x1の疑似ベイヤー入力データと128x128x3のRGB教師データを生成する。
    マルチプロセスで実行されるため、必要な全ての入力データを受け取り、
    結果、元のタイル座標、元のファイル名ベース、元のタイルデータを返す。

    Args:
        raw_tile_512: 512x512 float32 のブラックレベル補正・正規化済みRAWタイル。
        tile_origin_yx: このタイルが元の画像から切り出された左上隅の座標 (y, x)。
        global_bayer_pattern_indices: rawpy.raw_pattern (2x2 numpy array)。 (Note: Not directly used inside, but kept for compatibility/info if needed)
        pos_to_channel_map: 2x2パターン内の位置 (r_off, c_off) -> 標準チャンネルインデックス (0-3) のマッピング。
        file_name: 処理中の元ファイル名 (デバッグログ用)


    Returns:
        tuple: (input_pseudo_bayer_128: 128x128x1 float32 NumPy array,
                target_rgb_128: 128x128x3 float32 NumPy array,
                tile_origin_yx: tuple[int, int],
                file_name_base: str,
                raw_tile_512: 512x512 float32 NumPy array)
        エラー時は (None, None, tile_origin_yx, file_name_base, raw_tile_512)
    """
    h, w = raw_tile_512.shape # Expect (512, 512)
    tile_ty = tile_origin_yx[0] // TILE_SIZE
    tile_tx = tile_origin_yx[1] // TILE_SIZE
    file_name_base = f"{os.path.splitext(file_name)[0]}_tile_{tile_ty}_{tile_tx}"

    if h != TILE_SIZE or w != TILE_SIZE:
         if DEBUG_MODE:
              print(f"Warning: Input tile size mismatch for {file_name} at origin {tile_origin_yx}. Expected ({TILE_SIZE}, {TILE_SIZE}), got {raw_tile_512.shape}. Skipping.")
         return None, None, tile_origin_yx, file_name_base, raw_tile_512

    if len(pos_to_channel_map) != 4:
         if DEBUG_MODE:
              print(f"Warning: Incomplete pos_to_channel_map ({len(pos_to_channel_map)}/4) for tile at {tile_origin_yx} in {file_name}. Data generation might be incorrect. Skipping tile.")
         return None, None, tile_origin_yx, file_name_base, raw_tile_512


    # 出力データの初期化
    # 学習データは128x128x1に変更
    input_pseudo_bayer_2d = np.zeros((OUTPUT_PATCH_SIZE, OUTPUT_PATCH_SIZE), dtype=np.float32) # Initialize as 2D first
    target_rgb_128 = np.zeros((OUTPUT_PATCH_SIZE, OUTPUT_PATCH_SIZE, 3), dtype=np.float32)

    # --- Generate Teacher Data (128x128x3) ---
    # Iterate through 4x4 blocks in input tile, which map to 1x1 pixels in output
    for oy in range(OUTPUT_PATCH_SIZE): # 0 to 127
        for ox in range(OUTPUT_PATCH_SIZE): # 0 to 127
            # Corresponding 4x4 block in the 512x512 input tile
            iy_start = oy * 4
            ix_start = ox * 4
            input_4x4_block = raw_tile_512[iy_start : iy_start+4, ix_start : ix_start+4] # Shape (4, 4)

            r_values, g_values, b_values = [], [], []

            # Iterate through pixels in the 4x4 block
            # Need global coordinates relative to the tile for channel checking
            global_iy_start = tile_origin_yx[0] + iy_start
            global_ix_start = tile_origin_yx[1] + ix_start

            for r_rel in range(4):
                for c_rel in range(4):
                    # Global coordinates within the original RAW image for this pixel
                    current_global_y = global_iy_start + r_rel
                    current_global_x = global_ix_start + c_rel
                    pixel_value = input_4x4_block[r_rel, c_rel]

                    # Determine channel based on global position within the tile's Bayer pattern
                    # Use the helper function based on position relative to the pattern start (0,0)
                    channel_index = get_channel_index(current_global_y, current_global_x, pos_to_channel_map)

                    if channel_index == 0: # R
                        r_values.append(pixel_value)
                    elif channel_index in [1, 2]: # Gr or Gb (average all G for teacher)
                        g_values.append(pixel_value)
                    elif channel_index == 3: # B
                        b_values.append(pixel_value)


            # Calculate averages for RGB teacher data
            avg_r_rgb = np.mean(r_values) if r_values else 0.0
            avg_g_rgb = np.mean(g_values) if g_values else 0.0
            avg_b_rgb = np.mean(b_values) if b_values else 0.0

            # Store averages in the 3-channel target pixel
            target_rgb_128[oy, ox, 0] = avg_r_rgb
            target_rgb_128[oy, ox, 1] = avg_g_rgb
            target_rgb_128[oy, ox, 2] = avg_b_rgb


    # --- Generate Input Data (128x128x1 Pseudo Bayer) ---
    # Iterate through 8x8 blocks in input tile, which map to 2x2 regions in output
    for iy_8 in range(0, TILE_SIZE, 8): # 0, 8, 16, ..., 504
        for ix_8 in range(0, TILE_SIZE, 8): # 0, 8, 16, ..., 504

            # Corresponding top-left coordinates in the 128x128 output grid for the 2x2 pattern
            # An 8x8 input block at (iy_8, ix_8) maps to a 2x2 output region starting at (iy_8/4, ix_8/4).
            oy_2x2_start = iy_8 // 4 # 0, 2, 4, ..., 126
            ox_2x2_start = ix_8 // 4 # 0, 2, 4, ..., 126

            # Define the four 4x4 quadrants within this 8x8 input block
            tl_4x4 = raw_tile_512[iy_8 : iy_8+4, ix_8 : ix_8+4]
            tr_4x4 = raw_tile_512[iy_8 : iy_8+4, ix_8+4 : ix_8+8]
            bl_4x4 = raw_tile_512[iy_8+4 : iy_8+8, ix_8 : ix_8+4]
            br_4x4 = raw_tile_512[iy_8+4 : iy_8+8, ix_8+4 : ix_8+8]

            # Calculate the required averages from these quadrants, filtering by the SPECIFIC channel (R, Gr, Gb, B)
            # This requires iterating through the 4x4 quadrant and checking the global position's channel.

            # Global coordinates of the top-left corner of the current 8x8 block relative to RAW image origin
            global_iy_8_start = tile_origin_yx[0] + iy_8
            global_ix_8_start = tile_origin_yx[1] + ix_8


            # Avg R from TL 4x4 (of 8x8). Filter pixels that are R in the GLOBAL pattern.
            r_values_tl = []
            for r_rel in range(4):
                for c_rel in range(4):
                    current_global_y = global_iy_8_start + r_rel
                    current_global_x = global_ix_8_start + c_rel
                    channel_index = get_channel_index(current_global_y, current_global_x, pos_to_channel_map)
                    if channel_index == 0: # Only R pixels
                        r_values_tl.append(tl_4x4[r_rel, c_rel])
            avg_r_pseudo = np.mean(r_values_tl) if r_values_tl else 0.0


            # Avg Gr from TR 4x4 (of 8x8). Filter pixels that are Gr in the GLOBAL pattern.
            gr_values_tr = []
            for r_rel in range(4):
                for c_rel in range(4):
                    current_global_y = global_iy_8_start + r_rel
                    current_global_x = global_ix_8_start + 4 + c_rel # +4 for the column offset of TR quadrant
                    channel_index = get_channel_index(current_global_y, current_global_x, pos_to_channel_map)
                    if channel_index == 1: # Only Gr pixels (channel 1)
                        gr_values_tr.append(tr_4x4[r_rel, c_rel])
            avg_gr_pseudo = np.mean(gr_values_tr) if gr_values_tr else 0.0


            # Avg Gb from BL 4x4 (of 8x8). Filter pixels that are Gb in the GLOBAL pattern.
            gb_values_bl = []
            for r_rel in range(4):
                for c_rel in range(4):
                    current_global_y = global_iy_8_start + 4 + r_rel # +4 for the row offset of BL quadrant
                    current_global_x = global_ix_8_start + c_rel
                    channel_index = get_channel_index(current_global_y, current_global_x, pos_to_channel_map)
                    if channel_index == 2: # Only Gb pixels (channel 2)
                        gb_values_bl.append(bl_4x4[r_rel, c_rel])
            avg_gb_pseudo = np.mean(gb_values_bl) if gb_values_bl else 0.0

            # Avg B from BR 4x4 (of 8x8). Filter pixels that are B in the GLOBAL pattern.
            b_values_br = []
            for r_rel in range(4):
                for c_rel in range(4):
                    current_global_y = global_iy_8_start + 4 + r_rel # +4 for the row offset of BR quadrant
                    current_global_x = global_ix_8_start + 4 + c_rel # +4 for the column offset of BR quadrant
                    channel_index = get_channel_index(current_global_y, current_global_x, pos_to_channel_map)
                    if channel_index == 3: # Only B pixels
                        b_values_br.append(br_4x4[r_rel, c_rel])
            avg_b_pseudo = np.mean(b_values_br) if b_values_br else 0.0


            # Place these averages into the corresponding 2x2 region in the 128x128x1 output (2D array first)
            # Assuming the output 2x2 pseudo-Bayer pattern is always RGGB at the top-left of each 2x2 block:
            # (0,0) -> R, (0,1) -> Gr, (1,0) -> Gb, (1,1) -> B

            # The output coordinates for this 2x2 region start at (oy_2x2_start, ox_2x2_start).
            # (oy, ox) in input_pseudo_bayer_2d
            # (oy_2x2_start, ox_2x2_start) is the (0,0) position in the 2x2 output block -> Should get avg_r_pseudo
            input_pseudo_bayer_2d[oy_2x2_start, ox_2x2_start] = avg_r_pseudo

            # (oy_2x2_start, ox_2x2_start + 1) is the (0,1) position in the 2x2 output block -> Should get avg_gr_pseudo
            # FIX: Corrected placement based on RGGB pattern
            input_pseudo_bayer_2d[oy_2x2_start, ox_2x2_start + 1] = avg_gr_pseudo

            # (oy_2x2_start + 1, ox_2x2_start) is the (1,0) position in the 2x2 output block -> Should get avg_gb_pseudo
            # FIX: Corrected placement based on RGGB pattern
            input_pseudo_bayer_2d[oy_2x2_start + 1, ox_2x2_start] = avg_gb_pseudo

            # (oy_2x2_start + 1, ox_2x2_start + 1) is the (1,1) position in the 2x2 output block -> Should get avg_b_pseudo
            input_pseudo_bayer_2d[oy_2x2_start + 1, ox_2x2_start + 1] = avg_b_pseudo


            # --- Debug Logs (adjust to 8x8 context mapping to 2x2 output) ---
            # Debug logs should reflect the 8x8 input block and the 2x2 output region
            # Limit debug output to a few blocks at the start of the tile
            tile_8x8_index_y = iy_8 // 8
            tile_8x8_index_x = ix_8 // 8
            # Calculate total 8x8 blocks in the tile for progress tracking within the tile
            # total_8x8_blocks_in_tile = (TILE_SIZE // 8) * (TILE_SIZE // 8) # 64*64 = 4096
            current_8x8_block_linear_index = tile_8x8_index_y * (TILE_SIZE // 8) + tile_8x8_index_x

            if DEBUG_MODE and (current_8x8_block_linear_index < DEBUG_LOG_PATCHES): # Use DEBUG_LOG_PATCHES for 8x8 blocks
                 print(f"  [{file_name}] Tile ({tile_origin_yx[0]},{tile_origin_yx[1]}), Input 8x8 Block at ({iy_8},{ix_8}) -> Output 2x2 Region at ({oy_2x2_start},{ox_2x2_start}):")
                 print(f"    Averages from 4x4 Quadrants (R, Gr, Gb, B): {avg_r_pseudo:.4f}, {avg_gr_pseudo:.4f}, {avg_gb_pseudo:.4f}, {avg_b_pseudo:.4f}")
                 # Print the 2x2 output patch in the 128x128 input data (before adding channel dim)
                 output_2x2_patch = input_pseudo_bayer_2d[oy_2x2_start : oy_2x2_start+2, ox_2x2_start : ox_2x2_start+2]
                 print(f"    Output 2x2 Pseudo-Bayer Patch (128x128 slice):\n{output_2x2_patch}")


    # Add the channel dimension to the pseudo bayer input
    input_pseudo_bayer_128 = np.expand_dims(input_pseudo_bayer_2d, axis=-1)


    if DEBUG_MODE:
        print(f"Debug: Generated data ranges for tile at {tile_origin_yx} in {file_name}:")
        if input_pseudo_bayer_128.size > 0:
             print(f"  input_pseudo_bayer_128 range: min={np.min(input_pseudo_bayer_128):.4f}, max={np.max(input_pseudo_bayer_128):.4f}, mean={np.mean(input_pseudo_bayer_128):.4f}")
             print(f"  input_pseudo_bayer_128 shape: {input_pseudo_bayer_128.shape}")
             if np.any(np.isnan(input_pseudo_bayer_128)):
                 print("  Warning: input_pseudo_bayer_128 contains NaN values.")
        else:
             print("  input_pseudo_bayer_128 is empty.")
        if target_rgb_128.size > 0:
             print(f"  target_rgb_128 range: min={np.min(target_rgb_128):.4f}, max={np.max(target_rgb_128):.4f}, mean={np.mean(target_rgb_128):.4f}")
             print(f"  target_rgb_128 shape: {target_rgb_128.shape}")
             if np.any(np.isnan(target_rgb_128)):
                 print("  Warning: target_rgb_128 contains NaN values.")
        else:
             print("  target_rgb_128 is empty.")

    # Return the generated data and info needed for saving/displaying in the main thread
    # Return format: (input_pseudo_bayer, target_rgb, tile_origin_yx, file_name_base, raw_tile_512)

    # Check if generated data is valid (not all zeros or NaNs) before returning
    if np.any(input_pseudo_bayer_128) and not np.any(np.isnan(input_pseudo_bayer_128)) and \
       np.any(target_rgb_128) and not np.any(np.isnan(target_rgb_128)):
        return input_pseudo_bayer_128, target_rgb_128, tile_origin_yx, file_name_base, raw_tile_512
    else:
        if DEBUG_MODE:
             print(f"Debug: Generated data for {file_name_base} is None, all zeros, or contains NaNs.")
        # Return None for data but pass back identifying info and the original tile for potential logging/debugging in main thread
        return None, None, tile_origin_yx, file_name_base, raw_tile_512


# --- PySide6 GUIとデータ生成スレッド ---

class SquareImageLabel(QLabel):
    """
    QLabelを継承し、表示される画像を常に正方形領域にアスペクト比を保って描画するカスタムウィジェット
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setScaledContents(False) # QLabelの自動スケーリングを無効化
        self.setAlignment(Qt.AlignCenter) # 描画位置を中央寄せ
        self._pixmap: QPixmap = QPixmap() # 表示するQPixmapを保持

        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        # GUI表示サイズを使用
        self.setMinimumSize(GUI_DISPLAY_SIZE, GUI_DISPLAY_SIZE)


    def setPixmap(self, pixmap: QPixmap | None):
        """表示するQPixmapを設定"""
        if isinstance(pixmap, QPixmap):
            self._pixmap = pixmap
        elif pixmap is None:
            self._pixmap = QPixmap()
        else:
            if DEBUG_MODE:
                print(f"Warning: setPixmap received invalid object type: {type(pixmap)}. Expected QPixmap or None.")
            self._pixmap = QPixmap()
        self.update() # paintEventをトリガーして再描画

    def paintEvent(self, event):
        """ウィジェットの描画イベント"""
        painter = QPainter(self)
        if not self._pixmap.isNull():
            # ウィジェットのサイズ内で、描画可能な最大の正方形の辺の長さを計算
            side = min(self.width(), self.height())
            if side > 0:
                # Pixmapを指定サイズに、アスペクト比を保ってスケーリング
                scaled_pixmap = self._pixmap.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # スケーリングされた画像をウィジェットの中央に描画するための位置を計算
                x = (self.width() - scaled_pixmap.width()) // 2
                y = (self.height() - scaled_pixmap.height()) // 2

                painter.drawPixmap(x, y, scaled_pixmap)
        else:
            # Pixmapが設定されていない場合、基底クラスのpaintEventを呼び出す
            super().paintEvent(event)


class DataGeneratorWorker(QThread):
    """
    データ生成処理を行うスレッド
    """
    # シグナル定義
    # progress_updated: (current_file_index, total_files, completed_tiles_in_file, total_tiles_in_file, current_file_name)
    progress_updated = Signal(int, int, int, int, str)
    # image_updated: (label_key, image_data_np) - image_data_np は表示用の float または uint8 NumPy 配列
    image_updated = Signal(str, np.ndarray)
    # finished: ()
    finished = Signal()
    # error: (message)
    error = Signal(str)

    def __init__(self, raw_input_dir: str, output_dir: str):
        super().__init__()
        self.raw_input_dir = raw_input_dir
        self.output_dir = output_dir
        self._is_running = True
        self._process_pool = None # Hold the ProcessPoolExecutor instance

        # 出力サブディレクトリパスを事前に生成
        self.output_pseudo_bayer_dir = os.path.join(self.output_dir, OUTPUT_PSEUDO_BAYER_SUBDIR) # 更新
        self.output_target_dir = os.path.join(self.output_dir, OUTPUT_TARGET_SUBDIR)

    def stop(self):
        """スレッドの停止を要求する"""
        self._is_running = False
        print("Data generation stop requested...")
        # Shutdown the process pool if it exists
        if self._process_pool:
            # This will cancel any futures that haven't started and wait for running futures to complete
            # or be cancelled (depends on the underlying implementation and OS signal handling).
            # Setting wait=True would block, so we use False and rely on the main thread's wait in closeEvent
            # or the natural end of processing already submitted tasks.
            # cancel_futures=True attempts to stop tasks that haven't started yet.
            self._process_pool.shutdown(wait=False, cancel_futures=True)
            print("Process pool shutdown requested.")


    def run(self):
        """スレッドのメイン実行ループ"""
        try:
            # 出力ディレクトリを作成 (既にメインで作成済みだが念のため)
            os.makedirs(self.output_pseudo_bayer_dir, exist_ok=True) # 更新
            os.makedirs(self.output_target_dir, exist_ok=True)

            # RAWファイルリストを取得
            raw_extensions = ['*.arw', '*.cf2', '*.cr2', '*.crw', '*.dng', '*.erf', '*.3fr', '*.fff', '*.hdr', '*.k25', '*.kdc', '*.mdc', '*.mos', '*.mrw', '*.nef', '*.orf', '*.pef', '*.raf', '*.raw', '*.rdc', '*.sr2', '*.srf', '*.x3f']
            raw_files = []
            for ext in raw_extensions:
                raw_files.extend(glob.glob(os.path.join(self.raw_input_dir, '**', ext), recursive=True))
            raw_files.sort() # ソート

            if not raw_files:
                msg = f"Error: No supported RAW files found in '{self.raw_input_dir}'. Please place RAW files there and restart."
                print(msg)
                self.error.emit(msg)
                self.finished.emit()
                return

            total_files = len(raw_files)

            # --- Process files sequentially ---
            for file_index, raw_filepath in enumerate(raw_files):
                if not self._is_running:
                    print("Data generation stopped.")
                    break

                file_name = os.path.basename(raw_filepath)
                print(f"Processing file {file_index + 1}/{total_files}: {file_name}")

                # RAWファイルを読み込み、リニアデータとメタデータを取得 (Sequential part)
                # This part is kept sequential as it involves file I/O and rawpy processing
                raw_linear_data, bayer_pattern_indices, white_level, black_levels, pos_to_channel_map = read_raw_process_linear(raw_filepath)

                # RAW読み込みまたはパターンマッピングに失敗した場合はスキップ
                if raw_linear_data is None or pos_to_channel_map is None or len(pos_to_channel_map) != 4:
                    print(f"Skipping {file_name} due to processing errors or incomplete pattern mapping.")
                    self.progress_updated.emit(file_index + 1, total_files, 0, 0, file_name) # ファイルスキップとして進捗更新
                    continue

                raw_h, raw_w = raw_linear_data.shape

                # タイルに切り出し (TILE_SIZE = 512)
                # TILE_SIZE の倍数でない部分は切り捨てる
                num_tiles_y = raw_h // TILE_SIZE
                num_tiles_x = raw_w // TILE_SIZE
                total_tiles_in_file = num_tiles_y * num_tiles_x

                if total_tiles_in_file == 0:
                     if DEBUG_MODE:
                          print(f"Warning: No full {TILE_SIZE}x{TILE_SIZE} tiles found in {file_name}. Skipping file.")
                     self.progress_updated.emit(file_index + 1, total_files, 0, 0, file_name) # ファイルスキップとして進捗更新
                     continue

                if DEBUG_MODE:
                     print(f"Found {total_tiles_in_file} tiles ({num_tiles_y}x{num_tiles_x}) in {file_name}.")

                # --- Prepare tile tasks ---
                tile_tasks = []
                tiles_to_skip_count = 0
                for ty in range(num_tiles_y):
                    for tx in range(num_tiles_x):
                         tile_start_y = ty * TILE_SIZE
                         tile_start_x = tx * TILE_SIZE
                         # Check if output files already exist
                         base_filename = f"{os.path.splitext(file_name)[0]}_tile_{ty}_{tx}"
                         input_save_path = os.path.join(self.output_pseudo_bayer_dir, f"{base_filename}.npy")
                         target_save_path = os.path.join(self.output_target_dir, f"{base_filename}.npy")

                         if not self._is_running:
                              print("Stop requested, stopping task preparation.")
                              break # Break from tile preparation loop

                         if not os.path.exists(input_save_path) or not os.path.exists(target_save_path):
                             # If either file is missing, add to tasks
                              tile_tasks.append(((tile_start_y, tile_start_x), raw_linear_data, bayer_pattern_indices, pos_to_channel_map, file_name))
                         else:
                             if DEBUG_MODE:
                                  print(f"Debug: Output files for tile ({ty},{tx}) of {file_name} already exist. Skipping task creation.")
                             tiles_to_skip_count += 1
                             # If skipping task, update progress immediately for this tile
                             self.progress_updated.emit(file_index + 1, total_files, tiles_to_skip_count, total_tiles_in_file, file_name)

                    if not self._is_running: # Check again after inner loop
                         break

                if not self._is_running: # Check again after outer loop
                    break # Break from file loop

                # --- Parallel processing of tiles for the current file ---
                processed_tiles_count = tiles_to_skip_count # Start count with skipped tiles
                futures = {} # To store futures and their corresponding tile info

                # Initialize the process pool for this file
                # Using `with` statement handles graceful shutdown
                # Use max_workers based on CPU count
                with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as self._process_pool:
                    print(f"Starting ProcessPoolExecutor with {multiprocessing.cpu_count()} workers for {len(tile_tasks)} tiles...")

                    if not tile_tasks and tiles_to_skip_count == total_tiles_in_file:
                        # If all tiles were skipped due to existing files, update final progress for this file and continue to next.
                        if DEBUG_MODE: print(f"Debug: All tiles for {file_name} already exist. Skipping process pool for this file.")
                        self.progress_updated.emit(file_index + 1, total_files, total_tiles_in_file, total_tiles_in_file, file_name)
                        # Important: If the pool is created but no tasks are submitted, the `as_completed` loop won't run.
                        # Ensure the final progress is emitted. It was already done above for the all-skipped case.
                        continue # Go to the next file

                    for (tile_start_y, tile_start_x), current_raw_linear_data, current_bayer_pattern_indices, current_pos_to_channel_map, current_file_name in tile_tasks:
                         if not self._is_running:
                             print("Stop requested, cancelling pending tasks...")
                             # Cancel remaining futures
                             for future in futures:
                                  if not future.done():
                                       future.cancel()
                             break # Break from task submission loop

                         # Extract the tile slice to pass to the process
                         raw_tile_slice = current_raw_linear_data[tile_start_y : tile_start_y + TILE_SIZE,
                                                                  tile_start_x : tile_start_x + TILE_SIZE]

                         # Submit the tile processing task to the pool
                         # Pass necessary data for the function
                         future = self._process_pool.submit(create_data_pair_from_tile,
                                                            raw_tile_slice, # Pass the tile slice
                                                            (tile_start_y, tile_start_x), # Tile origin (global)
                                                            current_bayer_pattern_indices, # Bayer pattern indices (global)
                                                            current_pos_to_channel_map, # Channel mapping
                                                            current_file_name) # File name for logging

                         # Store the future along with tile info for later processing
                         futures[future] = (tile_start_y, tile_start_x, current_file_name)


                    # Process completed futures
                    # Iterate through futures as they complete
                    for future in concurrent.futures.as_completed(futures):
                         if not self._is_running:
                             print("Stop requested, stopping processing completed futures.")
                             break # Break from processing completed futures

                         tile_start_y, tile_start_x, current_file_name = futures[future]
                         tile_ty = tile_start_y // TILE_SIZE
                         tile_tx = tile_start_x // TILE_SIZE
                         base_filename = f"{os.path.splitext(current_file_name)[0]}_tile_{tile_ty}_{tile_tx}"
                         input_save_path = os.path.join(self.output_pseudo_bayer_dir, f"{base_filename}.npy")
                         target_save_path = os.path.join(self.output_target_dir, f"{base_filename}.npy")

                         try:
                             # Get result from the completed future
                             # Result is (input_pseudo_bayer_128, target_rgb_128, tile_origin_yx, file_name_base, raw_tile_512)
                             result = future.result()

                             # Check if generated data is valid before saving and updating GUI
                             # create_data_pair_from_tile now returns (None, None, ...) if data is invalid
                             if result is not None and result[0] is not None and result[1] is not None:
                                 input_data, target_data, _, _, raw_tile_display = result

                                 # Save the generated data
                                 try:
                                     np.save(input_save_path, input_data)
                                     np.save(target_save_path, target_data)
                                     # if DEBUG_MODE: print(f"Saved {base_filename}.npy") # Too verbose
                                 except Exception as e:
                                     print(f"Error saving data pair for {base_filename}: {e}")
                                     if DEBUG_MODE: traceback.print_exc()

                                 # Emit image update signals for GUI (using the latest processed tile)
                                 # Note: This will only show the last tile processed by the pool as it completes.
                                 # It doesn't guarantee showing tiles in order, but updates the display periodically.
                                 # Need the raw tile data for the first display, which was also returned
                                 self.image_updated.emit("Original Tile", np.clip(raw_tile_display * 255.0, 0, 255).astype(np.uint8)) # Convert back to uint8 for display
                                 self.image_updated.emit("Simulated RGBG", input_data) # pseudo-bayer input is float32 HxWx1
                                 self.image_updated.emit("Demosaiced RGB", target_data) # target RGB is float32 HxWx3

                             else:
                                  if DEBUG_MODE:
                                       print(f"Debug: Generated data for {base_filename} is None or invalid. Skipping saving and display for this tile.")


                         except concurrent.futures.CancelledError:
                              print(f"Task for tile ({tile_ty},{tile_tx}) of {current_file_name} was cancelled.")
                         except Exception as exc:
                             print(f'Error processing tile ({tile_ty},{tile_tx}) of {current_file_name}: {exc}')
                             if DEBUG_MODE: traceback.print_exc()
                             # Optionally emit an error signal for this specific tile, or just log

                         # Update progress for the completed (or failed/skipped) tile
                         processed_tiles_count += 1
                         # Ensure current_file_name is passed correctly
                         self.progress_updated.emit(file_index + 1, total_files, processed_tiles_count, total_tiles_in_file, file_name)


                # Process pool finishes for this file
                # The 'with' block automatically calls shutdown()

                # Clean up the process pool reference after the 'with' block exits
                self._process_pool = None

                # Ensure final progress is updated for this file, even if tasks were cancelled
                # This might happen if stop() was called during processing tiles for a file.
                # Also handle the case where processed_tiles_count is already equal to total_tiles_in_file
                # because all tiles were skipped. The progress was already emitted then.
                if processed_tiles_count < total_tiles_in_file:
                     if DEBUG_MODE: print(f"Debug: Completed processing all submitted tasks for {file_name}. Final tile count: {processed_tiles_count}/{total_tiles_in_file}.")
                     # Emit the final progress for this file if it wasn't fully processed
                     self.progress_updated.emit(file_index + 1, total_files, processed_tiles_count, total_tiles_in_file, file_name)
                elif processed_tiles_count == total_tiles_in_file and DEBUG_MODE:
                     print(f"Debug: All tiles for {file_name} processed or skipped. Final tile count: {processed_tiles_count}/{total_tiles_in_file}.")


            # All files processed or stop requested
            if self._is_running:
                 print("Data generation loop finished.")
            else:
                 print("Data generation loop interrupted by stop request.")


        except Exception as e:
            msg = f"An unexpected error occurred during data generation: {e}"
            print(msg)
            if DEBUG_MODE: traceback.print_exc()
            self.error.emit(msg)

        self.finished.emit() # 処理完了シグナルを発行 (or interruption signal)


class DataGeneratorGUI(QMainWindow):
    """
    データ生成用GUIメインウィンドウ
    """
    def __init__(self, raw_input_dir: str, output_dir: str):
        super().__init__()
        self.setWindowTitle("RAW Data Pair Generator for Demosaic")
        self.setGeometry(100, 100, 1000, 700) # ウィンドウサイズ調整

        self.raw_input_dir = raw_input_dir
        self.output_dir = output_dir

        # --- GUIコンポーネント ---
        self.status_label = QLabel("Initializing...", alignment=Qt.AlignCenter)
        self.status_label.setFixedHeight(20)

        # 画像表示用のラベル (サイズ表示を更新)
        self.label_original_tile = QLabel(f"Original RAW Tile ({TILE_SIZE}x{TILE_SIZE} Normalized uint8)", alignment=Qt.AlignCenter)
        self.label_simulated_rgbg = QLabel(f"Simulated Pseudo-Bayer ({OUTPUT_PATCH_SIZE}x{OUTPUT_PATCH_SIZE} Pseudo-color)", alignment=Qt.AlignCenter) # テキスト更新
        self.label_demosaiced_rgb = QLabel(f"Demosaiced RGB ({OUTPUT_PATCH_SIZE}x{OUTPUT_PATCH_SIZE} Gamma Corrected Display)", alignment=Qt.AlignCenter)

        # 画像表示領域 (カスタムクラスを使用)
        self.image_original_tile = SquareImageLabel()
        self.image_simulated_rgbg = SquareImageLabel()
        self.image_demosaiced_rgb = SquareImageLabel()

        # 各画像表示領域にラベルと画像をまとめるウィジェット
        def create_image_widget(label: QLabel, image_label: SquareImageLabel) -> QWidget:
            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(label)
            layout.addWidget(image_label)
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(5)
            widget.setLayout(layout)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # GUI表示サイズを使用
            widget.setMinimumSize(GUI_DISPLAY_SIZE + 10, GUI_DISPLAY_SIZE + 30) # ラベルの高さなどを考慮して少し大きめに
            return widget

        # 画像表示を横に並べるレイアウト
        image_layout = QHBoxLayout()
        image_layout.addWidget(create_image_widget(self.label_original_tile, self.image_original_tile))
        image_layout.addWidget(create_image_widget(self.label_simulated_rgbg, self.image_simulated_rgbg))
        image_layout.addWidget(create_image_widget(self.label_demosaiced_rgb, self.image_demosaiced_rgb))
        image_layout.setSpacing(10)

        # メインレイアウト
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(image_layout)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- データ生成スレッド ---
        self.worker_thread = DataGeneratorWorker(self.raw_input_dir, self.output_dir)

        # --- シグナルとスロットの接続 ---
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.image_updated.connect(self.update_image_display)
        self.worker_thread.finished.connect(self.on_generation_finished)
        self.worker_thread.error.connect(self.show_error_message)


        # --- 処理開始 ---
        # アプリケーション起動時に自動で処理を開始
        self.worker_thread.start()
        self.status_label.setText("Starting data generation...")


    @Slot(int, int, int, int, str)
    def update_progress(self, current_file_index: int, total_files: int, completed_tiles_in_file: int, total_tiles_in_file: int, current_file_name: str):
        """データ生成進捗に応じてステータスラベルを更新"""
        if total_tiles_in_file > 0:
             status_text = f"Processing File {current_file_index}/{total_files}: {os.path.basename(current_file_name)} | Tile {completed_tiles_in_file}/{total_tiles_in_file}"
        else:
             # Case where no full tiles were found in the current file
             status_text = f"Processing File {current_file_index}/{total_files}: {os.path.basename(current_file_name)} | No full tiles found." # Indicate 0/0 skipped
        self.status_label.setText(status_text)

    @Slot(str, np.ndarray)
    def update_image_display(self, label_key: str, image_data_np: np.ndarray):
        """
        データ生成スレッドから送られてきた画像をGUIに表示

        Args:
            label_key: 画像の種類を示す文字列 ("Original Tile", "Simulated RGBG", "Demosaiced RGB")
            image_data_np: 表示する numpy 配列 (float or uint8, HWC or HW) - 高ビットリニアデータはfloatで渡される想定
        """
        # Determine if this is the pseudo-bayer input for special handling
        # ラベルキー "Simulated RGBG" は、Worker側から送られる疑似ベイヤーデータを示す
        is_pseudo_bayer_input = (label_key == "Simulated RGBG")

        # Display data processing varies by key
        display_data_np = image_data_np

        if label_key == "Demosaiced RGB":
            # 教師データ（デモザイク済みRGB）は高ビットリニア(0-1 float)で渡される想定
            # GUI表示のため、ガンマ補正を適用して見やすくする
            gamma = 2.2
            # float32データに対して直接ガンマ補正 (value^(1/gamma))
            # np.power は 0**x = 0 を適切に処理します
            display_data_np = np.power(image_data_np, 1.0 / gamma)
            # ガンマ補正後も0-1の範囲にクリップ（念のため）
            display_data_np = np.clip(display_data_np, 0.0, 1.0)
            if DEBUG_MODE: print(f"Debug: Applied gamma correction ({gamma}) for Demosaiced RGB display.") # デバッグ用

        # For "Original Tile", image_data_np is already uint8.
        # For "Simulated RGBG" (pseudo-bayer input), image_data_np is float32 128x128x1.
        # For "Demosaiced RGB", display_data_np is float32 HxWx3 (gamma corrected, usually 128x128x3).
        # The convert_np_to_qpixmap_display function handles the conversion to uint8 for display.

        # Call the converter with the appropriate flag
        pixmap = convert_np_to_qpixmap_display(display_data_np, GUI_DISPLAY_SIZE, is_pseudo_bayer=is_pseudo_bayer_input)

        if label_key == "Original Tile":
            self.image_original_tile.setPixmap(pixmap)
        elif label_key == "Simulated RGBG":
            self.image_simulated_rgbg.setPixmap(pixmap)
        elif label_key == "Demosaiced RGB":
            self.image_demosaiced_rgb.setPixmap(pixmap)
        # 他のキーは無視


    @Slot()
    def on_generation_finished(self):
        """データ生成完了時に呼び出される"""
        print("Data generation finished signal received.")
        self.status_label.setText("Data Generation Finished.")

        # スレッドを安全に終了
        if self.worker_thread and self.worker_thread.isRunning():
             # stop()が呼ばれていない場合はここで呼ばれるべきだが、closeEventでも呼んでいるので念のため
             if self.worker_thread._is_running: # Check internal flag if stop() wasn't called externally
                  self.worker_thread.stop()
             # The worker thread should exit its run loop after stop() is called
             # and the process pool is shut down.
             # We don't need a hard wait here as closeEvent will handle it.
             pass


    @Slot(str)
    def show_error_message(self, message: str):
        """エラーメッセージを表示"""
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText(f"Error: {message}")


    def closeEvent(self, event):
        """ウィンドウを閉じるときのイベントハンドラ"""
        if self.worker_thread and self.worker_thread.isRunning():
            print("Attempting to stop data generation thread...")
            self.worker_thread.stop() # スレッドに停止を要求 (also shuts down process pool)
            # スレッドが終了するまで最大10秒待つ。終了しない場合はそのままGUI終了。
            # Process pool shutdown might take some time, especially if many tasks are running. Increased wait time.
            if not self.worker_thread.wait(10000): # wait returns True if thread finished, False otherwise
                 print("Warning: Data generation thread did not stop cleanly within 10 seconds. Forcing application quit.")
                 # Force quit the application if the worker thread doesn't stop,
                 # otherwise the process pool might keep the application alive.
                 QApplication.quit() # Use QApplication.quit() to exit the event loop
                 event.ignore() # Ignore the close event to prevent immediate window close
                 return # Do not call super().closeEvent yet

        # If the worker thread stopped cleanly or wasn't running, proceed with closing the window.
        super().closeEvent(event)


if __name__ == "__main__":
    # Ensure multiprocessing starts cleanly on some platforms
    # https://docs.python.org/en/3/library/multiprocessing.html#multiprocessing.freeze_support
    # On Windows, and sometimes on macOS using the 'spawn' or 'forkserver' start methods
    # is recommended for robustness, especially with GUI applications and shared resources.
    # The default method varies by OS ('fork' on Linux, 'spawn' on Windows, 'fork' or 'spawn' on macOS).
    # Explicitly setting 'spawn' can improve reliability but might have slight overhead.
    # Let's add a check for the current start method and potentially set it if not 'spawn'.
    # Note: set_start_method must be called early in the script, preferably in the __main__ block.

    try:
        # Try to get the current start method. If none is set, this might raise RuntimeError.
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
             print(f"Current multiprocessing start method is '{current_start_method}'. Setting to 'spawn' for potential robustness.")
             # Attempt to set 'spawn'. This will fail if a start method has already been set (e.g., by an imported library).
             # Use force=True to override, but be cautious as it might cause issues if other libraries relied on the previous method.
             # Let's try without force=True first.
             try:
                 multiprocessing.set_start_method('spawn')
                 print("Multiprocessing start method set to 'spawn'.")
             except RuntimeError as e:
                  print(f"Warning: Could not set multiprocessing start method to 'spawn'. It might be already set or not supported: {e}")
                  print("Proceeding with the default or existing start method. Data generation might be less stable on some platforms.")
    except Exception as e:
         print(f"Warning: Could not determine or set multiprocessing start method: {e}")
         print("Proceeding with the default or existing start method. Data generation might be less stable on some platforms.")


    # multiprocessing.freeze_support() # Needed if freezing the application into an executable

    parser = argparse.ArgumentParser(description="Generate high-bit linear data pairs for demosaic training.")
    parser.add_argument("--input_raw_dir", type=str, default=DEFAULT_RAW_INPUT_DIR,
                        help=f"Directory containing input RAW files (default: {DEFAULT_RAW_INPUT_DIR})")
    parser.add_argument("--output_data_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save generated training data pairs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")


    args = parser.parse_args()

    # コマンドライン引数でDEBUG_MODEを設定
    if args.debug:
        DEBUG_MODE = True # Corrected setting
        print("Debug mode is enabled.")
    else:
        DEBUG_MODE = False # Ensure it's False if --debug is not passed
        print("Debug mode is disabled.")


    # 入力ディレクトリが存在するか確認
    if not os.path.isdir(args.input_raw_dir):
        print(f"Error: Input RAW directory not found at '{args.input_raw_dir}'.")
        print(f"Please create the directory and place your RAW files inside, then run the script again.")
        sys.exit(1) # エラー終了

    # 出力ディレクトリが存在しない場合は作成
    output_pseudo_bayer_dir = os.path.join(args.output_data_dir, OUTPUT_PSEUDO_BAYER_SUBDIR) # 更新
    output_target_dir = os.path.join(args.output_data_dir, OUTPUT_TARGET_SUBDIR)
    os.makedirs(output_pseudo_bayer_dir, exist_ok=True) # 更新
    os.makedirs(output_target_dir, exist_ok=True)
    print(f"Saving generated data to: {args.output_data_dir}")


    app = QApplication(sys.argv)
    main_window = DataGeneratorGUI(args.input_raw_dir, args.output_data_dir)
    main_window.show()
    sys.exit(app.exec())