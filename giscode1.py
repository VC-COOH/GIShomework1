import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon

def read_game_map(image_path):
    """读取游戏地图（RGB格式）"""
    print(f"尝试读取图像: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_rock_contours(image, rock_hsv_lower=(10, 30, 50), rock_hsv_upper=(30, 100, 150)):
    """提取岩石轮廓线"""
    # 1. 颜色空间转换
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 2. 岩石颜色分割
    rock_mask = cv2.inRange(hsv, np.array(rock_hsv_lower), np.array(rock_hsv_upper))
    
    # 3. 形态学操作优化掩码
    kernel = np.ones((5, 5), np.uint8)
    rock_mask = cv2.morphologyEx(rock_mask, cv2.MORPH_CLOSE, kernel)
    
    # 4. 轮廓提取（只提取外部轮廓）
    contours, _ = cv2.findContours(rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"检测到 {len(contours)} 个岩石轮廓")
    
    return {
        "original": image,
        "rock_mask": rock_mask,
        "contours": contours
    }

def save_raster_contours(data, output_path):
    """保存栅格格式的轮廓图"""
    try:
        # 创建空白图像
        contour_img = np.zeros_like(data["original"])
        
        # 在空白图像上绘制轮廓
        cv2.drawContours(contour_img, data["contours"], -1, (255, 255, 255), 2)
        
        # 保存图像
        cv2.imwrite(output_path, cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))
        print(f"✅ 栅格轮廓图已保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存栅格轮廓图失败: {e}")

def save_raster_overlay(data, output_path):
    """保存轮廓叠加在原图上的栅格图"""
    try:
        # 在原图上绘制轮廓
        overlay_img = data["original"].copy()
        cv2.drawContours(overlay_img, data["contours"], -1, (0, 0, 255), 2)
        
        # 保存图像
        cv2.imwrite(output_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        print(f"✅ 栅格叠加图已保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存栅格叠加图失败: {e}")

def save_vector_contours(data, output_path):
    """保存矢量格式的轮廓图（GeoJSON）"""
    try:
        # 转换为GeoDataFrame
        geometries = []
        for contour in data["contours"]:
            if len(contour) >= 3:
                points = [(p[0][0], p[0][1]) for p in contour]
                geometries.append(Polygon(points))
                
        if not geometries:
            print("⚠️ 没有有效的轮廓数据，跳过矢量数据保存")
            return
            
        # 使用像素坐标的CRS
        gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"✅ 矢量轮廓数据已保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存矢量轮廓数据失败: {e}")

# -------------------------- 主流程 --------------------------
def main(game_map_path, output_dir="game_output"):
    """完整处理流程：读取→提取轮廓→保存栅格+矢量数据"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")

        # 1. 读取图像
        print("🔍 读取游戏地图...")
        game_map = read_game_map(game_map_path)
        
        # 打印图像尺寸信息
        height, width = game_map.shape[:2]
        print(f"图像尺寸: {width}x{height} 像素")

        # 2. 提取岩石轮廓
        print("🛠️ 提取岩石轮廓...")
        data = extract_rock_contours(game_map)

        # 3. 保存栅格轮廓图
        raster_path = os.path.join(output_dir, "rock_contours_raster.png")
        save_raster_contours(data, raster_path)

        # 4. 保存栅格叠加图
        overlay_path = os.path.join(output_dir, "rock_contours_overlay.png")
        save_raster_overlay(data, overlay_path)

        # 5. 保存矢量轮廓数据
        vector_path = os.path.join(output_dir, "rock_contours_vector.geojson")
        save_vector_contours(data, vector_path)

        print("\n=== 处理完成 ===")
        print(f"🖼️ 栅格轮廓图: {raster_path}")
        print(f"🖼️ 栅格叠加图: {overlay_path}")
        print(f"📏 矢量轮廓数据: {vector_path}")

    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")

if __name__ == "__main__":
    # 这里的路径改成自己的
    GAME_MAP_PATH = "your path"
    main(GAME_MAP_PATH)