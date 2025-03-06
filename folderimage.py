import pandas as pd
import shutil
import os

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('shuffled_new.csv')

# โฟลเดอร์ต้นทาง (สมมติว่ามีหลายโฟลเดอร์และคุณรู้โครงสร้าง)
source_folders = ['dataset/Intragram Images [Original]/Burger', 'dataset/Intragram Images [Original]/Dessert', 'dataset/Intragram Images [Original]/Pizza', 'dataset/Intragram Images [Original]/Ramen', 'dataset/Intragram Images [Original]/Sushi', 'dataset/Questionair Images']

# โฟลเดอร์ปลายทาง
destination_folder = 'dataset/merge'

# ตรวจสอบว่าโฟลเดอร์ปลายทางมีอยู่แล้วหรือไม่ ถ้าไม่มีให้สร้าง
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# อ่านแต่ละแถวใน DataFrame
for index, row in df.iterrows():
    # วนลูปตามโฟลเดอร์ต้นทาง
    for source_folder in source_folders:
        image1_path = os.path.join(source_folder, row['Image 1'])
        image2_path = os.path.join(source_folder, row['Image 2'])

        # ตรวจสอบว่าไฟล์รูปภาพมีอยู่จริง และคัดลอกไฟล์ไปยังโฟลเดอร์ปลายทาง
        if os.path.exists(image1_path):
            shutil.copy(image1_path, destination_folder)
        if os.path.exists(image2_path):
            shutil.copy(image2_path, destination_folder)
