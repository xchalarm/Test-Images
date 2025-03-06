import pandas as pd
import numpy as np

# โหลดข้อมูลจากไฟล์ Instagram
df = pd.read_csv('dataset/data_from_intragram.csv')

# สร้างคอลัมน์ใหม่สำหรับการสุ่มสลับ
df['swap'] = np.random.choice([0, 1], size=len(df))

# สลับคอลัมน์ Image1 และ Image2 ตามเงื่อนไข
df.loc[df['swap'] == 1, ['Image 1', 'Image 2']] = df.loc[df['swap'] == 1, ['Image 2', 'Image 1']].values

# ปรับค่าในคอลัมน์ Winner ตามการสลับ
df.loc[df['swap'] == 1, 'Winner'] = df.loc[df['swap'] == 1, 'Winner'].apply(lambda x: 2 if x == 1 else 1)

# ลบคอลัมน์ swap ที่ไม่จำเป็นออก
df.drop('swap', axis=1, inplace=True)

# โหลดข้อมูลจากไฟล์ Questionnaire
df2 = pd.read_csv("dataset/data_from_questionaire.csv")

# รวมไฟล์ DataFrame ทั้งสอง
combined_df = pd.concat([df, df2], ignore_index=True)

# สุ่มเรียงลำดับข้อมูลใน DataFrame ที่รวมแล้ว
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

# บันทึกผลลัพธ์ไปยังไฟล์ CSV ใหม่
shuffled_df.to_csv('shuffled_new.csv', index=False)
