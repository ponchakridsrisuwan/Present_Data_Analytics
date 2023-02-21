import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


st.title("Data Analytics")
st.markdown(
    """
    สมาชิก
    - 63114540197 นายวรพล สุนทร 
    - 63114540210 นายวันเจริญ อุปมัย
    - 63114540424 นายพลชกฤษณ์ ศรีสุวรรณ์
    - 63114540554 นายฉัตรชัย แก้วฉุย
    """
)
st.markdown(
    """
    ข้อมูลจาก : https://www.traffy.in.th/?page_id=27351 
    - ระหว่าง : 1 กันยายน - 31 ตุลาคม 2565 
    - จำนวนข้อมูล : 30,000 แถว 20 คอลัมน์
    - Data Cleansing เหลือข้อมูล : 18,539 แถว 9 คอลัมน์ 
    - เทคนิคที่ใช้ในการ Data Cleansing : ลบค่า Missing value, ตัดคำ หรือ Tokenize, ลบ Emoji, ลบ Demoji ฯลฯ
    """)
st.markdown(
    """
    เป้าหมาย : 
    - หาประเภทการร้องเรียน จากประโยคที่ผู้ใช้กรอกเข้ามา เนื่องจากบางคนระบุประเภทและประโยคที่กรอกเข้ามาไม่ตรงกัน
    - Time Series Forecasting เพื่อหาและทำนายประเภทที่ผู้ใช้ร้องเรียนในแต่ละช่วงเวลา
    """)
st.markdown(
    """
    ประโยชน์ : 
    - ผู้บริหารหรือผู้ที่เกี่ยวข้อง สามารถดูได้ว่ามีคนร้องเรียนเรื่องใดในแต่ละช่วงเวลาใดบ้าง ช่วยให้สามารถวางแผนป้องกันและรับมือกับปัญหาได้ทันท่วงที
    """)

image = Image.open('index.png')
st.image(image, caption='count by type')

# time_series = pd.read_csv("./data.csv")
# plt.figure(figsize=(36,8))
# for column in time_series.columns:
#     plt.plot(time_series.index, time_series[column], label=column)
# plt.legend(loc='best')
# plt.title('Counts by Type', fontdict={'fontsize': 16, 'fontweight' : 5, 'color' : 'Black'})
# plt.xticks(rotation = 90, fontweight="bold")
# plt.rc('font', family='TH Sarabun New')
# plt.show



option = st.selectbox(
    'กรุณาเลือก ?',
    ('กลิ่น', 'กีดขวาง', 'ขยะ', 'จราจร', 'ชำรุด', 'ต้นไม้', 'ถนน', 'ทางเดิน', 'ท่อระบายน้ำ', 'น้ำท่วม', 'ป้าย', 'ร้องเรียน', 'สะพาน', 'เสียง'))

st.write('คุณเลือก :', option)

col1, col2, col3 = st.columns([1,6,1])

if option == "กลิ่น":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('กลิ่น.png')
        st.image(image, caption='กลิ่น')

    with col3:
        st.write(' ')

if option == "กีดขวาง":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('กีดขวาง.png')
        st.image(image, caption='กีดขวาง')

    with col3:
        st.write(' ')

if option == "ขยะ":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ขยะ.png')
        st.image(image, caption='ขยะ')

    with col3:
        st.write(' ')

if option == "จราจร":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('จราจร.png')
        st.image(image, caption='จราจร')

    with col3:
        st.write(' ')

if option == "ชำรุด":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ชำรุด.png')
        st.image(image, caption='ชำรุด')

    with col3:
        st.write(' ')

if option == "ต้นไม้":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ต้นไม้.png')
        st.image(image, caption='ต้นไม้')

    with col3:
        st.write(' ')

if option == "ถนน":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ถนน.png')
        st.image(image, caption='ถนน')

    with col3:
        st.write(' ')

if option == "ท่อระบายน้ำ":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ท่อระบายน้ำ.png')
        st.image(image, caption='ท่อระบายน้ำ')

    with col3:
        st.write(' ')

if option == "ทางเดิน":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ทางเดิน.png')
        st.image(image, caption='ทางเดิน')

    with col3:
        st.write(' ')

if option == "น้ำท่วม":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('น้ำท่วม.png')
        st.image(image, caption='น้ำท่วม')

    with col3:
        st.write(' ')

if option == "ป้าย":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ป้าย.png')
        st.image(image, caption='ป้าย')

    with col3:
        st.write(' ')

if option == "ร้องเรียน":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('ร้องเรียน.png')
        st.image(image, caption='ร้องเรียน')

    with col3:
        st.write(' ')

if option == "สะพาน":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('สะพาน.png')
        st.image(image, caption='สะพาน')

    with col3:
        st.write(' ')

if option == "เสียง":
    with col1:
        st.write(' ')

    with col2:
        image = Image.open('เสียง.png')
        st.image(image, caption='เสียง')

    with col3:
        st.write(' ')


# df_index = pd.read_csv("./data_present.csv")
# df_index['date'] = pd.to_datetime(df_index['date'], format='%Y-%m-%d %H:%M:%S')
# df_index = df_index.set_index('date')

# Check for missing values
# print(df_index.isnull().sum())

# Fill missing values with the mean of the column
# df_index = df_index.fillna(df_index.mean())

# Drop any rows that still contain missing values
# df_index = df_index.dropna()

# Filter for 'ถนน'
# df_filtered = df_index.loc[df_index['type'] == 'ถนน']

# train_size = int(len(df_filtered) * 0.8)
# train, test = df_filtered[:train_size], df_filtered[train_size:]

# train_labels = train.index.strftime('%Y-%m-%d')
# train = train.values.flatten()
# test_labels = test.index.strftime('%Y-%m-%d')
# test = test.values.flatten()

# model = ARIMA(train, order=(2,1,0))
# model_fit = model.fit(disp=0)

# predictions = model_fit.forecast(steps=len(test))[0]

# mse = mean_squared_error(test, predictions)
# rmse = np.sqrt(mse)
# mae = np.mean(np.abs(predictions - test))

# plt.plot(train_labels, train, label='Train')
# plt.plot(test_labels, test, label='Test')
# plt.plot(test_labels, predictions, label='Predictions')
# plt.title(f'ARIMA forecast (RMSE={rmse:.2f}, MAE={mae:.2f})')
# plt.legend()
# plt.xticks(test_labels[::10], rotation=45)
# plt.show()
