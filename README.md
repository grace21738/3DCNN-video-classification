# 3DCNN-video-classification

## 訓練概要

有 30000 部人體姿勢影片並需將其分類，共有 39 類。 最後需針對 10000 部測試資料分類

+ 資料集: 影片長寬不固定，8 FPS 且 都小於等於 10 秒
+ 最後輸出格式為針對 10000 部測試資料預測結果分類的 csv 檔( 格式如下 )
  + name, label
  + 00001.mp4, 23
  + 00002.mp4,0
  + ...

## 訓練模型方式

### 前處理

+ 將 Video 切成好幾個 frame ，並且每 8 幀取一幀，總共取 30 幀，若影片未能取滿 30 幀則重複取最後一幀。
+ 將取的幀切割中央的部分且 resize 成 50*50。
+ 針對取的幀的 RGB 做正規化。

### 開始訓練

+ 使用 keras Convolution3D 來拆解 影片特徵，再加上影片有連續性，故使用 3D 來做訓練。
+ 在每個 Convolution3D 之間做 batch normalization，讓數值比例較為相近方便未來運算和訓練。
+ 基本上訓練 8 個模型，最後再使用 Model Ensemble 的方法預測 test 影片在不同參數的模型下的分類，並做投票取最高票的預測。

## 如何訓練模型

1.  建立 data 資料夾，並且 train 和 test 資料夾都在此 data 資料夾中( data資料夾必須和data_csv.py 、 train_color.py、predict.py在同個目錄 )

2. 一開始先執行 **data_csv.py** ，將 train 的 video name 和 label 存成 csv 檔案，test 的 video name 同時也儲存成csv 檔案

   ```
   python3 data_csv.py
   ```

3. 將 train 中的所有影片放在 train 資料夾中( 不會放在分類的資料夾裡，從分類資料夾中拉出來至train資料夾中 ) 

   > train_color.py 中:
   >
   > root_dir 可以更改放置 training 影片的位置
   >
   > root_test_dir 可以更改放置 testing 影片的位置

4. 接著執行 **train_color.py** ，需訓練 10 個模型並且此10個模型的權重會儲存在 Train_nmodel 資料夾中

   > 這裡會訓練 10 個 model NMODEL 預設為10 (可更改)

   ```
   python3 train_color.py
   ```

## 預測模型

1. 執行 **predict.py**，執行完成後會有一個 prediction.csv 檔案

   > 這裡會做 model ensemble ， NMODEL 預設為 10 (可更改)。

   ```python
   python3 predict.py
   ```

   



