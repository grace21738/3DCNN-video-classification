import os
import csv
import pandas as pd

 

# 指定要查詢的路徑

yourPath = '.\data'
'''

# 列出指定路徑底下所有檔案(包含資料夾)

allFileList = os.listdir(yourPath)

# 逐一查詢檔案清單

for file in allFileList:

#   這邊也可以視情況，做檔案的操作(複製、讀取...等)

#   使用isdir檢查是否為目錄

#   使用join的方式把路徑與檔案名稱串起來(等同filePath+fileName)

  if os.path.isdir(os.path.join(yourPath,file)):

    print("I'm a directory: " + file)

#   使用isfile判斷是否為檔案

  elif os.path.isfile(yourPath+file):

    print(file)

  else:

    print('OH MY GOD !!')

 '''

# 與listdir不同的是，listdir只是將指定路徑底下的目錄和檔案列出來

# walk的方式則會將指定路徑底下所有的目錄與檔案都列出來(包含子目錄以及子目錄底下的檔案)

allList = os.walk(yourPath)

# 列出所有子目錄與子目錄底下所有的檔案
with open('train.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入一列資料
    writer.writerow(['video_name', 'label'])
    
with open('test.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入一列資料
    writer.writerow(['video_name'])

for root, dirs, files in allList:

#   列出目前讀取到的路徑

  print("path：", root)
  
  print( type(root) )
  tmp = root.split("\\")
  print('tmp',tmp)

#   列出在這個路徑下讀取到的資料夾(第一層讀完才會讀第二層)

  print("directory：", dirs)

#   列出在這個路徑下讀取到的所有檔案

  print("file：", files)
  print("file length：", len(files))
  if len(tmp) > 3:
      if tmp[2] == 'train':
          for file in files:
              with open('train.csv', 'a', newline='') as csvfile:
                  writer = csv.writer(csvfile)
                  writer.writerow([file,tmp[3]])
  elif len(tmp) == 3:
      for file in files:
          with open('test.csv', 'a', newline='') as csvfile:
              writer = csv.writer(csvfile)
              writer.writerow([file])
      
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")
'''
# 寫入csv 檔案
# 開啟輸出的 CSV 檔案
with open('train.csv', 'w', newline='') as csvfile:
  # 建立 CSV 檔寫入器
  writer = csv.writer(csvfile)

  # 寫入一列資料
  writer.writerow(['video_name', 'label'])

  # 寫入另外幾列資料
  writer.writerow(['令狐沖', 175, 60])
  writer.writerow(['岳靈珊', 165, 57])
    


  train_df = pd.read_csv("train.csv")
  test_df = pd.read_csv("test.csv")

  print(f"Total videos for training: {len(train_df)}")
  print(f"Total videos for testing: {len(test_df)}")

 '''