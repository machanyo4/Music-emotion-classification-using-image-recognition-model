import os
import shutil
import re

# データセットの場所

dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/"

# ファイル数カウント関数
def count_file(folder_path):

  import pathlib
  initial_count = 0
  for path in pathlib.Path(folder_path).iterdir():
    if path.is_file():
      initial_count += 1

  return(initial_count)

# audio_list の取得

def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio

# ディレクトリ作成
if not (os.path.exists(dataset_path + "/renamed")):
    os.mkdir(dataset_path + "/renamed")
if not (os.path.exists(dataset_path + "/renamed/Q1")):
    os.mkdir(dataset_path + "/renamed/Q1")
if not (os.path.exists(dataset_path + "/renamed/Q2")):
    os.mkdir(dataset_path + "/renamed/Q2")
if not (os.path.exists(dataset_path + "/renamed/Q3")):
    os.mkdir(dataset_path + "/renamed/Q3")
if not (os.path.exists(dataset_path + "/renamed/Q4")):
    os.mkdir(dataset_path + "/renamed/Q4")

# Q1データ
Q1_filelist = path_to_audiofiles(dataset_path + "/Q1")

for file in Q1_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q1/"+genre +"." + songname)


# shutil.copyfile("file", "./test2.txt")
# Q2データ
Q2_filelist = path_to_audiofiles(dataset_path + "/Q2")

for file in Q2_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q2/"+genre +"." + songname)

# Q3データ
Q3_filelist = path_to_audiofiles(dataset_path + "/Q3")

for file in Q3_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q3/"+genre +"." + songname)

# Q4データ
Q4_filelist = path_to_audiofiles(dataset_path + "/Q4")

for file in Q4_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q4/"+genre +"." + songname)


# How many datas?    
print("[INFO] Datas in renamed/Q1: ", end='')
print(count_file(dataset_path + "/renamed/Q1"))
print("[INFO] Datas in renamed/Q2: ", end='')
print(count_file(dataset_path + "/renamed/Q2"))
print("[INFO] Datas in renamed/Q3: ", end='')
print(count_file(dataset_path + "/renamed/Q3"))
print("[INFO] Datas in renamed/Q4: ", end='')
print(count_file(dataset_path + "/renamed/Q4"))