import os

import glob


path = "/home/yachao-li/Downloads/real_videos-20190203T063545Z-001/real_videos/pixabay/"
os.system("which ffmpeg")
for i,vi in enumerate(sorted(os.listdir(path))):
    print(vi)
    print("ffmpeg -i " + path + vi + " -vf fps=3 " +
              "/home/yachao-li/Downloads/real_videos_mariyama/" + str(i) + "%d.jpg")
    os.system("ffmpeg -i " + path + vi + " -vf fps=3 " +
              "/home/yachao-li/Downloads/real_videos_mariyama/" + str(i) + "%d.jpg")