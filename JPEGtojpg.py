import os

# img_dir = ('/home/lihanyu/datasets/VID_VOC/VOC2007/JPEGImages/ILSVRC2015_VID_train_0002')
#
# for root, dirs, files in os.walk(img_dir):
#     for f_name in files:
#         newname = f_name
#         newname = newname.split(".")
#         if newname[-1]=="JPEG":
#             newname[-1]="jpg"
#             newname = str.join(".",newname)  #这里要用str.join
#             filename = root+'/'+f_name
#             newname = root+'/'+newname
#             os.rename(filename,newname)
#             print(newname,"updated successfully")

'''
顺序存放到trainval.txt中
'''
# img_dir = ('/home/luoqibin/pycodes/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/JPEGImages')
# w_img=os.path.join('/home/luoqibin/pycodes/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main','train_part.txt')
# with open(w_img,'w+') as fw:
#     path_list=os.listdir(img_dir)
#     path_list.sort()
#     #print(path_list)
#     for i in range(len(path_list)):
#         if "ILSVRC" in path_list[i]:
#             dir=os.path.join(img_dir,path_list[i])
#             img_list=os.listdir(dir)
#             img_list.sort()
#             for j in range(len(img_list)):
#                 img_num=img_list[j].split('.')[0]
#                 fw.write(os.path.join(path_list[i],img_num))
#                 fw.write('\n')
#                 print(os.path.join(path_list[i],img_num))
            #print(img_list)

    # for root,dirs,files in os.walk(img_dir):
    #     for f_name in files:
    #         fw.write(root.split('JPEGImages/')[-1]+'/'+f_name.split('.')[0]+'\n')

'''
选取部分视频
'''
trainval_path="/home/luoqibin/pycodes/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main/original_trainval.txt"
trainval_part="/home/luoqibin/pycodes/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval_part.txt"
i=0
with open(trainval_part, 'a') as wr:
    with open(trainval_path,'r') as re:
        for line in re.readlines():
            print(line)
            if i<=20000:
                i+=1
                wr.write(line)


