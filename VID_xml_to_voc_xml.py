# coding=utf-8

import os
import os.path
import xml.dom.minidom

classes=['n02691156','n02419796', 'n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227' ,'n02129604' ,'n04468005' ,'n01662784','n04530566' ,'n02062744' ,'n02391049']
classes_name=['airplane','antelope','bear','bicycle','bird','bus','car','cattle','dog','domestic_cat','elephant','fox','giant_panda','hamster','horse','lion','lizard','monkey','motorcycle','rabbit','red_panda','sheep','snake','squirrel','tiger','train','turtle','watercraft','whale','zebra']


xml_dir = "/home/lihanyu/pycodes/faster-rcnn.pytorch/data/VOCdevkit2007_all/VOC2007/Annotations"
for xml_root, dirs, files in os.walk(xml_dir):
# files = os.listdir(path)  # 得到文件夹下所有文件名称
#     print(xml_root)
#     print(dirs)
#     print(files)
    for xmlFile in files:  # 遍历文件夹
        #print(xmlFile.split('.')[-1])
        if 'xml' in xmlFile:  # 判断是否是文件夹,不是文件夹才打开
            print(xmlFile)

            # 将获取的xml文件名送入到dom解析
            dom = xml.dom.minidom.parse(os.path.join(xml_root, xmlFile))  # 输入xml文件具体路径
            root = dom.documentElement
            # 获取标签<name>以及<folder>的值
            name = root.getElementsByTagName('name')
            #objects = root.getElementsByTagName("object")
            folder = root.getElementsByTagName('folder')
            filename=root.getElementsByTagName('filename')

            # 对每个xml文件的多个同样的属性值进行修改。此处将每一个<name>属性修改为plane,每一个<folder>属性修改为VOC2007
            # for object in objects:
            #     # read
            #     objname = object.getElementsByTagName("name")[0]
            #     if objname.childNodes == []:
            #         name = ''
            #     else:
            #         name = objname.childNodes[0].data
            #         name = classes.index(name)
            #         name = classes_name[name]
            #         print(name)

            for i in range(len(name)):
                print(name[i].firstChild.data)
                x_name=name[i].firstChild.data
                x_name=classes.index(x_name)
                x_name=classes_name[x_name]
                name[i].firstChild.data = x_name
                print(name[i].firstChild.data)

            for i in range(len(folder)):
                print(folder[i].firstChild.data)
                folder[i].firstChild.data = 'VOC2007'
                print(folder[i].firstChild.data)

            for i in range(len(filename)):
                print(filename[i].firstChild.data)
                filename[i].firstChild.data = '%s/%s.JPEG' %(xml_root.split('Annotations/')[-1],xmlFile.split('.')[0])
                print(filename[i].firstChild.data)
            # 将属性存储至xml文件中
            with open(os.path.join(xml_root, xmlFile), 'w') as fh:
                dom.writexml(fh)
print('已写入')