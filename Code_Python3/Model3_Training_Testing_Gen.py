#3.Output training data name to cvs file for model 2
import numpy as np
import cv2
import glob
import itertools
import random
import csv
import os
import re

training_file_name = "/content/drive/My Drive/training_model2.csv"
testing_file_name = "/content/drive/My Drive/testing_model2.csv"
visibility_for_testing = []

images_path = '/content/drive/My Drive/Tennis-data-2/Tennis-data-2'
dirs = glob.glob(images_path)

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

with open(training_file_name,'w') as file:
    for index in dirs:
        #################change the path####################################################
        #annos_path = images_path +'groundtruth/'+os.path.split(index)[-1]+'/'
        annos_path = '/content/drive/My Drive/Tennis-data-2' +'groundtruth/'+os.path.split(index)[-1]+'/'
        print(annos_path)
        images_path = index+'/'
        ####################################################################################

        images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
        images = sorted_nicely(images)
        annotations  = glob.glob( annos_path + "*.jpg"  ) + glob.glob( annos_path + "*.png"  ) +  glob.glob( annos_path + "*.jpeg"  )       
        annotations = sorted_nicely(annotations)
        print(len(images),len(annotations))
        #check if annotation counts equals to image counts
        assert len( images ) == len(annotations)
        for im , seg in zip(images,annotations):
            assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

        visibility = {}
        with open("/content/drive/My Drive/Labeled dataset 2.csv", 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            #skip the headers
            next(spamreader, None)  
            
            for row in spamreader:
                visibility[row[0]] = row[1]
                    
                    
        #output all of images path, 0000.jpg & 0001.jpg cant be used as input, so we have to start from 0002.jpg
        for i in range(2,len(images)): 
                #remove image path, get image name   
                #ex: D/Dateset/Clip1/0056.jpg => 0056.jpg 
                file_name = images[i].split('/')[-1]
                #visibility 3 will not be used for training
                if visibility[file_name] == '3': 
                    visibility_for_testing.append(images[i])
                #check if file image name same as annotation name
                assert(  images[i].split('/')[-1].split(".")[0] ==  annotations[i].split('/')[-1].split(".")[0] )
                #write all of images path
                file.write(images[i] + "," + images[i-1] + "," + images[i-2] + "," + annotations[i] + "\n")
                
                    

file.close()

#read all of images path
lines = open(training_file_name).read().splitlines()

#70% for training, 30% for testing 
training_images_number = int(len(lines)*0.7)
testing_images_number = len(lines) - training_images_number
print("Total images:", len(lines), "Training images:", training_images_number,"Testing images:", testing_images_number)

#shuffle the images
random.shuffle(lines)
#training images
with open(training_file_name,'w') as training_file:
    training_file.write("img, img1, img2, ann\n")
    #testing images
    with open(testing_file_name,'w') as testing_file:
        testing_file.write("img, img1, img2, ann\n")
        
        #write img, img1, img2, ann to csv file
        for i in range(0,len(lines)):
            if lines[i] != "":
                if training_images_number > 0 and lines[i].split(",")[0] not in visibility_for_testing :
                    training_file.write(lines[i] + "\n")
                    training_images_number -=1
                else:
                    testing_file.write(lines[i] + "\n")
                    
training_file.close()
testing_file.close()


## 4.Output All of training data name to cvs file for new labeling
training_file_name = "/content/drive/My Drive/training_model3.csv"
count = 0
with open(training_file_name,'w') as file:

            #################change the path####################################################
            images_path = '/content/drive/My Drive/Tennis-data-2/Tennis-data-2/'
            annos_path = '/content/drive/My Drive/Tennis-data-2' +'groundtruth/'+os.path.split(index)[-1]+'/'
            ####################################################################################

            images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
            images = sorted_nicely(images)
            annotations  = glob.glob( annos_path + "*.jpg"  ) + glob.glob( annos_path + "*.png"  ) +  glob.glob( annos_path + "*.jpeg"  )
            annotations = sorted_nicely(annotations)
            print(len(images),len(annotations))
            #check if annotation counts equals to image counts
            assert len( images ) == len(annotations)

            for im , seg in zip(images,annotations):
                assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

                
            with open("/content/drive/My Drive/Labeled dataset 2.csv", 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                #skip the headers
                next(spamreader, None)  

            count+=2
            
            #output all of images path, 0000.jpg & 0001.jpg cant be used as input, so we have to start from 0002.jpg
            for i in range(2,len(images)): 
                    #remove image path, get image name   
                    #ex: D/Dateset/Clip1/0056.jpg => 0056.jpg 
                    file_name = images[i].split('/')[-1]
                    
                    #check if file image name same as annotation name
                    assert( images[i].split('/')[-1].split(".")[0] ==  annotations[i].split('/')[-1].split(".")[0] )
                    
                    #write all of images path
                    file.write(images[i] + "," + images[i-1] + "," + images[i-2] + "," + annotations[i] + "\n")
                    count+=1
print("Total Count:", count)
file.close()

## 5.output training csv file for new labeling image
output_file_name = "/content/drive/My Drive/training_model3.csv"
training_file_name = "/content/drive/My Drive/training_model3.csv"
#read all of images path
lines = open(training_file_name).read().splitlines()


training_file_name1 = "/content/drive/My Drive/training_model2.csv"
#read all of images path
lines1 = open(training_file_name1).read().splitlines()

#combine two file
new_lines = lines1[1:]+lines

#shuffle the images
random.shuffle(new_lines)
#training images
with open(output_file_name,'w') as training_file:
        training_file.write("img, img1, img2, ann\n")

        #write img, img1, img2, ann to csv file
        for i in range(0,len(new_lines)):
            if new_lines[i] != "":
                training_file.write(new_lines[i] + "\n")
                    
training_file.close()
