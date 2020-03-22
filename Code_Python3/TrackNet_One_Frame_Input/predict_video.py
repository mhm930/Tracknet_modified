import argparse
import Models
import queue as Queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from matplotlib import pyplot as plt

def Utils_indexes(y, thres=0.3, min_dist=1):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)
    
    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])
    
    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default = "")
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()
input_video_path =  args.input_video_path
output_video_path =  args.output_video_path
save_weights_path = args.save_weights_path
n_classes =  args.n_classes

if output_video_path == "":
	#output video in same path
	output_video_path = '/content/drive/My Drive/' + "_TrackNet_Model1_5695.mp4"

#get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#start from first frame
currentFrame = 0

#width and height in TrackNet
width , height = 640, 360
img, img1, img2 = None, None, None

#load TrackNet model
modelFN = Models.TrackNet.TrackNet
m = modelFN( n_classes , input_height=height, input_width=width   )
m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
m.load_weights(  save_weights_path  )
buffer_length = 10
# In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames 
q = Queue.deque()
for i in range(0,buffer_length):
	q.appendleft(None)

#save prediction images as vidoe
#Tutorial: https://stackoverflow.com/questions/33631489/error-during-saving-a-video-using-python-and-opencv
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))
fps = int(video.get(cv2.CAP_PROP_FPS))
property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
frames = int(video.get(property_id))
print(frames)
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print (fps,output_width,output_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))

count = 1

while(count < 1000):

	tic_total = time.time()
	tic = time.time()
	possible_LS = [];

	#capture frame-by-frame
	video.set(1,currentFrame); 
	ret, img = video.read()

	#if there dont have any frame in video, break
	if not ret: 
		break

	#img is the frame that TrackNet will predict the position
	#since we need to change the size and type of img, copy it to output_img
	output_img = cv2.flip(img,0)

	#resize it 
	img = cv2.resize(cv2.flip(img,0), ( width , height ))
	#input must be float type
	img = img.astype(np.float32)


	#since the odering of TrackNet  is 'channels_first', so we need to change the axis
	X = np.rollaxis(img, 2, 0)
	#prdict heatmap
	pr = m.predict( np.array([X]) )[0]

	#since TrackNet output is ( net_output_height*model_output_width , n_classes )
	#so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
	#.argmax( axis=2 ) => select the largest probability as class
	pr = pr.reshape(( height ,  width , n_classes ) ).argmax( axis=2 )

	#cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
	pr = pr.astype(np.uint8) 

	#reshape the image size as original input image
	heatmap = cv2.resize(pr  , (output_width, output_height ))

	#heatmap is converted into a binary image by threshold method.
	ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)

	#find the circle in image with 2<=radius<=7
	circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)

	#In order to draw the circle in output_img, we need to used PIL library
	#Convert opencv image format to PIL image format
	PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)   
	PIL_image = Image.fromarray(PIL_image)

	#check if there have any tennis be detected
	if circles is not None:
		#if only one tennis be detected
		if len(circles) == 1:

			x = int(circles[0][0][0])
			y = int(circles[0][0][1])
			print (currentFrame, x,y)
			temp = []
			q.appendleft([x,y])          
			temp = np.array([i for i in q if i]) ;
			temp.reshape(-1,2)
			temp_y = temp[:,1]
			temp_x = temp[:,0]
			#print ('outliers',detect_outlier(temp_y))
			indexes = Utils_indexes(temp_y,thres=0.8/max(temp_y), min_dist=0.5)
			dist = [np.abs(temp_y[i-1]-temp_y[i]) for i in range(1,len(temp_y))]
			d = (np.std(dist))
			print(temp_y,dist,d)
                          
			if len(indexes)==1:
			    print ('Possible LS')
			    print(indexes)
			    cv2.putText(output_img,'Possible LS',(temp_x[indexes],temp_y[indexes]),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
			    possible_LS.append([temp_x[indexes],temp_y[indexes]])
			    image_np1 = output_img.copy()
                #cv2.putText(image_np,'Possible LS',(uniq_arr[k][0],uniq_arr[k][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
			    plt.clf()
			    plt.plot(temp_x,temp_y,linestyle='-', marker='o')
			    plt.gca().invert_yaxis()
			    print('indexes',indexes,len(indexes))
			    #print('Difference Index',np.diff(indexes).all())
			    #plt.plot(temp_x[indexes],temp_y[indexes],'.r')
			    #plt.savefig('graph_{}.png'.format(count))
  
			#pop x,y from queue
			q.pop()    
		else:
			#push None to queue
			q.appendleft(None)
			#pop x,y from queue
			q.pop()
	else:
		#push None to queue
		q.appendleft(None)
		#pop x,y from queue
		q.pop()

	#draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
	for i in range(0,buffer_length):
		if q[i] is not None:
			draw_x = q[i][0]
			draw_y = q[i][1]
			bbox =  (draw_x - 8, draw_y - 8, draw_x + 8, draw_y + 8)
			draw = ImageDraw.Draw(PIL_image)
			draw.ellipse(bbox, outline ='red')
			del draw

	#Convert PIL image format back to opencv image format
	opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
	#write image to output_video
	output_video.write(opencvImage)
	print("FPS : ", 1 / (time.time() - tic))
	#next frame
	currentFrame += 1
	count += 1

# everything is done, release the video
video.release()
output_video.release()

