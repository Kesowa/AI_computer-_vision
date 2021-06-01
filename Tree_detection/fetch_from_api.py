import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

#Longitude =  X-axis
#Latitude = Y-axis
#minLong, minLat = 88.47991, 22.55835
#offset = 0.00030
#endLat = 22.59148

minLong, minLat = 88.52717, 22.53676
offset = 0.00030

endLat = 22.55242
image_max_size = 2000


def extract_from_URL(minLat = None, 
                     minLong = None,
                     maxLat = None,
                     maxLong = None,
                     show = True):
    # Input:  {ALL THE COORDINATES}
    # The image URL
    IMAGE_URL = f"https://cog.kesowa.com/cog/crop/{minLong},{minLat},{maxLong},{maxLat}.png?url=http://localhost:5000/project/raster_layer/1620540289436-ortho_cog.tif&max_size={image_max_size}&resampling_method=nearest&return_mask=true"
    #print(IMAGE_URL)
    # the response that we get from the URL
    response = requests.get(IMAGE_URL)
    # Reading the image into PILLOW
    img = Image.open(BytesIO(response.content))
    if show:
        img.show()     
    return img


print(f"Starting downloading image from Coordinates: {minLat, minLong}\n")

name = 530
maxLong, maxLat =  minLong + offset, minLat + offset

while maxLat < endLat:    
    # getting the image from the API
    img = extract_from_URL(minLat = minLat, 
                            minLong = minLong,
                            maxLat = maxLat,
                            maxLong = maxLong,
                            show = False)
    
    # updating the latitude and longitude ( with the offset also)
    minLat = maxLat
    maxLat =  minLat + offset
    # saving the image
    img.save(f"dataset/{name}.png")
    print(f"{name}.png saved.")
    name += 1
    
    
    
    
