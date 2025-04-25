import sys                                                                               
import json         
import base64                 
import cv2                                                    
                                                                                         
sys.path.append('../../')                                                                
import triton_python_backend_utils as pb_utils                                           
import numpy as np
import albumentations as A                                                                       

def read_image_from_byte(raw_image):
    base64_image = raw_image.decode("utf-8")
    raw_image = base64.b64decode(base64_image)
    image = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def letterbox(im, new_shape=(1280, 1280), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def gen_input_batch(raw_image):

    padded_image= []

    #print(len(raw_image))

    image = read_image_from_byte(raw_image[0])
    h,w,_=image.shape
    
    # print(f'image_type:{type(image)}')
    image_padded, ratio, _ = letterbox(image)
    image_padded = image_padded.astype('float32')
    image_padded = image_padded.transpose((2,0,1)) / 255.0
    image_padded = np.ascontiguousarray(image_padded)   
    padded_image.append(image_padded)

        
    padded_image=np.array(padded_image)
    ratio=np.array(ratio)
    
    return padded_image, ratio, h, w


def gen_dla_batch(raw_image):

    # print(len(raw_image))

    image = read_image_from_byte(raw_image[0])
    h,w,_=image.shape
    img = cv2.resize(image, dsize=(1000, 1000))
    
    #padded_img = np.transpose(img, (2, 0, 1))
    padded_image = img.transpose((2,0,1))
    
    # print(padded_image.shape)

    ratio=np.array([w/1000,h/1000])

    return padded_image, ratio

                                                            
class TritonPythonModel:                                                                 
    """This model always returns the input that it has received.                         
    """                                                                                  
                                                                                         
    def initialize(self, args):                                                          
        self.model_config = json.loads(args['model_config'])                             
                                                                                         
    def execute(self, requests):                                                         
        """ This function is called on inference request.                                
        """                                                                              
        responses = []   
                                         
        for request in requests:                                                            
            in_0 = pb_utils.get_input_tensor_by_name(request, "input_image").as_numpy()
            

            padded_image, ratio, h, w = gen_input_batch(in_0)
            dla_input, ratio_dla= gen_dla_batch(in_0)
            
            h=np.array([h])
            w=h=np.array([w])

            out_tensor_0 = pb_utils.Tensor("padded_image", padded_image.astype(np.float32))
            out_tensor_1 = pb_utils.Tensor("ratio", ratio.astype(np.float32))
            out_tensor_2 = pb_utils.Tensor("dla_input", dla_input.astype(np.uint8))
            out_tensor_3 = pb_utils.Tensor("ratio_dla", ratio_dla.astype(np.float32))
            out_tensor_4 = pb_utils.Tensor("height", h.astype(np.float32))
            out_tensor_5 = pb_utils.Tensor("width", w.astype(np.float32))
            
            responses.append(pb_utils.InferenceResponse([out_tensor_0,out_tensor_1,out_tensor_2,out_tensor_3,out_tensor_4,out_tensor_5]))        

        return responses    