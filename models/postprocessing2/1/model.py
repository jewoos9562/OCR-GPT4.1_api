import sys                                                                               
import json         
import base64                 
import cv2              
import numpy as np                                      
                                                                                         
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

def cv2_to_numpy(img):
    img_float = img.astype(np.float32)
    img_transposed = img_float.transpose((2, 0, 1))
    img_expanded = np.expand_dims(img_transposed, axis=0)
    return img_expanded

def pad_and_resize(img, target_size=(480, 480), pad_val=0):
    """
    이미지를 가로세로 비율을 유지하면서 지정된 크기로 리사이징하고,
    필요한 경우 오른쪽과 하단에만 패딩을 적용합니다.
    
    :param img: 입력 이미지
    :param target_size: 목표 크기 (width, height)
    :param pad_val: 패딩에 사용할 값
    :return: 리사이징 및 패딩된 이미지
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 이미지의 비율을 유지하면서 리사이징
    if h / target_h > w / target_w:
        new_h = target_h
        new_w = int(w * target_h / h)
    else:
        new_w = target_w
        new_h = int(h * target_w / w)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 하단과 오른쪽에만 패딩 적용
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    padded_img = cv2.copyMakeBorder(resized_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_val)

    return padded_img, new_w, new_h, w, h

def normalize_image(img, mean, std):
    """
    OpenCV 이미지를 평균과 표준편차를 사용하여 정규화합니다.

    :param img: 입력 이미지
    :param mean: 정규화에 사용될 평균 값
    :param std: 정규화에 사용될 표준편차 값
    :return: 정규화된 이미지
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    # 이미지를 float 형으로 변환하고 정규화 수행
    img = img.astype(np.float32) / 255
    img = (img - mean) / std
    return img

def preprocessing(raw_image, table_bbox):
    image = read_image_from_byte(raw_image[0])
    table_bbox=[int(float(table_bbox[0])),int(float(table_bbox[1])),int(float(table_bbox[2])),int(float(table_bbox[3]))]
    #cropped_image=image.crop(tuple(table_bbox))
    cropped_image=image[table_bbox[1]:table_bbox[3],table_bbox[0]:table_bbox[2]]
    img, nw, nh, oriw, orih = pad_and_resize(cropped_image)
    img = normalize_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    input_data = cv2_to_numpy(img)
    return input_data, np.array([nw]), np.array([nh]), np.array([oriw]), np.array([orih])

                                                            
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
            in_0 = pb_utils.get_input_tensor_by_name(request, "raw_input_image").as_numpy()
            in_1= pb_utils.get_input_tensor_by_name(request, "table_bbox").as_numpy()
            in_2= pb_utils.get_input_tensor_by_name(request, "table_execute_signal").as_numpy()
            
            if in_2[0]==1:
                table_execute_signal=np.array([1]) 
    
                input_datas=[]
                nws=[]
                nhs=[]
                oriws=[]
                orihs=[]
                
                for i in range(len(in_1)):
                    input_data, nw, nh, oriw, orih = preprocessing(in_0, in_1[i])
                    input_datas.append(input_data)
                    nws.append(nw)
                    nhs.append(nh)
                    oriws.append(oriw)
                    orihs.append(orih)
                
                input_datas=np.array(input_datas).reshape(len(in_1),3,480,480)
                nws=np.array(nws).reshape(len(in_1))
                nhs=np.array(nhs).reshape(len(in_1))
                oriws=np.array(oriws).reshape(len(in_1))
                orihs=np.array(orihs).reshape(len(in_1))
            
            else:
                input_datas=np.random.rand(1, 3, 480, 480)
                nws=np.array([1])
                nhs=np.array([1])
                oriws=np.array([1])
                orihs=np.array([1])
            
                
            out_tensor_0 = pb_utils.Tensor("preprocessed_table", input_datas.astype(np.float32))
            out_tensor_1 = pb_utils.Tensor("nw", nws.astype(np.float32))
            out_tensor_2 = pb_utils.Tensor("nh", nhs.astype(np.float32))
            out_tensor_3 = pb_utils.Tensor("oriw", oriws.astype(np.float32))
            out_tensor_4 = pb_utils.Tensor("orih", orihs.astype(np.float32))
            
            
            
            responses.append(pb_utils.InferenceResponse([out_tensor_0,out_tensor_1,out_tensor_2,out_tensor_3,out_tensor_4]))        

        return responses    