import sys                                                                               
import json         
import base64                 
import cv2                                                    
                                                                                         
sys.path.append('../../')                                                                
import triton_python_backend_utils as pb_utils                                           
import numpy as np

def read_image_from_byte(raw_image):
    base64_image = raw_image.decode("utf-8")
    raw_image = base64.b64decode(base64_image)
    image = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class TritonPythonModel:                                                                                                                                        
                                                                                                                                                                                             
    def execute(self, requests):                                                         
                                                                                           

        responses = []   
        for request in requests:
            detections= pb_utils.get_input_tensor_by_name(request, "detection_result").as_numpy()      
            padded_image= pb_utils.get_input_tensor_by_name(request, "padded_image").as_numpy()     
            raw_image_string= pb_utils.get_input_tensor_by_name(request, "raw_image").as_numpy()
            ratio= pb_utils.get_input_tensor_by_name(request, "ratio").as_numpy()  

            output_images=[]
            cropped_images_batch=[]
            bboxes_batch=[]
            

            output=detections[0]
            padded_image=padded_image[0]
            raw_image_string=raw_image_string[0]
            r=ratio
            
            input_data=read_image_from_byte(raw_image_string).astype('float32')
                
            
            bboxes = output[:, :4]
            scores = output[:, 4]
            classes = output[:, 5:]
            
            CONF_THRESHOLD = 0.40
            IOU_THRESHOLD = 0.5
            
            keep_indices = (scores >= CONF_THRESHOLD)
            bboxes = bboxes[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]
            
            indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)\
            
            input_data_copy=input_data.copy()    

            recognition_flag=1
            cropped_images=[]
            bboxes_final=[]
            #print(f'indices: {indices}')
            
            if len(indices)==0:
                recognition_flag=np.array([0])
                bboxes_final=[[0,0,0,0]]
                
                #random_image_fixed_size = np.random.randint(0, 256, (128, 32, 3), dtype=np.uint8)
                #cropped_images=np.array([random_image_fixed_size.transpose((2,0,1))/255])

                cropped_images=[np.random.rand(3, 32, 128)]
                
            else:
                recognition_flag=np.array([1])
                for i in indices:
                    bbox = bboxes[i]
                    score = scores[i]
                    class_scores = classes[i]
                    class_id=(np.argmax(class_scores))
                    
                    # print(bbox)
                    c = bbox[:2]
                    h = bbox[2:] / 2
                    p1, p2 = (c - h) / r, (c + h) / r
                    p1, p2 = p1.astype('int32'), p2.astype('int32')
                    input_data=cv2.rectangle(input_data, p1, p2, (255,0,0), 2)
                    # print(f'p1,p2: {p1},{p2}')
                    cropped_image_temp=input_data_copy[p1[1]:p2[1],p1[0]:p2[0]]
                    # print(f'size: {cropped_image_temp.shape}')
                    cropped_image=(cv2.resize(input_data_copy[p1[1]:p2[1],p1[0]:p2[0]],(128,32)).transpose((2,0,1)))/255
                    
                    #print(type(cropped_image))
                    #print(f'size: {cropped_image.shape}')
                
                    bbox_final=[p1[0],p1[1],p2[0],p2[1]]
                    bboxes_final.append(bbox_final)
                    cropped_images.append(cropped_image)  
                    
            input_data=input_data.transpose((2,0,1))
            #print(input_data.shape) 
            cropped_images_batch.append(cropped_images)
            bboxes_batch.append(bboxes_final)
            output_images.append(input_data)
            
            #print('hereeee')

            output_images=np.array(output_images)
            bboxes_batch=np.array(bboxes_batch)
            cropped_images_batch=np.array(cropped_images_batch[0])

            raw_iamge_batch_tensor = pb_utils.Tensor(
                "input_image_show", output_images.astype(np.float32)
            )
            
            cropped_images_batch_tensor = pb_utils.Tensor(
                "cropped_images", cropped_images_batch.astype(np.float32)
            )
            
            bboxes_batch_tensor = pb_utils.Tensor(
                "bboxes", bboxes_batch.astype(np.float32)
            )
            
            recognition_flag_tensor = pb_utils.Tensor(
                "recognition_flag", recognition_flag.astype(np.float32)
            )
            

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    raw_iamge_batch_tensor,
                    cropped_images_batch_tensor,
                    bboxes_batch_tensor,
                    recognition_flag_tensor
                ]
            )
            
            responses.append(response)
            
        return responses    