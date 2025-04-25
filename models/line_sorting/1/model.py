import sys                                                                               
import json         
import base64                 
import cv2                                                    
                                                                                         
sys.path.append('../../')                                                                
import triton_python_backend_utils as pb_utils                                           
import numpy as np                                                                 

                                                            
class TritonPythonModel:                                                                 
                                                        
    def initialize(self, args):                                                          
        self.model_config = json.loads(args['model_config'])                             
                                                                                         
    def execute(self, requests):                                                         
                                                                     
        responses = []   
                                         
        for request in requests:                                                          
            in_0 = pb_utils.get_input_tensor_by_name(request, "final_texts_wo_line").as_numpy()
            in_1 = pb_utils.get_input_tensor_by_name(request, "image_height").as_numpy()
            in_2 = pb_utils.get_input_tensor_by_name(request, "image_width").as_numpy()
            
            for byte_texts in in_0:
                byte_texts[4]=byte_texts[4].decode('utf-8-sig')

            in_0=in_0.tolist()
            
            
            image_height=float(in_1[0])
            threshold=image_height*0.005
            final_texts=[]
            for bbox in in_0:
                float_bbox=[float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3]),bbox[4]]
                #float_bbox = tuple(float(coord) for coord in bbox[0:4])  # 처음 4개 값만 float으로 변환
                final_texts.append(float_bbox)
            final_texts.sort(key=lambda final_texts: final_texts[1])
            #print(final_texts)

            
            lines=[]
            current_line = []
            for bbox in final_texts:
                if not current_line:
                    current_line.append(bbox)
                else:
                    last_bbox = current_line[-1]
                    # Y축 겹침 여부 판단, 여기서 'threshold'는 겹침 판단 기준값
                    if bbox[1] <= last_bbox[3] + threshold:
                        current_line.append(bbox)
                    else:
                        lines.append(current_line)
                        current_line = [bbox]

            if current_line:  # 마지막 라인 처리
                lines.append(current_line)

            # 각 라인 내에서 X축 기준 정렬
            for line in lines:
                line.sort(key=lambda bbox: bbox[0])
                
            #print(lines)
            
            final_result=[]
            for idx,line in enumerate(lines):
                for bbox in line:
                    final_result.append([str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(bbox[4]),str(idx)])
            final_result=np.array(final_result)
            #print(final_result) 
            
            # final_texts_with_line = np.hstack((in_0, np.zeros((in_0.shape[0], 1), dtype=in_0.dtype)))
            # print(final_texts_with_line)

            out_tensor_0 = pb_utils.Tensor("final_texts_with_line", final_result.astype(np.object_))
            
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))        

        return responses    