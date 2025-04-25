import sys                                                                               
import json         
import base64                 
import cv2              
import numpy as np    
                      
                                                                                         
sys.path.append('../../')                                          
import triton_python_backend_utils as pb_utils     
from converter import TableMasterConvertor
from reconstruct_table import convert_xywh_to_xyxy, find_td_details, get_length, get_box_index, insert_into_nth_td, apply_line_breaks_in_table, sort_and_combine_text
                                          
import numpy as np                                                                      

# sys.path.append('../') 
# print(__file__)

def convert_xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from [x_center, y_center, width, height] to [min_x, min_y, max_x, max_y]
    """
    # Separate the components of the boxes
    x_center, y_center, width, height = np.split(boxes, 4, axis=-1)
    
    # Calculate the min and max coordinates
    min_x = x_center - width / 2.0
    min_y = y_center - height / 2.0
    max_x = x_center + width / 2.0
    max_y = y_center + height / 2.0
    
    return np.concatenate([min_x, min_y, max_x, max_y], axis=-1)

def text_to_list(master_token):
    # insert virtual master token
    master_token_list = master_token.split(',')

    if master_token_list[-1] == '<td></td>':
        master_token_list.append('</tr>')
        master_token_list.append('</tbody>')
    elif master_token_list[-1] != '</tbody>':
        master_token_list.append('</tbody>')

    if master_token_list[-2] != '</tr>':
        master_token_list.insert(-1, '</tr>')

    return master_token_list

def merge_span_token(master_token_list):
    """
    Merge the span style token (row span or col span).
    :param master_token_list:
    :return:
    """
    new_master_token_list = []
    pointer = 0
    if master_token_list[-1] != '</tbody>':
        master_token_list.append('</tbody>')
    while pointer < len(master_token_list) and master_token_list[pointer] != '</tbody>':
        try:
            if master_token_list[pointer] == '<td':
                if (pointer + 5) <= len(master_token_list) and (master_token_list[pointer+2].startswith(' colspan=') or
                                                                master_token_list[pointer+2].startswith(' rowspan=')):
                    """
                    example:
                    pattern <td rowspan="2" colspan="3">
                    '<td' + 'rowspan=" "' + 'colspan=" "' + '>' + '</td>'
                    """
                    # tmp = master_token_list[pointer] + master_token_list[pointer+1] + \
                    #       master_token_list[pointer+2] + master_token_list[pointer+3] + master_token_list[pointer+4]
                    tmp = ''.join(master_token_list[pointer:pointer+4+1])
                    pointer += 5
                    new_master_token_list.append(tmp)

                elif (pointer + 4) <= len(master_token_list) and \
                        (master_token_list[pointer+1].startswith(' colspan=') or
                         master_token_list[pointer+1].startswith(' rowspan=')):
                    """
                    example:
                    pattern <td colspan="3">
                    '<td' + 'colspan=" "' + '>' + '</td>'
                    """
                    # tmp = master_token_list[pointer] + master_token_list[pointer+1] + master_token_list[pointer+2] + \
                    #       master_token_list[pointer+3]
                    tmp = ''.join(master_token_list[pointer:pointer+3+1])
                    pointer += 4
                    new_master_token_list.append(tmp)

                else:
                    new_master_token_list.append(master_token_list[pointer])
                    pointer += 1
            else:
                new_master_token_list.append(master_token_list[pointer])
                pointer += 1
        except:
            print("Break in merge...")
            break
    new_master_token_list.append('</tbody>')

    return new_master_token_list

def deal_eb_token(master_token):
    """
    post process with <eb></eb>, <eb1></eb1>, ...
    emptyBboxTokenDict = {
        "[]": '<eb></eb>',
        "[' ']": '<eb1></eb1>',
        "['<b>', ' ', '</b>']": '<eb2></eb2>',
        "['\\u2028', '\\u2028']": '<eb3></eb3>',
        "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
        "['<b>', '</b>']": '<eb5></eb5>',
        "['<i>', ' ', '</i>']": '<eb6></eb6>',
        "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
        "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
        "['<i>', '</i>']": '<eb9></eb9>',
        "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']": '<eb10></eb10>',
    }
    :param master_token:
    :return:
    """
    master_token = master_token.replace('<eb></eb>', '<td></td>')
    master_token = master_token.replace('<eb1></eb1>', '<td> </td>')
    master_token = master_token.replace('<eb2></eb2>', '<td><b> </b></td>')
    master_token = master_token.replace('<eb3></eb3>', '<td>\u2028\u2028</td>')
    master_token = master_token.replace('<eb4></eb4>', '<td><sup> </sup></td>')
    master_token = master_token.replace('<eb5></eb5>', '<td><b></b></td>')
    master_token = master_token.replace('<eb6></eb6>', '<td><i> </i></td>')
    master_token = master_token.replace('<eb7></eb7>', '<td><b><i></i></b></td>')
    master_token = master_token.replace('<eb8></eb8>', '<td><b><i> </i></b></td>')
    master_token = master_token.replace('<eb9></eb9>', '<td><i></i></td>')
    master_token = master_token.replace('<eb10></eb10>', '<td><b> \u2028 \u2028 </b></td>')
    return master_token

def insert_text_to_token(master_token_list):
    """
    Insert OCR text result to structure token.
    :param master_token_list:
    :param cell_content_list:
    :return:
    """
    master_token_list = merge_span_token(master_token_list)
    merged_result_list = []
    text_count = 0
    for master_token in master_token_list:
        master_token = deal_eb_token(master_token)
        merged_result_list.append(master_token)

    return ''.join(merged_result_list)

def htmlPostProcess(text):
    text = '<html><head><style>table {border-collapse: collapse;}th, td {border: 1px solid black;padding: 8px;text-align: left;}</style></head><body><table>' + text + '</table></body></html>'
    return text


def postprocessing(onnx_outputs, nw, nh, oriw, orih, text_results,t):
    img_metas = [{'filename': None,
            'ori_shape': (orih, oriw, 3),
            'img_shape': (nh, nw, 3),
            'scale_factor': (nh/orih, nw/oriw),
            'img_norm_cfg': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
            'ori_filename': None,
            'pad_shape': (480, 480, 3)
            }]  
    

    
    onnx_outputs[0]=np.array(onnx_outputs[0])
    onnx_outputs[1]=np.array(onnx_outputs[1])
    label_convertor = TableMasterConvertor(dict_file='/models/postprocessing3/structure_alphabet.txt', 
                                    max_seq_len=119, start_end_same=False, with_unknown=True)

    strings, scores, pred_bboxes = label_convertor.output_format(onnx_outputs[0], onnx_outputs[1], img_metas)
    
    results = []
    for string, score, pred_bbox in zip(strings, scores, pred_bboxes):
        results.append(dict(text=string, score=score, bbox=pred_bbox))
    
    pred_text = results[0]['text']
    pred_html = insert_text_to_token(text_to_list(pred_text))
    pred_html = htmlPostProcess(pred_html)
    pred_bbox = pred_bbox[~np.all(pred_bbox == 0, axis=1)]   
    pred_bbox = convert_xywh_to_xyxy(pred_bbox)

    
    index_list = find_td_details(pred_html)
 
    col_position, row_position = get_length(index_list, pred_bbox)
    index_dict = dict()
    
    
    for i in range(len(index_list)):
        for c in range(index_list[i]['colspan']):
            for r in range(index_list[i]['rowspan']):
                index_dict[(index_list[i]['row']+r-1, index_list[i]['column']+c-1)] = i
          
    if(col_position == None):
        pred_html='failed'
    else:
        invalid = False
        for c in range(len(col_position)-1):
            for r in range(len(row_position)-1):
                if(index_dict.get((r, c)) == None):
                    invalid = True
        
        if(invalid):
            pred_html='failed'
        else:
            text_dict = dict()



            for text_box in text_results:
                tmp_box = [float(text_box[bi]) - float(t[bi%2]) for bi in range(4)] + [text_box[4]]
                box_idx = get_box_index(tmp_box, col_position, row_position, index_list, index_dict)
                if(box_idx != None):
                    if(box_idx not in text_dict.keys()):
                        text_dict[box_idx] = list()
                    text_dict[box_idx].append((tmp_box[:4],tmp_box[4]))

            for k in text_dict.keys():
                text_dict[k] = sort_and_combine_text(text_dict[k])
            pred_html = insert_into_nth_td(pred_html, text_dict)
            pred_html = apply_line_breaks_in_table(pred_html)

            

    return pred_html


                                                            
class TritonPythonModel:                                                                 
    """This model always returns the input that it has received.                         
    """                                                                                                        
                                                                                         
    def execute(self, requests):                                                         
        """ This function is called on inference request.                                
        """                                                                              
        responses = []   
             
        for request in requests:                                                            
            tr_model_output1 = pb_utils.get_input_tensor_by_name(request, "tr_model_output1").as_numpy()
            tr_model_output2 = pb_utils.get_input_tensor_by_name(request, "tr_model_output2").as_numpy()
            text_results=pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            table_results=pb_utils.get_input_tensor_by_name(request, "table_bbox_results").as_numpy()
            recognition_flag_pp3 = pb_utils.get_input_tensor_by_name(request, "recognition_flag_pp3").as_numpy()
            

            if recognition_flag_pp3[0]==0:
                html_tags=[]
                final_table_results=[]
                final_texts=np.array([])
                
            else:
                text_results=text_results.reshape(-1,5)
                final_texts=text_results
                onnx_outputs=[]
                for i in range(len(tr_model_output1)):
                    onnx_outputs.append([[tr_model_output1[i]],[tr_model_output2[i]]])
            
                nw_pp = pb_utils.get_input_tensor_by_name(request, "nw_pp").as_numpy()
                nh_pp = pb_utils.get_input_tensor_by_name(request, "nh_pp").as_numpy()
                oriw_pp = pb_utils.get_input_tensor_by_name(request, "oriw_pp").as_numpy()
                orih_pp = pb_utils.get_input_tensor_by_name(request, "orih_pp").as_numpy()
                table_execute_signal_pp=pb_utils.get_input_tensor_by_name(request, "table_execute_signal_pp").as_numpy()

                if table_execute_signal_pp[0]==1:

                    html_tags=[]
                    for i in range(len(onnx_outputs)):
                        pp3_result=postprocessing(onnx_outputs[i], np.array([nw_pp[i]]), np.array([nh_pp[i]]), np.array([oriw_pp[i]]), np.array([orih_pp[i]]),text_results,np.array(table_results[i]))
                        html_tags.append(pp3_result)
                    final_table_results=table_results
                    print('hello')
                else:
                    html_tags=[]
                    final_table_results=[]
            

            
            html_tags=np.array(html_tags)
            final_table_results=np.array(final_table_results)
            final_texts=np.array(final_texts)

            
            out_tensor_0 = pb_utils.Tensor("html_tags", html_tags.astype(np.object_))
            out_tensor_1 = pb_utils.Tensor("final_table_bboxes", final_table_results.astype(np.float32))
            out_tensor_2 = pb_utils.Tensor("final_texts", final_texts.astype(np.object_))
            responses.append(pb_utils.InferenceResponse([out_tensor_0,out_tensor_1,out_tensor_2])) 
            

        return responses    