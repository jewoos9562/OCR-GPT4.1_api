from bs4 import BeautifulSoup
import numpy as np
import cv2
import time

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

def find_td_details(html):
    """
    Find the row, start column, colspan, and rowspan of the nth <td> in a HTML table.

    Args:
    html (str): HTML string containing the table.
    nth_td (int): The 1-based index of the <td> to find.

    Returns:
    dict: A dictionary containing the row, start column, colspan, and rowspan of the nth <td>.
    """
    soup = BeautifulSoup(html, 'html.parser')

    row_number = 0
    col_tracker = [0] * 100  # Assuming a maximum of 100 columns
    res = list()
    for tr in soup.find_all('tr'):
        row_number += 1
        current_col = 0
        
        for col in tr.find_all(['td', 'th']):
            current_col += 1
            while col_tracker[current_col] > 0:  # Skip columns covered by a previous rowspan
                current_col += 1

            colspan = int(col.get('colspan', 1))
            rowspan = int(col.get('rowspan', 1))
            res.append({
                "row": row_number,
                "column": current_col,
                "colspan": colspan,
                "rowspan": rowspan
            })

            # Update tracker for colspan and rowspan
            for i in range(current_col, current_col + int(col.get('colspan', 1))):
                col_tracker[i] = max(col_tracker[i], int(col.get('rowspan', 1)))

        # Decrease rowspan tracker after processing each row
        col_tracker = [max(x - 1, 0) for x in col_tracker]

    return res

def check_elements(l):
    for j in range(len(l) - 1):
        if l[j] >= l[j + 1]:
            return False
    return True

def get_length(index_list, bbox_array):
    max_right = max(index['column'] - 1 + index['colspan'] for index in index_list)
    max_bottom = max(index['row'] - 1 + index['rowspan'] for index in index_list)


    col_list = [[] for _ in range(max_right + 1)]
    row_list = [[] for _ in range(max_bottom + 1)]
    
    for index, box in zip(index_list, bbox_array):
        left = index['column'] - 1
        up = index['row'] - 1
        right = left + index['colspan']
        bottom = up + index['rowspan']

        col_list[left].append(box[0])
        row_list[up].append(box[1])
        col_list[right].append(box[2])
        row_list[bottom].append(box[3])


    col_medians = [np.median(sublist) for sublist in col_list if len(sublist)!=0]
    row_medians = [np.median(sublist) for sublist in row_list if len(sublist)!=0]
    #print(col_medians,row_medians)
    col_medians = [int(c) for c in col_medians]
    row_medians = [int(r) for r in row_medians]
    valid = check_elements(col_medians) and check_elements(row_medians)

    if not valid:
        return None, None
    
    return col_medians, row_medians



def find_just_smaller_index(lst, num):
    return next((i for i, x in reversed(list(enumerate(lst))) if x <= num), -1)
def find_greater_index(lst, num):
    return next((i for i, x in enumerate(lst) if x >= num), len(lst)) 

def get_inter_area(box1, box2):
    # 각 박스의 좌표를 추출합니다.
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 교차 영역의 좌표를 계산합니다.
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 교차 영역의 면적을 계산합니다.
    return max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

def get_box_index(box, col_position, row_position, index_list, index_dict):
    left_index = max(0, find_just_smaller_index(col_position, box[0]))
    right_index = min(len(col_position)-1, find_greater_index(col_position, box[2]))
    upper_index = max(0, find_just_smaller_index(row_position, box[1]))
    bottom_index = min(len(row_position)-1, find_greater_index(row_position, box[3]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    index_to_check = set()

    for c in range(left_index, right_index):
        for r in range(upper_index, bottom_index):
            index_to_check.add(index_dict[(r,c)])
    inter_area_list = []
    for i in index_to_check:
        tmp1 = index_list[i]['row']-1
        tmp2 = index_list[i]['column']-1
        tmp3 = tmp1 + index_list[i]['rowspan']
        tmp4 = tmp2 + index_list[i]['colspan']
        tmp = [col_position[tmp2], row_position[tmp1], col_position[tmp4], row_position[tmp3]]
        inter_area = get_inter_area(box[:4], tmp)
        inter_area_list.append((inter_area,i))
    inter_area_list.sort()
    if(len(inter_area_list) == 0 or area // 3 > inter_area_list[-1][0]):
        return None
    return inter_area_list[-1][1]

def insert_into_nth_td(html, text_dict):
    # HTML 파싱
    soup = BeautifulSoup(html, 'html.parser')

    # 모든 td 태그 찾기
    td_tags = soup.find_all('td')
    
    # n번째 td 태그에 텍스트 삽입
    for k in text_dict.keys():
        if 0 <= k <= len(td_tags):
            td_tags[k].string = text_dict[k]

    return str(soup)

def apply_line_breaks_in_table(html_string):
    # HTML 파싱
    soup = BeautifulSoup(html_string, 'html.parser')

    # 모든 td 태그 찾기
    td_tags = soup.find_all('td')

    # 각 td 태그 내의 텍스트에서 줄바꿈을 <br> 태그로 대체
    for td in td_tags:
        lines = td.get_text(separator='\n').split('\n')  # 줄바꿈 문자로 분리
        td.clear()  # 기존 내용을 제거
        for line in lines:
            if line:
                td.append(BeautifulSoup(line, 'html.parser'))
                td.append(BeautifulSoup('<br/>', 'html.parser'))  # 줄바꿈 추가
        if td.contents and isinstance(td.contents[-1], BeautifulSoup) and td.contents[-1].name == 'br':
            td.contents.pop()  # 마지막 <br/> 제거

    # 수정된 HTML 반환
    return str(soup)

def sort_and_combine_text(data):
    # 높이의 중앙값을 계산하여 tolerance로 사용
    heights = [box[3] - box[1] for box, _ in data]
    tolerance = np.median(heights)//2

    # y 좌표에 따라 정렬
    data.sort(key=lambda item: (item[0][1], item[0][0]))

    # 같은 라인으로 간주될 텍스트 그룹화
    lines = []
    current_line = []

    # print(f'data: {data}')
    for box, text in data:
        if not current_line or abs(current_line[-1][0][1] - box[1]) <= tolerance:
            current_line.append((box, text))
        else:
            current_line.sort(key=lambda item: item[0][0])
            lines.append(current_line)
            current_line = [(box, text)]

    # 마지막 라인 추가
    if current_line:
        current_line.sort(key=lambda item: item[0][0])
        lines.append(current_line)

    # 각 라인의 텍스트를 결합
    return "\n".join([" ".join([text.decode('utf-8-sig') for _, text in line]) for line in lines])


# if __name__ == '__main__':

#     # HTML string
#     html_code = """
#     <html>
#     <head>
#     <style>
#     table {
#         border-collapse: collapse;
#         margin-left: 20px; /* Adjust this value to increase or decrease the indentation */
#     }
#     th, td {
#         border: 1px solid black;
#         padding: 8px;
#         text-align: left;
#     }
#     </style>
#     </head>
#     <body>
#     <table>
#         <thead>
#             <tr>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#         </thead>
#         <tbody>
#             <tr>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             <tr>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             <tr>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             <tr>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             <tr>
#                 <td colspan="4"></td>
#             </tr>
#         </tbody>
#     </table>
#     </body>
#     </html>

#     """
#     bbox_array = np.array([
#         [ 44.61951345,  19.40230987,  80.12373984,  30.49063963],
#         [177.24954039,  19.65220059, 187.55440503,  31.73837213],
#         [309.68634367,  19.04349327,  76.27617091,  29.94326311],
#         [443.31384927,  18.28415646, 185.539864  ,  30.78739727],
#         [ 45.13480961,  48.27031472,  80.54138213,  29.60857504],
#         [178.90499175,  47.87416346, 188.42224091,  30.76154765],
#         [309.97837901,  47.45765686,  76.91836417,  29.27110672],
#         [444.32866216,  46.64804403, 186.86821461,  30.43146021],
#         [ 44.94595259,  77.63211026,  81.1564824 ,  29.80613765],
#         [178.48652244,  76.6618392 , 189.34704155,  31.56199231],
#         [310.94644904,  76.28802917,  78.2355997 ,  30.30383391],
#         [446.31670564,  74.96724914, 188.76244158,  31.73082576],
#         [ 44.36509073, 104.45541326,  81.1770916 ,  29.77108675],
#         [178.5371688 , 104.88112113, 191.22380197,  31.57602591],
#         [310.95534772, 105.76252376,  78.56471837,  30.95649551],
#         [447.61591226, 105.08540266, 189.14144576,  33.17820605],
#         [ 45.00993043, 139.95680865,  81.19885862,  45.30786851],
#         [180.103468  , 139.08129748, 195.52705586,  45.62696737],
#         [311.48410767, 139.19657819,  79.93878633,  43.67153112],
#         [448.24133366, 138.64096024, 191.82868034,  44.89020348],
#         [282.22982794, 167.04976194, 528.64443898,  20.14235048]])


#     # Test the function
#     time1 = time.time()
#     oriw, orih = 555,177
#     index_list = find_td_details(html_code)
#     bbox_array = bbox_array[~np.all(bbox_array == 0, axis=1)]
#     bbox_array = convert_xywh_to_xyxy(bbox_array)
#     col_position, row_position = get_length(index_list, bbox_array)
#     index_dict = dict()
#     for i in range(len(index_list)):
#         for c in range(index_list[i]['colspan']):
#             for r in range(index_list[i]['rowspan']):
#                 index_dict[(index_list[i]['row']+r-1, index_list[i]['column']+c-1)] = i
                
#     example = [30,45,59,90,'example']
#     box_idx = get_box_index(example, col_position, row_position, index_list, index_dict)


#     # Create a blank image with white background
#     image = np.ones((orih, oriw, 3), dtype=np.uint8) * 255
#     id = 0
#     for i in index_list:
#         top_left = (col_position[i['column']-1], row_position[i['row']-1])
#         bottom_right = (col_position[i['column']+i['colspan']-1], row_position[i['row']+i['rowspan']-1])
#         cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        
#         # 텍스트 위치 (사각형 중앙) 계산
#         text_x = (top_left[0] + bottom_right[0]) // 2
#         text_y = (top_left[1] + bottom_right[1]) // 2

#         # 텍스트 그리기
#         cv2.putText(image, str(id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#         id += 1
