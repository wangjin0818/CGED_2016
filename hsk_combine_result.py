from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging

import pandas as pd
from hsk_preprocess import hsk_final_serialize

error_dict = {
    'R': 1,
    'M': 2,
    'S': 3,
    'W': 4
}

error_invdict = {
    1: 'R',
    2: 'M',
    3: 'S',
    4: 'W' 
}

def update_dict(target_dict, key_a, key_b, val): 
    if key_a in target_dict:
        target_dict[key_a].update({key_b: val})
    else:
        target_dict.update({key_a:{key_b: val}})

def load_detect_dict(file_name):
    ret_dict = {}

    file_data = pd.read_table(file_name, header=None, sep='\t', quoting=3)
    for i in range(len(file_data[0])):
        ret_dict[file_data[0][i]] = file_data[1][i]

    return ret_dict

def load_identification_dict(file_path):
    ret_dict = {}

    for err_type in error_dict.keys():
        file_name = os.path.join(file_path, 'identification_%s_label.txt' % (err_type))
        file_data = pd.read_table(file_name, header=None, sep='\t', quoting=3)

        for i in range(len(file_data[0])):
            update_dict(ret_dict, error_dict[err_type], file_data[0][i], file_data[1][i])

    return ret_dict

def load_position_dict(file_name):
    ret_dict = {}

    file_data = pd.read_table(file_name, header=None, sep='\t', quoting=3)
    for i in range(len(file_data[0])):
        update_dict(ret_dict, file_data[0][i], file_data[1][i], file_data[2][i])

    return ret_dict

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # load origin data
    test_file = os.path.join('data', 'CGED16_HSK_Test_Input.txt')
    test_sid, test_length = hsk_final_serialize(test_file)

    # test_path
    test_path = os.path.join('result', 'track1')

    # load detect dict
    detect_file = os.path.join(test_path, 'detect_result.txt')
    detect_dict = load_detect_dict(detect_file)

    # load identification R, S, W, M label dict
    identification_dict = load_identification_dict(test_path)

    # load position dict
    position_file = os.path.join(test_path, 'position_result.txt')
    position_dict = load_position_dict(position_file)

    # with open(os.path.join(test_path, 'final_result_bak.txt'), 'w') as my_file:
    #     for i in range(len(test_sid)):
    #         sid = test_sid[i]
    #         length = test_length[i]

    #         # if detect is correct, output correct
    #         detect_label = detect_dict[sid]
    #         if detect_label == 0:
    #             my_file.write('%s, correct\n' % (sid))

    #         # if detect level is recognized as error
    #         else:
    #             err_flag = False
    #             tmp_list = []
    #             for j in range(length):
    #                 position_label = position_dict[sid][j]

    #                 if position_label != 0 and identification_dict[position_label][sid] != 0:
    #                     my_file.write('%s, %d, %d, %s\n' % (sid, j+1, j+1, error_invdict[position_label]))
    #                     err_flag = True

    #             # cannot find any position
    #             if err_flag == False:
    #                 my_file.write('%s, correct\n' % (sid))

    output_list = []
    for i in range(len(test_sid)):
        sid = test_sid[i]
        length = test_length[i]

            # if detect is correct, output correct
        detect_label = detect_dict[sid]
        if detect_label == 0:
            output_list.append([sid, 0, 0, 'correct'])

            # if detect level is recognized as error
        else:
            err_flag = False
            tmp_list = []
            for j in range(length):
                position_label = position_dict[sid][j]

                if position_label != 0 and identification_dict[position_label][sid] != 0:
                        # my_file.write('%s, %d, %d, %s\n' % (sid, j, j, error_invdict[position_label]))
                    output_list.append([sid, j+1, j+1, error_invdict[position_label]])
                    err_flag = True

            # cannot find any position
            if err_flag == False:
                # my_file.write('%s, correct\n' % (sid))
                output_list.append([sid, 0, 0, 'correct'])

    with open(os.path.join(test_path, 'final_result.txt'), 'w') as my_file:
        handle_id = output_list[0][0]
        start_off = output_list[0][1]
        end_off = output_list[0][2]
        next_label = output_list[0][3]
        for i in range(len(output_list)):
            if output_list[i][3] == 'correct':
                my_file.write('%s, %s\n' % (output_list[i][0], output_list[i][3]))
            else:
                if handle_id == output_list[i][0] and next_label == output_list[i][3]:
                    end_off = output_list[i][2]

                elif handle_id == output_list[i][0] and next_label != output_list[i][3]:
                    my_file.write('%s, %d, %d, %s\n' % (output_list[i][0], start_off, end_off, next_label))

                    start_off = output_list[i][1]
                    end_off = output_list[i][2]
                    next_label = output_list[i][3]

                elif handle_id != output_list[i][0]:
                    if start_off != 0 and end_off != 0:
                        my_file.write('%s, %d, %d, %s\n' % (output_list[i][0], start_off, end_off, next_label))

                    handle_id = output_list[i][0]
                    start_off = output_list[i][1]
                    end_off = output_list[i][2]
                    next_label = output_list[i][3]


                # if continue_flag == True and previous_label == output_list[i][3]:
                #     end_off = output_list[i][2]
                #     previous_label = output_list[i][3]

                # if continue_flag == True and previous_label != output_list[i][3]:
                #     my_file.write('%s, %d, %d, %s\n' % (output_list[i][0], start_off, end_off, previous_label))

                #     start_off = output_list[i][1]
                #     end_off = output_list[i][2]
                #     continue_flag = True

                # if continue_flag == False:
                #     start_off = output_list[i][1]
                #     end_off = output_list[i][2]
                #     previous_label = output_list[i][3]
                #     continue_flag = True
