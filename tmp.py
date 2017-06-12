# coding: utf-8
import os
import codecs
import pandas as pd

bucket_size = 7

# text_line = ['0'] * 6
# text_line.extend(u'到了内蒙古，终于实现了我的梦想。')

# print(text_line[0:7])

# print(range(1, len(text_line)))
# for i in range(len(text_line) - bucket_size):
#     # text = []
#     print(i)
#     if i < bucket_size:
#         text = ['0'] * (bucket_size - (i+1))
#         text.extend(text_line[0:(i+1)])
#         # print('111')
#     else:
#         end_point = i + bucket_size
#         text = text_line[i:end_point]
#         # print(text_line[8:14])
#         print(end_point)
#         print(text)

# for i in range(bucket_size, len(text_line)):
#     print(text_line[(i-bucket_size):i])


# with codecs.open(os.path.join('data', 'CGED16_HSK_Test_Input.txt'), 'r', 'utf-8') as my_file:
#     tt = my_file.readlines()[0]
#     tt = tt.strip().split('\t')

file_data = os.path.join('data', 'CGED16_TOCFL_Test_Input.txt')
# file = pd.read_table(file_data, header=None, sep='\t', quoting=3)

with codecs.open(file_data, 'r') as my_file:
    for line in my_file.readlines():
        line = line.strip().split('\t')
        try:
            tt = line[1].decode('utf-8')
            print(tt)
        except:
            print(line)
