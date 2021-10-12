import os
import joblib

import numpy as np

import matplotlib.pyplot as plt


# sample_list = joblib.load('sample_list_dhd_traffic_all_with_feature_map.s')
# # print(len(sample_list))

# joblib.dump(sample_list[0].feature[0], 'feature_test.tmp')
feature = joblib.load('feature_test.tmp')

feature = feature.astype(np.float32)

# for sample in sample_list:
#     feature = sample.feature
#     feature_figure = feature[0][0]
#     plt.imshow(feature_figure)
#     plt.savefig('feature.pdf')
#     break

print(feature.shape)
print(feature.dtype)

for i in range(32):
    plt.imshow(feature[i])
    plt.savefig(os.path.join('./features', 'feature' + str(i) + '.jpg'))
