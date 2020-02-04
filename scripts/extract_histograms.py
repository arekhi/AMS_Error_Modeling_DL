import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle

ea = EventAccumulator('/arekhi.scratch.DL/tmp/tensorboard/mobilenet_v1/')
ea.Reload()
avail_hist = ea.Tags()['histograms']
hist_list = []
for hist_name in avail_hist:
    hist_data = ea.Histograms(hist_name)
    bucket_lims = hist_data[0][2][5]
    bucket_vals = hist_data[0][2][6]
    hist_list.append([hist_name, bucket_lims, bucket_vals])

pickle.dump(hist_list, open('/arekhi.scratch.DL/tmp/mobilenet_v1_hist_list.p', 'wb'))
