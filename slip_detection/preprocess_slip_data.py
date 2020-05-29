import pandas as pd
import numpy as np
from scipy.io import loadmat
from joblib import Parallel, delayed

import argparse
np.random.seed(0)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger()


parser = argparse.ArgumentParser(description="Data preprocessor.")
parser.add_argument(
    "--save_dir", type=str, help="Location to save data to.", required=True
)
parser.add_argument(
    "--data_path", type=str, help="Path to tactile dataset.", required=True
)

parser.add_argument(
    "--threshold", type=int, help="Threshold for tactile.", required=True
)

parser.add_argument(
    "--bin_duration", type=float, help="Binning duration.", required=True
)

args = parser.parse_args()

class TrajStartEnd():
    def __init__(self, obj_name, path = '/datasets/eventdata/slip2/'):
        self.path = path
        self.obj_name = obj_name
        self.obj_path = self.path + 'traj_start_ends/' + obj_name + '.startend'
        self.traj_start_end = np.array(open(self.obj_path, "r").read().split(" ")).astype(float)
        
class TactileData():
    def __init__(self, obj_name, selection="full", path = ''):
        self.path = path
        self.obj_name = obj_name
#         self.path2 = traj_start_end_path
        self.file_path_tac = self.path + 'aces_recordings/' + obj_name + '.tact'

        trajStartEnd = TrajStartEnd(obj_name, path)
        self.traj_start_end = trajStartEnd.traj_start_end

        # read tact file
        df = pd.read_csv(self.file_path_tac, delimiter=' ', names=['polarity', 'cell_index', 'timestamp_sec', 'timestamp_nsec'], dtype=int)
        df = df.assign(timestamp=df.timestamp_sec + df.timestamp_nsec/1000000000)
        df = df.drop(['timestamp_sec', 'timestamp_nsec'], axis=1)

        # timestamp start from zero
#         print(self.traj_start_end[1], df.timestamp[0])
        df.timestamp = df.timestamp - self.traj_start_end[0]
        print(self.traj_start_end[1], df.timestamp[0])
        if selection == 'full':
            self.start_t = df.timestamp[1]
            _T = 0.15

        self.df = df
#         print(self.start_t, self.start_t + _T)
        #self.df = df[(df.timestamp >= mask_start) & (df.timestamp <= mask_end)]
        #self.df.timestamp = self.df.timestamp
        # Cells are not used: 17 and 49

        # binarization
        self.T = _T #4 #0.75 # 9 # 20
        self.bin_duration = 0.001
        self.threshold = 0
        self.bin_number = int(np.floor(self.T/self.bin_duration))
        self.binarized=False
        self.data_matrix = np.zeros([80,2,self.bin_number], dtype=int)

    def binarize(self, bin_duration = 0.001):

        # check if already binarized
        if self.binarized and self.bin_duration == bin_duration:
            return self.data_matrix

        self.bin_duration = bin_duration
        self.bin_number = int(np.floor(self.T/self.bin_duration))
#         print(self.T/self.bin_duration, self.bin_number)
        self.data_matrix = np.zeros([80,2,self.bin_number], dtype=int)

        pos_df = self.df[self.df.polarity==1]
        neg_df = self.df[self.df.polarity==0]

        end_t = self.start_t + self.bin_duration
        count = 0

        init_t = self.start_t

        while(end_t <= self.T + init_t): # start_t <= self.T
#             print('Its:', count, self.start_t, end_t)

            _pos_count = pos_df[( (pos_df.timestamp >= self.start_t ) & (pos_df.timestamp < end_t ) )]
            _pos_selective_cells = _pos_count.cell_index.value_counts() > self.threshold
            if len(_pos_selective_cells):
                self.data_matrix[_pos_selective_cells[_pos_selective_cells].index.values - 1, 0, count] = 1

            _neg_count = neg_df[( (neg_df.timestamp >= self.start_t ) & (neg_df.timestamp < end_t ) )]
            _neg_selective_cells = _neg_count.cell_index.value_counts() > self.threshold
            if len(_neg_selective_cells):
                self.data_matrix[_neg_selective_cells[_neg_selective_cells].index.values - 1, 1, count] = 1

            self.start_t = end_t
            end_t += self.bin_duration
            count += 1

        self.binarized = True

        self.data_matrix = np.delete(self.data_matrix, [16, 48], 0)
        return self.data_matrix

    def clean(self):
        self.data_matrix=[]
        self.binarized=False
        
        
class CameraData():
    def __init__(self, obj_name, selection="full", path = ''):
        self.path = path

        # propophesee hyperparameters
        self.c = 2
        self.w = 200#300
        self.h = 250#350      
        x0 = 180#230#180
        y0 = 0#100#0
        file_path = path + "prophesee_recordings/" + obj_name #+ ".start" # _td.mat
        start_time = float(open(file_path + ".start", "r").read())
        
        trajStartEnd = TrajStartEnd(obj_name, path)
        self.traj_start_end = trajStartEnd.traj_start_end
        delta_time = start_time - self.traj_start_end[0]
        self.traj_start_end += delta_time
        self.traj_start_end = self.traj_start_end - self.traj_start_end[0]
        
        if selection == 'full':
            self.start_t = self.traj_start_end[1]
            _T = 0.15
        
        td_data = loadmat(file_path + "_td.mat")['td_data']
        df=pd.DataFrame(columns=['x', 'y', 'polarity', 'timestamp'])
        a = td_data['x'][0][0]
        b = td_data['y'][0][0]

        mask_y = (b >= 100)
        mask_x = (a >= 230) & (a < 430)
        a1 = a[ mask_x & mask_y ] - 230
        b1 = b[ mask_x & mask_y ] - 100
        df.x = a1.flatten()
        df.y = b1.flatten()
#         df.x = td_data['x'][0][0].flatten() - x0 # x coordinate
#         df.y = td_data['y'][0][0].flatten() - y0 # y coordinate
        df.polarity = td_data['p'][0][0][ mask_x & mask_y ].flatten() # polarity with value -1 or 1
        df.timestamp = td_data['ts'][0][0][ mask_x & mask_y ].flatten()/1000000.0 # spiking time in microseconds, convert to seconds
        df = df.reset_index()
        self.df = df
        
        self.T = _T #20
        self.bin_duration = -1.0 #0.002
        self.bin_number = -1 #int(np.floor(self.T/self.bin_duration))
        self.binarized = False
        
        self.data_matrix = []#np.zeros([self.c, self.w, self.h, self.bin_number], dtype=int)
        self.threshold = args.threshold
        
    def binarize(self, bin_duration = 0.01):
        
        self.bin_duration = bin_duration
        self.bin_number = int(np.floor(self.T/self.bin_duration))
#         print(self.T/self.bin_duration, self.bin_number, self.T)
        self.data_matrix = np.zeros([self.c, self.w, self.h, self.bin_number], dtype=int)
        #print(self.data_matrix.shape)
        # self.data_matrix = np.zeros([self.c, 250, 200, self.bin_number], dtype=int)
        
        pos_df = self.df[self.df.polarity==1]
        neg_df = self.df[self.df.polarity==-1]
        #print(pos_df.shape, neg_df.shape)
        
        end_t = self.start_t + self.bin_duration
        count = 0
        
        init_t = self.start_t
        
        while(end_t <= self.T + init_t): # start_t <= self.T
#             print('Its:', count, self.start_t, end_t)
        
            _pos_count = pos_df[( (pos_df.timestamp >= self.start_t ) & (pos_df.timestamp < end_t ) )]
#             print(_pos_count.shape)

            b = pd.DataFrame(index=_pos_count.index)
            #print(_pos_count['y'])
            b = b.assign(xy = _pos_count['x'].astype(str) + '_' + _pos_count['y'].astype(str))
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            #print(some_array)
            xy = np.array(list(map(lambda x: x.split('_'), some_array))).astype(int)
            if xy.shape[0] > 0:
                self.data_matrix[0,xy[:,0],
                                   xy[:,1], count] = 1
                
            _neg_count = neg_df[( (neg_df.timestamp >= self.start_t ) & (neg_df.timestamp < end_t ) )]
            b = pd.DataFrame(index=_neg_count.index)
            b = b.assign(xy = _neg_count['x'].astype(str) + '_' + _neg_count['y'].astype(str))
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            xy = np.array(list(map(lambda x: x.split('_'), some_array))).astype(int)
            if xy.shape[0] > 0:
                self.data_matrix[1,xy[:,0],
                                  xy[:,1], count] = 1
            # print(_pos_count.shape[0] + _neg_count.shape[0])
            


            self.start_t = end_t
            end_t += self.bin_duration
            count += 1
            
        self.binarized = True
        self.data_matrix = np.swapaxes(self.data_matrix, 1,2)
        
        return self.data_matrix
    
    def clean(self):
        self.data_matrix=[]
        self.binarized=False
        
def tact_bin_save(file_name, overall_count, bin_duration, path, selection, save_dir):
    tac_data = TactileData(file_name, selection, path)
    tacData = tac_data.binarize(bin_duration)
    tac_data.clean()
    np.save(save_dir + str(overall_count) + '_tact.npy', tacData.astype(np.uint8))
    
    
def vis_bin_save(file_name, overall_count, bin_duration, path, selection, save_dir):
    cam_data = CameraData(file_name, selection, path)
    visData = cam_data.binarize(bin_duration)
    np.save(save_dir + str(overall_count) + '_vis.npy', visData.astype(np.uint8))
    cam_data.clean()
    
class ViTacData():
    def __init__(self, ViTacDataPath, save_dir, list_of_objects, selection="full"):
        self.path = ViTacDataPath
        self.list_of_objects = list_of_objects
        self.iters = 50
        self.bin_duration = 0.01
        self.save_dir = save_dir #
        self.selection = selection
        
    def binarize_save(self, bin_duration =0.01, modality=2):
        self.bin_duration = bin_duration
        'saves binarized tactile and prophesee data'
        current_label = 0
        overall_count = 0
        big_list_tact = []
        big_list_vis = []
        for obj in self.list_of_objects:
            print(obj, ' being processed ...')
            self.iters = 50
            for i in range(1, self.iters+1): #tqdm(range(1, self.iters+1)):
                
                i_str = str(i)
                if i < 10:
                    i_str = '0' + i_str
                
                file_name = obj + '_' + i_str
                # tactile
                if modality == 2 or modality == 0:
                    big_list_tact.append([file_name, overall_count, self.bin_duration,
                                     self.path, self.selection, self.save_dir])

                if modality == 2 or modality == 1:
                    big_list_vis.append([file_name, overall_count, self.bin_duration,
                                     self.path, self.selection, self.save_dir])
#                     cam_data = CameraData(file_name, self.selection, self.path)
#                     visData = cam_data.binarize(self.bin_duration)
#                     np.save(self.save_dir + str(overall_count) + '_vis.npy', visData.astype(np.uint8))
#                     cam_data.clean()
                overall_count+=1
            current_label+=1
        if modality == 2 or modality == 0:
            Parallel(n_jobs=18)(delayed(tact_bin_save)(zz[0], zz[1], zz[2], zz[3], zz[4], zz[5]) for zz in big_list_tact)
            print('Tactile has done')
        if modality == 2 or modality == 1:
            Parallel(n_jobs=18)(delayed(vis_bin_save)(zz[0], zz[1], zz[2], zz[3], zz[4], zz[5]) for zz in big_list_vis)
            
# batch2
list_of_objects2 = [
    'stable',
    'rotate'
]

data_loc = args.data_path
save_dir_to = args.save_dir

ViTac = ViTacData(data_loc, save_dir_to, list_of_objects2, selection="full")
ViTac.binarize_save(bin_duration=args.bin_duration, modality=2)
print('Binning has done ...')

# create labels
labels = []
current_label = -1
overall_count = -1
for obj in list_of_objects2:
    current_label += 1
    for i in range(0, 50):
        overall_count+=1
        labels.append([overall_count, current_label])
labels = np.array(labels)

from sklearn.model_selection import StratifiedKFold

# stratified k fold
skf = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)
train_indices = []
test_indices = []
for train_index, test_index in  skf.split(np.zeros(len(labels)), labels[:,1]):
    train_indices.append(train_index)
    test_indices.append(test_index)
    #print("TRAIN:", train_index, "TEST:", test_index)
    
# write to the file
splits = ['80_20_1','80_20_2','80_20_3','80_20_4', '80_20_5']
count = 0
for split in splits:
    np.savetxt(save_dir_to + 'train_' + split + '.txt', np.array(labels[train_indices[count], :], dtype=int), fmt='%d', delimiter='\t')
    np.savetxt(save_dir_to + 'test_' + split + '.txt', np.array(labels[test_indices[count], :], dtype=int), fmt='%d', delimiter='\t')
    count += 1