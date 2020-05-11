import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import * from preprocessing


class PeptideData(Dataset):
    def __init__(self, df_path, target_dic):
        # paths to all images and masks
        self.df = pd.read_csv(df_path)
        self.target_dic = target_dic
        self.inequality_dic = {"=":1,"<":2,">":3}
        self.df_input = np.asarray(self.df.iloc[:,0])
        self.df_output_label = np.asarray(self.df.iloc[:,1:3])
        self.df_output_value = np.asarray(self.df.iloc[:,3], dtype=np.float64)
        self.df_output_value[self.df_output_value>50000] = 50000
        self.df_output_value = 1 - np.log(self.df_output_value)/np.log([50000])
        self.data_len = self.df_input.shape[0]
    def __getitem__(self, index):
        # INPUT
        # Find peptide
        peptide = self.df_input[index]

        # Make 15 length peptide
        peptide = fifteen_mer(peptide)

        # Peptide Encoding      
        peptide = BLOSUM62(peptide)
        peptide = np.asarray(peptide)
        peptide = peptide.reshape(1,15,21)
      
        
        # OUTPUT
        output_data_label = self.df_output_label[index,:]
        output_data_label[0] = self.target_dic[output_data_label[0]]
        output_data_label[1] = self.inequality_dic[output_data_label[1]]
        output_data_label = output_data_label.astype('uint8')
        output_data_value = np.asarray(self.df_output_value[index])


        return torch.from_numpy(peptide).float(), torch.from_numpy(output_data_label), torch.from_numpy(output_data_value)

    def __len__(self):
        return self.data_len
    
