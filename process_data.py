import pandas as pd 
import numpy as np 
import glob
def process():
    """
    this function is processing csv file for weka processing
    """
    data_ben = pd.read_csv("cantina_dataset/cantina_feature_ben_addend.csv",index_col=[0])
    data_mal = pd.read_csv("cantina_dataset/cantina_feature_addend.csv",index_col=[0])

    data_ben["label"] = 0
    data_mal["label"] = 1
    data_mal = data_mal[~data_mal.index.duplicated(keep='first')]
    #add age time for data ben
    for id in data_ben.index:
        if pd.isnull(data_ben.loc[id,"age"]):
            data_ben.loc[id,"age"] = -1
    
    #add age time for data mal
    for id in data_mal.index:
        if pd.isnull(data_mal.loc[id,"age"]):
            data_mal.loc[id,"age"] = -1
        else:
            days = data_mal.loc[id,"age"].split(" ")[0]
            data_mal.loc[id,"age"] = days
    data_ben.to_csv("cantina_dataset/cantina_feature_ben_addend_weka.csv")
    data_mal.to_csv("cantina_dataset/cantina_feature_addend_weka.csv")

def concat_phishpath():
    file = pd.read_csv("phishpath_dataset/cancate_phishpath_weka.csv")
    for id in file.index:
        if file.loc[id, "blacklisted"] == 0:
            file.loc[id, "blacklisted"] = False
        else:
            file.loc[id, "blacklisted"] = True
        
        if file.loc[id, "open_redirect"] == 0:
            file.loc[id, "open_redirect"] = False
        else:
            file.loc[id, "open_redirect"] = True
    
    file.to_csv("phishpath_dataset/catcat_weka_false_true.csv")

def kick_out_unredirected():
    ref_table = pd.read_csv("cantina_dataset/redirection_bl.csv",index_col=[0])
    ref_id = list(ref_table.index)

    data_mal = pd.read_csv("cantina_dataset/cantina_feature_addend_weka.csv", index_col=[0], low_memory=False)
    data_ben = pd.read_csv("cantina_dataset/cantina_feature_ben_addend_weka.csv", index_col=[0],low_memory=False)

    mal_id_list = list(data_mal.index)
    # ben_id_list = data_ben.index

    mal_drop_list = []
    #check if malicious link is redirected link    
    for id in data_mal.index:
        if ref_id.count(id) == 1:
            mal_drop_list.append(id)

    
    dropped_single_link = data_mal.drop(mal_drop_list)
    dropped_single_link.to_csv("cantina_dataset/cantina_feature_addend_drop_sgl_link_weka.csv")
    a = pd.concat([data_ben, dropped_single_link])
    a.to_csv("cantina_dataset/drop_single_link_weka.csv")
    



    



    








if __name__ == "__main__":
    # process()
    # concat_phishpath()
    kick_out_unredirected()
