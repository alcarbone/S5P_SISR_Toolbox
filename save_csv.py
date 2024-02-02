# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
           This script is used to print results in .csv files
           in /results directory.
"""

#%%   

if protocol == 'RR':  
    if not (os.path.isfile(dirs_res_path  + '/' + im_tag + '_quality_indices.csv')):  
        names = ['Method','Q','ERGAS','sCC','PSNR']
        f = open(dirs_res_path  + '/'+ im_tag + '_quality_indices.csv', 'w') 
        wr = csv.writer(f, delimiter = ',', lineterminator='\n',)
        wr.writerow(names)  
    l = [] 
    l.append(['GT',1,0,1,float('inf')])
    l.append(['Cubic',Q_SR_int,ERGAS_SR_int,sCC_SR_int,PSNR_SR_int]) 
    l.append(['SRCNN',Q_SR_SRCNN0,ERGAS_SR_SRCNN0,sCC_SR_SRCNN0,PSNR_SR_SRCNN0]) 
    l.append(['PAN',Q_SR_PAN,ERGAS_SR_PAN,sCC_SR_PAN,PSNR_SR_PAN])
    l.append(['HAT',Q_SR_HAT,ERGAS_SR_HAT,sCC_SR_HAT,PSNR_SR_HAT])
    l.append(['CGA_nomatch',Q_SR_CGA_nomatch,ERGAS_SR_CGA_nomatch,sCC_SR_CGA_nomatch,PSNR_SR_CGA_nomatch])
    l.append(['CGA_match',Q_SR_CGA_match,ERGAS_SR_CGA_match,sCC_SR_CGA_match,PSNR_SR_CGA_match])
    l.append(['S5Net_nomatch',Q_SR_S5Net_nomatch,ERGAS_SR_S5Net_nomatch,sCC_SR_S5Net_nomatch,PSNR_SR_S5Net_nomatch])
    l.append(['S5Net_match',Q_SR_S5Net_match,ERGAS_SR_S5Net_match,sCC_SR_S5Net_match,PSNR_SR_S5Net_match])
    l.append(['S5Net_cubic',Q_SR_S5Net_nodec,ERGAS_SR_S5Net_nodec,sCC_SR_S5Net_nodec,PSNR_SR_S5Net_nodec])
    wr.writerows(l)
    f.close()
else:
    if not (os.path.isfile(dirs_res_path  + '/' + im_tag + '_quality_indices.csv')):  
        names = ['Method','BRISQUE']
        f = open(dirs_res_path  + '/'+ im_tag + '_quality_indices.csv', 'w') 
        wr = csv.writer(f, delimiter = ',', lineterminator='\n',)
        wr.writerow(names)  
    l = [] 
    l.append(['GT',0])
    l.append(['Cubic',BRISQUE_SR_int[i]]) 
    l.append(['SRCNN',BRISQUE_SR_SRCNN0]) 
    l.append(['PAN',BRISQUE_SR_PAN])
    l.append(['HAT',BRISQUE_SR_HAT])
    l.append(['CGA_nomatch',BRISQUE_SR_CGA_nomatch])
    l.append(['CGA_match',BRISQUE_SR_CGA_match])
    l.append(['S5Net_nomatch',BRISQUE_SR_S5Net_nomatch,PSNR_SR_S5Net_nomatch])
    l.append(['S5Net_match',BRISQUE_SR_S5Net_match])
    l.append(['S5Net_cubic',BRISQUE_SR_S5Net_nodec])
    wr.writerows(l)
    f.close()