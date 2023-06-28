import argparse
import math
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import re
import hashlib
import numpy as np
import random
import librosa
import time
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--exp_day', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float,required=True)
    parser.add_argument('--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use other will be unknown label',
      )
    parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
      '--validation_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
      '--testing_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
    args = parser.parse_args()
    return args


                                
def prepare_data_index(silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage,root_data_dir):

    wav_files ={cls_name: [] for cls_name in wanted_words}
    wav_files['_unknown_']=[]
    wav_files['_silence_']=[]
    search_path = os.path.join(root_data_dir, '*', '*.wav')
    for wav_path in glob(search_path):
        _, word = os.path.split(os.path.dirname(wav_path))
        word = word.lower()
        if word == '_background_noise_':
            continue
        if word in wanted_words:
            wav_files[word].append(wav_path)
        else:
            wav_files['_unknown_'].append(wav_path)
    np.random.seed(1)
    
    file_list = {'train':[],'val':[],'test':[]}
    for class_id, c in enumerate(wanted_words):
        n_data = len(wav_files[c])
        
        rindx = np.random.permutation(n_data)
        
        n_validation = int(args.validation_percentage*0.01*n_data)
        n_testing = int((args.testing_percentage+args.validation_percentage)*0.01*n_data)
        val_indx = rindx[:n_validation]
      
        test_indx = rindx[n_validation:n_testing]
        train_indx = rindx[n_testing:]
        
        file_list['train'] += [ (wav_files[c][k], class_id+2) for k in train_indx ] 
        file_list['val'] += [ (wav_files[c][k], class_id+2) for k in val_indx ] 
        file_list['test'] += [ (wav_files[c][k], class_id+2) for k in test_indx ] 
    
    
    
    len_test = len(file_list['test'])
    len_val = len(file_list['val'])
    len_train = len(file_list['train'])

    
    ######################unknown, silence####################
    n_data = len(wav_files['_unknown_'])
    rindx = np.random.permutation(n_data)
    n_validation = int(len_test*args.unknown_percentage*0.01)
    n_testing = int(len_val*args.unknown_percentage*0.01)
    n_training = int(len_train*args.unknown_percentage*0.01)
    
    val_indx = rindx[:n_validation]
    test_indx = rindx[n_validation:n_validation+n_testing]
    train_indx = rindx[n_testing:n_testing+n_training]
    
    
    file_list['train'] += [ (wav_files['_unknown_'][k], 1) for k in train_indx ] 
    file_list['val'] += [ (wav_files['_unknown_'][k], 1) for k in val_indx ] 
    file_list['test'] += [ (wav_files['_unknown_'][k], 1) for k in test_indx ]
    

    
    file_list['train'] += [ (0, 0) for k in train_indx ] 
    file_list['val'] += [ (0, 0) for k in val_indx ] 
    file_list['test'] += [ (0,0) for k in test_indx ] 
    
    return file_list
        
class Mydataset(Dataset):
    
    def __init__(self,filelist):
        self.filelist = filelist
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,index):
        eps=1e-08
        audio_file, class_id = self.filelist[index]
        if class_id == 0:
            audio_wav = np.zeros([16000])
            audio_wav = audio_wav + eps
        else:
            audio_wav, fs = librosa.load(audio_file,sr=16000)
        if audio_wav.shape[0] < 16000:
            empty_wav = np.zeros([16000])
            empty_wav[:audio_wav.shape[0]]=audio_wav
            audio_wav = empty_wav
        
        spec_wav = librosa.stft(audio_wav,n_fft=640,hop_length=320,win_length=640,center=False)
        power_spec = np.abs(spec_wav)**2
        mel_spec = librosa.feature.melspectrogram(S=power_spec,sr=16000,n_fft=640,hop_length=320,power=2.0,n_mels=40,fmin=20,fmax=7000)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))
        return mfcc,class_id




class KWS_Net(nn.Module):
    def __init__(self):
        super(KWS_Net, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, (3,3), stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU())
                
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 16, (5,3), stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU())
                
        self.gru = nn.GRU(input_size=256,hidden_size=256,batch_first=True)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 12)

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1) #[B 1 20 49]
        x = x.permute(0,1,3,2) #[B 1 49 20]
        x = self.layer1(x)
        x = self.layer2(x) #[B 16 43 16]
        x = x.permute(0,2,1,3)
        x = x.reshape(-1,43,256)
        x,_ = self.gru(x) #[B 43 256]
        x = x[:,-1,:] #[B 1 256]
        x = x.squeeze(1) #[B 256]
        x = self.dropout(x)
        x = self.fc3(self.fc2((self.relu(self.fc1(x)))))
        
        return x


if __name__ == '__main__':
    
    
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp_day = args.exp_day
    wanted_words = args.wanted_words.split(',') #[yes,no,up,down,left,right,on,off,stop,go]
   
    root_data_dir = '/home/jungwook/kws/DB_data/'
    file_list = prepare_data_index(args.silence_percentage, #10
                                    args.unknown_percentage, #10
                                    wanted_words, #yes,no,up,down,left,right,on,off,stop,go'
                                    args.validation_percentage, #10
                                    args.testing_percentage, #10
                                    root_data_dir) 
        
    
    
    
    
    print(len(file_list['train']))
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    start_epoch = 0
    num_epochs = 500
    best_loss = 10
    
    train_dataset = Mydataset(file_list['train'])
    val_dataset = Mydataset(file_list['val'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    
    tensorboard_path = 'runs/'+str(exp_day)+'/learning_rate_'+str(learning_rate)+'_batch_'+str(batch_size)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    summary = SummaryWriter(tensorboard_path)
    
    model = KWS_Net().to(device)
    #model.load_state_dict(torch.load("model_ckpt/bestmodel.pth",map_location=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=10,min_lr=0.000005)
                                
    for epoch in range(start_epoch,num_epochs):
        start = time.time()
        model.train()
        train_loss=0
        for i, (data,class_id) in enumerate(train_loader):
            data = data.to(device)
            class_id = class_id.to(device)
            outputs = model(data)
            loss = criterion(outputs, class_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, running time: {}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item(),time.time()-start))
            train_loss+=loss.item()
        train_loss = train_loss/len(train_loader)
        summary.add_scalar('training loss',train_loss,epoch)
        
        modelsave_path = 'model_ckpt/'+str(exp_day)+'/learning_rate_'+str(learning_rate)+'_batch_'+str(batch_size)
        if not os.path.exists(modelsave_path):
            os.makedirs(modelsave_path)
        #torch.save(model.state_dict(), str(modelsave_path)+'/epoch'+str(epoch)+'model.pth')
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pth')
        model.eval()
        with torch.no_grad():
            val_loss =0.
            correct = 0
            total = 0
            for j, (data,class_id) in enumerate(val_loader):
                data = data.to(device)
                class_id = class_id.to(device)
                
                outputs = model(data)
                
                _, predicted = torch.max(outputs.data, 1)
                total += class_id.size(0)
                correct += (predicted == class_id).sum().item()
                
                loss = criterion(outputs, class_id)
                print('valEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()
            print('Accuracy of the network on the {} val images: {} %'.format(len(val_loader)*batch_size, 100 * correct / total))    
            val_loss = val_loss/len(val_loader)
            summary.add_scalar('val loss',val_loss,epoch)
            summary.add_scalar('acc',100 * correct / total,epoch)
            summary.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
            scheduler.step(val_loss)
            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pth')
                best_loss = val_loss        
                                
                                
                                
                                
                                
                        
                                
                                
                                
                                
                                
                                
                                
