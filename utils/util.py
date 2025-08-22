import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import os 
import skimage.io as io

def save_img_inter(out_path,img_tensor,data_type,patientSliceID,exp):
    if data_type=='img':
        img_np = img_tensor.cpu().data.numpy().squeeze()
    elif data_type=='mask':
        # img_np = img_tensor.cpu().data.numpy().astype("uint8").squeeze()
        img_np = img_tensor.cpu().data.numpy().squeeze()
        img_np = img_np * 255
    sum_a = np.sum(img_np)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    io.imsave('{}/{}_{}{:.1f}.png'.format(out_path,patientSliceID,exp,sum_a),img_np)

def plot_dot(m_indx,m_value,num=0,):
    plt.plot(m_indx, m_value, 'ks')
    show_max = '[' + str(m_indx) + ',' + str("{:.4f}".format(m_value)) + ']'
    plt.annotate(show_max, xytext=(m_indx, m_value+num), xy=(m_indx, m_value))

def plot_dice2(train_c,valid_c,base_dir,mode,valid_c_3d=[],interval_list=[]):
    train_x = range(len(train_c))
    train_y = train_c
    plt.plot(train_x, train_y)
    
    valid_x = range(len(valid_c))
    valid_y = valid_c
    plt.plot(valid_x, valid_y)

    m_indx=np.argmax(valid_c)
    m_indx_3d=np.argmax(valid_c_3d)
    valid_x_3d_init = list(range(len(valid_c_3d)))
    valid_x_3d = interval_list
    valid_y_3d = valid_c_3d
    plt.plot(valid_x_3d, valid_c_3d)
    if m_indx_3d < int(len(valid_c_3d)*0.8):
        plot_dot(interval_list[m_indx_3d],valid_c_3d[m_indx_3d])
    last_indx_3d = valid_x_3d_init[-1]
    v_last_2d = valid_c[-1]
    v_last_3d = valid_c_3d[-1]
    abs_vLast = abs(v_last_3d-v_last_2d)
    if abs_vLast < 0.04:
        num = 0.04-abs_vLast if v_last_3d > v_last_2d else -(0.06-abs_vLast)
        plot_dot(interval_list[last_indx_3d],valid_y_3d[last_indx_3d],num)
    else:
        plot_dot(interval_list[last_indx_3d],valid_y_3d[last_indx_3d])
    plt.legend(['train', 'val','val_3d'],loc='upper left')

    if m_indx < int(len(valid_c)*0.8):
        plot_dot(m_indx,valid_c[m_indx])
    last_indx = valid_x[-1]
    plot_dot(last_indx,valid_c[last_indx])
    plt.ylabel(mode + ' value')
    plt.xlabel('epoch')
    plt.title("Model " + mode)
    plt.savefig('{}/{}-{:.4f}.jpg'.format(base_dir,mode,valid_c[m_indx]))
    plt.close()


def plot_dice(train_c,valid_c,base_dir,mode,valid_c_3d=[],interval=0):
    train_x = range(len(train_c))
    train_y = train_c
    plt.plot(train_x, train_y)
    
    valid_x = range(len(valid_c))
    valid_y = valid_c
    plt.plot(valid_x, valid_y)

    m_indx=np.argmax(valid_c)
    #Dice 3d val
    m_indx_3d=np.argmax(valid_c_3d)
    valid_x_3d_init = list(range(len(valid_c_3d)))
    valid_x_3d = [x*interval for x in valid_x_3d_init]
    valid_y_3d = valid_c_3d
    plt.plot(valid_x_3d, valid_c_3d)
    if m_indx_3d < int(len(valid_c_3d)*0.8):
        plot_dot(m_indx_3d*interval,valid_c_3d[m_indx_3d])
    last_indx_3d = valid_x_3d_init[-1]
    v_last_2d = valid_c[-1]
    v_last_3d = valid_c_3d[-1]

    abs_vLast = abs(v_last_3d-v_last_2d)
    if abs_vLast < 0.04:
        num = 0.04-abs_vLast if v_last_3d > v_last_2d else -(0.06-abs_vLast)
        plot_dot(last_indx_3d*interval,valid_y_3d[last_indx_3d],num)
    else:
        plot_dot(last_indx_3d*interval,valid_y_3d[last_indx_3d])
    plt.legend(['train', 'val','val_3d'],loc='upper left')

    if m_indx < int(len(valid_c)*0.8):
        plot_dot(m_indx,valid_c[m_indx])

    last_indx = valid_x[-1]
    plot_dot(last_indx,valid_c[last_indx])
    plt.ylabel(mode + ' value')
    plt.xlabel('epoch')
    plt.title("Model " + mode)
    plt.savefig('{}/{}-{:.4f}.jpg'.format(base_dir,mode,valid_c[m_indx]))
    plt.close()

def plot_base(train_c,valid_c,base_dir,mode):
    train_x = range(len(train_c))
    train_y = train_c
    plt.plot(train_x, train_y)
    if len(valid_c)>0:
        valid_x = range(len(valid_c))
        valid_y = valid_c
        plt.plot(valid_x, valid_y)

        last_indx = valid_x[-1]
        plot_dot(last_indx,valid_c[last_indx])
        plt.legend(['train', 'val'],loc='upper left')
    else:
        plt.legend(['train'],loc='upper left')
    last_indx = train_x[-1]
    plot_dot(last_indx,train_c[last_indx])
    
    plt.ylabel(mode + ' value')
    plt.xlabel('epoch')
    plt.title("Model " + mode)
    plt.savefig('{}/{}-{:.4f}.jpg'.format(base_dir,mode,train_c[last_indx]))
    plt.close()

def plot_dice_loss(train_dict,val_dict,val_3d_interval,lr_curve,base_dir):
    # plot dice curve
    print(val_dict['dice_3d'])
    plot_dice(train_dict['dice'],val_dict['dice'],base_dir,'Dice',val_dict['dice_3d'],val_3d_interval)
    # plot loss curve
    for key in train_dict:
        if 'loss' in key:
            if key in val_dict:
                plot_base(train_dict[key],val_dict[key],base_dir,mode=key)
            else:
                plot_base(train_dict[key],[],base_dir,mode=key)

    # plot lr curve
    lr_x = range(len(lr_curve))
    lr_y = lr_curve
    plt.plot(lr_x, lr_y)
    plt.legend(['learning_rate'],loc='upper right')
    plt.ylabel('lr value')
    plt.xlabel('epoch')
    plt.title("Learning Rate" )
    plt.savefig('{}/lr.jpg'.format(base_dir))
    plt.close()   

def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(levelname)s: %(message)s'
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler() 
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(log_path) 
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

class Logger(object):
    def __init__(self, log_path="Default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def read_list(list_path):
    list_data = []
    with open(list_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  
            list_data.append(line)  
    return list_data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.count_dict = {}
        self.value_dict = {}
        self.res_dict = {}

    def add_value(self,tag_dict,n=1):
        for tag,value in tag_dict.items():
            if tag not in self.value_dict.keys():
                self.value_dict[tag] = 0
                self.count_dict[tag] = 0
            self.value_dict[tag] += value
            self.count_dict[tag] += n

    def updata_avg(self):
        for tag in self.value_dict:
            if tag not in self.res_dict.keys():
                self.res_dict[tag] = []
            avg = self.value_dict[tag] / self.count_dict[tag]
            self.res_dict[tag].append(avg)
        self.count_dict = {}
        self.value_dict = {}


if __name__ == "__main__": 
    log = AverageMeter()
    # log.add_value({"loss": 0.1},n=1)
    # log.add_value({"sup loss": 0.2},n=1)
    log.add_value({"loss": 0.1, "sup loss": 0.2},n=1)
    log.add_value({"loss": 0.1, "sup loss": 0.2},n=1)
    log.updata_avg() 
    print(log.res_dict)
    log.add_value({"loss": 0.1, "sup loss": 0.2},n=1)
    log.add_value({"loss": 0.1, "sup loss": 0.2},n=1)
    log.updata_avg() 
    print(log.res_dict)
    