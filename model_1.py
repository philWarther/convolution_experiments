#finite difference

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from scipy.ndimage.morphology import distance_transform_edt as dist 
import matplotlib.pyplot as plt

class build_model(nn.Module):
    global in_channel
    global out_channel
    global kernel_size
    global padding
    global stride
    global n

    def __init__(self):
        super(build_model, self).__init__()
        self.conv_layer1 = nn.Conv1d(in_channel,out_channel,kernel_size, stride, padding, bias=False)
        #self.input_layer   = nn.Linear(n,n)
        self.output_layer = nn.Linear(n,n)

    def forward(self,input_data):
        conv_1_out          = self.conv_layer1(input_data)
        #linear_out         = self.input_layer(input_data[0][0])
        sig_1               = nn.Tanh()(conv_1_out)#+linear_out)
        output              = self.output_layer(sig_1)
        return output

def data_sequence():
    global limit
    global steps
    global n 

    dist_data = torch.Tensor(steps,n)
    bdy_data = torch.Tensor(steps,n)
    boundary = np.random.uniform( dx, .2*limit )
    for t in range(steps):
        boundary_indicator = np.zeros( n ) 
        boundary_indicator[ np.argmin( abs( domain - boundary ) ) ]    = 1
        boundary_indicator[ np.argmin( abs( domain + boundary ) ) ]    = 1
        phi = dx*dist(boundary_indicator - 1)
        phi[abs(domain)<boundary] = -phi[abs(domain)<boundary]

        dist_data[t] = torch.from_numpy(phi)
        bdy_data[t] = torch.from_numpy(boundary_indicator)
        boundary += abs(boundary - limit)/(3+t)
    return dist_data,bdy_data
def find_bdy(var):
    size = var.data.size()
    bdy_binary = torch.zeros(size)
    for i in range(1,size[2]):
        if var.data[0][0][i]*var.data[0][0][i-1] >0:
            bdy_binary[0][0][i-1] = 0
        else:
            bdy[0][0][i] = 1

def GradUpdate(learning_rate):
    for param in model.parameters():
        param.data -=learning_rate*param.grad.data
        param.grad.data.zero_()

def Learning_rate_update(): # define rule for updating the learning rate
    global learning_rate
    if np.average(loss_data[i-10:i])>.8*np.average(loss_data[i-20:i-11]):
        learning_rate = .5*learning_rate  

def compute_loss(output,bdy_target):
    error = Variable(output.data * bdy_target.data)
    loss = loss_fn(error, Variable(torch.zeros(n)))
    return loss

# Define the physical domain of [-limit, limit] partitioned into n regular sub-intervals 
# where dx is the length of each sub-interval

limit   = 10 
n       = 200 #data length
domain  = np.linspace(-limit,limit,n)
dx      = 2*limit/n 

# Define the time step information where 0<t<T and dt the size of each successive time step
dt            = .025
T             = 1   
time          = np.arange(0,T,dt)
steps         = int(T/dt)


# Convolution layer parameters
in_channel          = 5 # The number of in channels gives the number of previous time steps in the input
out_channel         = 1
kernel_size         = 5
padding             = 2
stride              = 1

loss_fn             = nn.MSELoss()
iterations          = 500
model               = build_model()
batch_size          = 1
alpha               = .1
learning_rate       = alpha*dx

loss_data = np.zeros(iterations)

#for param in model.parameters():
 #   print(param.data)


for i in range(0,iterations):
    loss                = 0
    dist_data, bdy_data = data_sequence()
    for t in range(in_channel+1,steps):
        x_in        = dist_data[ t - in_channel-1:t - 1 ].view( 1,in_channel,n)
        x_var       = Variable(x_in)
        target_var  = Variable(dist_data[t].view(1,1,n))
        #bdy_var    = Variable(bdy_data[t].view(1,1,n))
        output      = model.forward(x_var)
        loss        += loss_fn(output,target_var)
        #loss       += compute_loss(output,bdy_var)
        #print(loss2)
    loss.backward()
    loss_data[i] = loss 
    if i%20 == 0 and i >0:
        Learning_rate_update()
        print('t iteration ', i , 'learning rate was updated to ', learning_rate)  
    if i%100 == 0:
        print(i)
        print('loss = ',loss.data[0])
        
    GradUpdate(learning_rate)
print('Final Loss = ', loss.data[0])
#print('final weights')
#for param in model.parameters():
 #   print(param.data)
plt.close()
plt.figure(1)
plt.title('Model Convergence')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss_data[3:iterations - 1])
plt.show(block=False)

