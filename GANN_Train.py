from __future__ import print_function
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import pandas as pd
import joblib

######Generator model
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_cells),
            nn.BatchNorm1d(hidden_cells),
            nn.ReLU(True),
            nn.Linear(in_features=hidden_cells, out_features=hidden_cells),
            nn.BatchNorm1d(hidden_cells),
            nn.ReLU(True),
            nn.Linear(in_features=hidden_cells, out_features=hidden_cells),
            nn.ReLU(True),
            nn.Linear(in_features=hidden_cells, out_features=1),
        )

    def forward(self, input):
        return self.main(input)

######Discriminator model
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=in_features_D, out_features=hidden_cells_D),
            nn.BatchNorm1d(hidden_cells_D),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=hidden_cells_D, out_features=hidden_cells_D),
            nn.BatchNorm1d(hidden_cells_D),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=hidden_cells_D, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)

######weights initialize
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Liner') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#### input path setting
sample_path = r'./2014_2018_sample_550_Selection.csv'
MinMaxScaler_savepath = r'./scaler.pkl'
model_savepath = r'./model'

#### Parameter setting
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
in_features = 12
hidden_cells = 256
nc = 1
bitchsize = 128
hidden_cells_D = 4
in_features_D = in_features + 3
ngpu = 1
lr = 0.0002
beta1 = 0.5
num_epochs = 100
batch_size = 2048

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

criterion_G = nn.MSELoss()
criterion_D = nn.BCELoss()

### Data processing
data_aod_pd = pd.read_csv(sample_path, header=0)

###Feature processing
data_aod_pd['NDVI'] = (data_aod_pd['b5'] - data_aod_pd['b7']) / (data_aod_pd['b5'] + data_aod_pd['b7'])
data_aod_pd = data_aod_pd[
    ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'NDVI', 'sz', 'vz', 'ra', 'Site_Elevation(m)', 'AOD_AERONET']]
dense_feature = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'NDVI', 'sz', 'vz', 'ra', 'Site_Elevation(m)']
### using Meteorological data
# data_aod_pd = data_aod_pd[
#     ['b1_apr', 'b2_apr', 'b3_apr', 'b4_apr', 'b5_apr', 'b6_apr', 'b7_apr', 'NDVI', 'sz', 'vz',
#      'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 'Site_Elevation(m)','sp','tm','u10','v10', 'AOD_500nm']]
# dense_feature = ['b1_apr', 'b2_apr', 'b3_apr', 'b4_apr', 'b5_apr', 'b6_apr', 'b7_apr', 'NDVI', 'sz', 'vz',
#                  'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)','Site_Elevation(m)','sp','tm','u10','v10']
nms = MinMaxScaler(feature_range=(0, 1))
data_aod_pd[dense_feature] = nms.fit_transform(data_aod_pd[dense_feature])
joblib.dump(nms, MinMaxScaler_savepath)

data_aod_pd = data_aod_pd.dropna(axis=0, how='any')
data_aod_pd = np.array(data_aod_pd)

netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
netG.apply(weights_init)
netD = Discriminator(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
netD.apply(weights_init)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

print('Size train', len(data_aod_pd))

dataset_train = TensorDataset(torch.Tensor(data_aod_pd[:, 0:-1]).float().to(device),
                              torch.Tensor(data_aod_pd[:, -1]).float().to(device))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)

G_losses = []
D_losses = []
Dx_ls = []
Dx_ls_2 = []
iters = 0
real_label = 1.
fake_label = 0.

### GANN traning

for epoch in range(num_epochs):

    for idx, batch_data in enumerate(dataloader_train):
        netD.zero_grad()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch

        x_train = batch_data[0]
        y_train = torch.unsqueeze(batch_data[1], dim=1)
        # real_cpu = torch.unsqueeze(batch_data, dim=2)
        # real_cpu = torch.unsqueeze(real_cpu, dim=3)
        real_cpu = torch.cat((x_train, y_train * 10, y_train, y_train / 10), dim=1)
        output = netD(real_cpu).view(-1)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        errD_real = criterion_D(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        fake = netG(x_train)
        # fake_all = torch.cat((x_train,fake),1)
        fake_all = torch.cat((x_train, fake * 10, fake, fake / 10), dim=1)
        label.fill_(fake_label)
        noise = torch.randn(b_size).to(device)
        y_fake = y_train + noise
        output = netD(fake_all.detach()).view(-1)
        errD_fake = criterion_D(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches

        errD = (errD_real + errD_fake) / 2
        # Update D
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        output_G = netG(x_train)
        # Calculate G's loss based on this output

        errG = criterion_G(output_G, y_train)

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        netG.zero_grad()
        label.fill_(real_label)
        # output = netD(fake_all).view(-1)
        # output_G = torch.unsqueeze(netG(x_train).view(-1), dim=1)
        output_G = netG(x_train)
        fake_all = torch.cat((x_train, output_G * 10, output_G, output_G / 10),
                             dim=1)  # torch.cat((x_train, output_G), 1)
        output_D = netD(fake_all).view(-1)
        # Calculate G's loss based on this output
        errG = criterion_D(output_D, label)

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        if idx % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, idx, len(dataloader_train),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        Dx_ls.append(D_x)
        Dx_ls_2.append(D_G_z1)

        iters += 1

#### Model save
torch.save(netG, model_savepath)
