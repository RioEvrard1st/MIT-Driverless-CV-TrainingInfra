import torch.nn as nn

def print_tensor_stats(x, name):
    flattened_x = x.cpu().detach().numpy().flatten()
    avg = sum(flattened_x)/len(flattened_x)
    print(f"\t\t\t\t{name}: {round(avg,10)},{round(min(flattened_x),10)},{round(max(flattened_x),10)}")

class CSPResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPResNetBlock, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels//2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels//2)
        
        # Final convolutional layer
        self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu5 = nn.ReLU()
        
    def forward(self, x):
        # First convolutional block
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        act1 = self.relu1(b1)
        c2 = self.conv2(act1)
        b2 = self.bn2(c2)
        
        # Second convolutional block
        c3 = self.conv3(x)
        b3 = self.bn3(c3)
        act3 = self.relu3(b3)
        c4 = self.conv4(act3)
        b4 = self.bn4(c4)
        
        # Concatenation and final convolution
        out = self.relu5(self.bn5(self.conv5(torch.cat((b2, b4), dim=1))))
        
        return out
