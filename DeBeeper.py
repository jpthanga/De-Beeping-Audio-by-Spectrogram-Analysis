import scipy.io.wavfile
import numpy as np
import math
import matplotlib.pyplot as plt

a = scipy.io.wavfile.read("./x.wav")

A = np.array(a[1])
mean = np.mean(A)
S = A.size
N = 256
n2 = int(256/2)

#make DFT
Xs = np.arange(N)
m = np.exp(-2j*np.pi*Xs/N).reshape(-1,1) ** Xs

print(m.shape)
#create hann
H = []

for i in range(0,N):
    H.append(math.sin((math.pi*i)/(N-1))**2)
H = np.array(H)

A = np.reshape(A,(496,n2))

X = []

for i in range(0,495):
    I = np.concatenate((A[i],A[i+1]),axis=0)
    X = np.concatenate((X,I*H))

X = np.transpose(X.reshape((495,N)))

FX = np.dot(m,X)


im = plt.matshow(np.real(FX),cmap='Blues',aspect='auto')
plt.show()
#remove and add 0s
for i in range(0,495):
    FX[30][i] = 0
    FX[31][i] = 0
    FX[32][i] = 0
    FX[33][i] = 0
    FX[34][i] = 0

    FX[222][i] = 0
    FX[223][i] = 0
    FX[224][i] = 0
    FX[225][i] = 0
    FX[226][i] = 0

im = plt.matshow(np.real(FX),cmap='Blues',aspect='auto')
plt.show()

Finv = np.linalg.inv(m)

#adds distortion so using above one
# Finv = (np.exp(2j*np.pi*Xs/N)**m)/N

Xbar = np.transpose(np.dot(Finv,FX))
Xbar = np.real(Xbar)

G = np.array(Xbar[0][0:128])

for i in range(0,494):
    I = Xbar[i][128:256] + Xbar[i+1][0:128]
    G = np.concatenate((G,I),axis=0)

G = np.concatenate((G,Xbar[494][128:256]))
G = G.astype(np.int16)
scipy.io.wavfile.write("out.wav",16000,G)



