import torch


import numpy
import pprint
import matplotlib.pyplot as plt





if __name__ == "__main__":
    checkpoint = torch.load("loss_log_original_3x.pt")
    
    x=checkpoint.numpy()
    
    epochs=[]
    loss=[]
    rows=x.shape[0]
    columns=x.shape[1]
    print(x.shape)
    for i in range(rows):
        for j in range(columns):
            
            epochs.insert(i,i)
            loss.insert(i,x[i][j])

    checkpoint1 = torch.load("loss_log_2_blocks_customized_3x.pt")
    print("enter")
    x1=checkpoint1.numpy()
    
    epochs1=[]
    loss1=[]
    rows1=x1.shape[0]
    columns1=x1.shape[1]
    for l in range(rows):
        for m in range(columns):
            
            epochs1.insert(l,l)
            loss1.insert(l,x1[l][m])


    plt.plot(epochs,loss,label = "original 3x",linestyle="-")
    plt.plot(epochs1,loss1,label = "model 7 3x " ,linestyle="-.")
    plt.title('Loss curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


   
    
