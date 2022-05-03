import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)




if __name__ == "__main__":
    if checkpoint.ok:
     loader = data.Data(args)
     print("data_loaded")
     model = model.Model(args, checkpoint)
     print(model)
     print("model_loaded")
     loss = loss.Loss(args, checkpoint) if not args.test_only else None
     print(loss)
     t = Trainer(args, loader, model, loss, checkpoint)
     #print(t)
     print('phas')
     while not t.terminate():
     
      #if __name__ == "__main__":
        print('loop')
        
        t.train()
        
        t.test()

     checkpoint.done()
