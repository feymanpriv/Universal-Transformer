import torch

model = torch.jit.load("saved_model_swin_large_win7_224_linprobe_5.pt")

model.eval()
#print (model)
embedding_fn = model

#input_batch1 = torch.randn(1, 3, 224, 224)
input_batch3 = torch.ones(1, 3, 224, 224)

with torch.no_grad():
    #embedding1 = torch.flatten(embedding_fn(input_batch1)[0]).cpu().data.numpy()
    #embedding2 = torch.flatten(embedding_fn(input_batch2)[0]).cpu().data.numpy()
    embedding3 = torch.flatten(embedding_fn(input_batch3)[0]).cpu().data.numpy()

print(embedding3)
#print(embedding1)