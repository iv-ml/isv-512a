import torch 
import torch.nn.functional as F
from lipsync.face_emb.iresnet import iresnet50

class FaceEmb(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.net = iresnet50(fp16=False)
        self.net.load_state_dict(torch.load(weight))
        self.net.eval()
    
    def forward(self, img):
        #resize the torch tensor to 112x112
        img = F.interpolate(img, (112, 112))
        return self.net(img)


if __name__ == "__main__":
    import torchvision
    import fastcore.all as fc 
    face_emb = FaceEmb("weights/backbone.pth")
    person1 = fc.L(fc.Path("/data/prakash_lipsync/video/hdtf/a18V-MRrnlk--0002/").glob("**/*.png"))
    person2 = fc.L(fc.Path("/data/prakash_lipsync/video/hdtf/NqTJJLSxijA--0002/").glob("**/*.png"))
    
    # person1 
    img1_loc = person1[0]
    img1_loc_2 = person1[1]
    img1 = torchvision.io.read_image(img1_loc).unsqueeze(0).to(torch.float32)/255.0
    img1_2 = torchvision.io.read_image(img1_loc_2).unsqueeze(0).to(torch.float32)/255.0

    x1 = face_emb(img1)
    x1_2 = face_emb(img1_2)
    cs_1 = torch.cosine_similarity(x1, x1_2, dim=1)
    print("similar", cs_1)

    # person2 
    img2_loc = person2[0]
    img2_loc_2 = person2[1]
    img2 = torchvision.io.read_image(img2_loc).unsqueeze(0).to(torch.float32)/255.0
    img2_2 = torchvision.io.read_image(img2_loc_2).unsqueeze(0).to(torch.float32)/255.0
    x2 = face_emb(img2)
    x2_2 = face_emb(img2_2)
    cs_2 = torch.cosine_similarity(x2, x2_2, dim=1)
    print("similar", cs_2)

    
    ## disimilar 
    cs = torch.cosine_similarity(x1, x2, dim=1)
    cs_2 = torch.cosine_similarity(x1_2, x2_2, dim=1)
    print(cs)
    print(cs_2)


    #store the image in PIL formmar 
    img1_pil = torchvision.transforms.ToPILImage()(img1.squeeze(0))
    img2_pil = torchvision.transforms.ToPILImage()(img2.squeeze(0))
    img1_pil.save("temp/p1.png")
    img2_pil.save("temp/p2.png")
