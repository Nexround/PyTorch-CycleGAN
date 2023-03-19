from facenet_pytorch import MTCNN

mtcnn = MTCNN().cuda()
pnet = mtcnn.pnet

def face_result(image_tensor):
    pnet_output = pnet(image_tensor)

    return pnet_output