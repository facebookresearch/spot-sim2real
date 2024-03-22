import zmq
from spot_rl.utils.pose_correction import detect, load_model, segment

device = "cpu"
sammodel = load_model("sam", device)

port = "21001"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")
print(f"Segment Server Listening on port {port}")

while True:
    (img, bbox) = socket.recv_pyobj()
    print("Recieved img & bbox for segmentation")
    h, w = img.shape[:2]
    masks = segment(img, bbox, [h, w], device, sammodel)
    mask = masks[0, 0].cpu().numpy()
    socket.send_pyobj(mask)
