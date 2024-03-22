import zmq
from spot_rl.utils.pose_correction import detect, load_model, segment

device = "cpu"
owlvitmodel, processor = load_model("owlvit", device)

port = "21001"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")
print(f"Detection Server Listening on port {port}")

while True:
    img, object_name, thresh, device = socket.recv_pyobj()
    print("Recieved img for Detection")
    h, w = img.shape[:2]
    predictions = detect(img, object_name, thresh, device, owlvitmodel, processor)
    # masks = segment(img, bbox, [h, w], device, sammodel)
    # mask = masks[0, 0].cpu().numpy()
    socket.send_pyobj(predictions)
