import zmq
from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8x-worldv2.pt").cuda()

port = "21001"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")
print(f"Yolo World Server Listening on port {port}")

while True:
    """A service for running segmentation service, send request using zmq socket"""
    img, query_classes, visualize = socket.recv_pyobj()
    print(f"Recieved img for Detection; query classes {query_classes}")
    model.set_classes(query_classes)
    if visualize:
        visualization_img = img.copy()
    results = model.predict(img, device="cuda:1", stream=False, conf=0.1)
    bboxes, probs, labels = [], [], []
    if len(results) == 1: #since we only send 1 image
        r = results[0]
        bboxes = r.boxes.xyxy.cpu().numpy().tolist()
        probs = r.boxes.conf.cpu().numpy().astype(float).tolist()
        clss = r.boxes.cls.cpu().numpy().astype(int).tolist()
        labels = [model.names[clsi] for clsi in clss]
        # Plot results image
        if visualize:
            visualization_img = r.plot(img=visualization_img)
            
    if not visualize:
        visualization_img = None
    socket.send_pyobj((bboxes, probs, labels, visualization_img))
