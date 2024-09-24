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
    results = model.predict(img, device="cuda", stream=False)
    bboxes, probs = [], []
    if len(results) > 0:
        for i, r in enumerate(results):
            # Plot results image
            if visualize:
                visualization_img = r.plot(img=visualization_img)
            if r.boxes.xyxy.shape[0] > 0:
                conf = r.boxes.conf.cpu().numpy().tolist()[0]
                if True:
                    try:
                        bboxes.append(r.boxes.xyxy.cpu().numpy().tolist()[0])
                        probs.append(conf)
                    except Exception as e:
                        print(e)
                        breakpoint()
    if not visualize:
        visualization_img = None
    socket.send_pyobj((bboxes, probs, visualization_img))
