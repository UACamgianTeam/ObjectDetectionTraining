class ARGS:
    def __init__(self,
                 config_file: str="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
                 confidence_threshold: float=0.5,
                 opts: list=[]):
        self.config_file = config_file
        self.opts = opts
        self.confidence_threshold = confidence_threshold

ct = 0.5
cf = "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
op = ["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"] # "MODEL.DEVICE", "cpu", 

args = ARGS(config_file=cf, confidence_threshold=ct, opts=op)
