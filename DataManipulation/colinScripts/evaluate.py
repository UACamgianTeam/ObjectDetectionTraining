import json
import os

# Be sure to add a '/' after the folders
predsFolder = "predictions/"
truthFolder = "inputs/pevid/"

CONF_THRESH = 0.6

if __name__ == '__main__':

    for filename in os.listdir():

        fn = filename.split('.')[0]
        truth = json.reads(truthFolder + fn.lower() + '/annotations/' + fn.lower() + '.json')
        preds = json.reads(predsFolder + fn + '.json')

        results = []

        for p in preds['annotations']:
            if p['score'] >= CONF_THRESH:
                pc = (
                        p['bbox'][0],
                        p['bbox'][1],
                        p['bbox'][0] + p['bbox'][2],
                        p['bbox'][1] + p['bbox'][3]
                )
                iou = 0.0
                for t in truth['annotations']:
                    if int(p['image_id']) == int(t['image_id']):
                        tc = (
                                t['bbox'][0],
                                t['bbox'][1],
                                t['bbox'][0] + t['bbox'][2],
                                t['bbox'][1] + t['bbox'][3]
                        )
                        if ((p[0] <= t[2] and r[0] <= p[2])) and ((p[1] <= r[3] and r[1] <= p[3])):
                            outer = max(p[2] - t[0], t[2] - p[0]) * \
                                    max(p[3] - t[1], t[3] - p[1])
                            inner = min(p[2] - t[0], t[2] - p[0]) * \
                                    min(p[3] - t[1], t[3] - p[1])
                            iou_temp = inner / outer
                            if iou_temp > iou:
                                iou = iou_temp
                results.append((p['id'], iou, (iou > 0.0))

        with open('results/' + fn + '.csv', 'w') as f:
            for r in results:
                f.write(f"{r[0]},{r[1]},{r[2]}\n")