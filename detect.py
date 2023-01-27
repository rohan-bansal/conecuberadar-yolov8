from ultralytics.yolo.v8.detect import DetectionPredictor
import cv2
from ultralytics.yolo.utils.plotting import colors

class ConeCubeDetector(DetectionPredictor):

    def convex_hull_pointing_up(self, ch):
    
        points_above_center, points_below_center = [], []
        
        x, y, w, h = cv2.boundingRect(ch)
        aspect_ratio = w / h

        if aspect_ratio < 0.8:
            vertical_center = y + h / 2

            for point in ch:
                if point[0][1] < vertical_center:
                    points_above_center.append(point)
                elif point[0][1] >= vertical_center:
                    points_below_center.append(point)

            left_x = points_below_center[0][0][0]
            right_x = points_below_center[0][0][0]
            for point in points_below_center:
                if point[0][0] < left_x:
                    left_x = point[0][0]
                if point[0][0] > right_x:
                    right_x = point[0][0]

            for point in points_above_center:
                if (point[0][0] < left_x) or (point[0][0] > right_x):
                    return False
        else:
            return False
            
        return True

    def postprocess(self, preds, img, orig_img, classes=None):
        results = super().postprocess(preds, img, orig_img, classes)
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam or self.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "


        # write - THE IMPORTANT PART
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()

            c = int(cls)
            label = f'{self.model.names[c]} {conf:.2f}'
            self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

            if "cone" in label:
                xyxy = d.xyxy.squeeze()

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                crop_img = im0[c1[1]-10:c2[1]+10, c1[0]:c2[0]]

                hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                cv2.imshow("threshed", thresh)

                contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                approx_contours = []
                for c in contours:
                    approx = cv2.approxPolyDP(c, 10, closed = True)
                    approx_contours.append(approx)

                all_convex_hulls = []
                for ac in approx_contours:
                    all_convex_hulls.append(cv2.convexHull(ac))

                convex_hulls_4to10 = []
                for ch in all_convex_hulls:
                    if 4 <= len(ch) <= 10:
                        convex_hulls_4to10.append(cv2.convexHull(ch))

                cv2.drawContours(crop_img, convex_hulls_4to10, -1, (0,255,0), 2)

                cones = []
                for ch in convex_hulls_4to10:
                    if self.convex_hull_pointing_up(ch):
                        cones.append(ch)
                        upright_cones += 1
                    else:
                        tipped_cones += 1

                if len(cones) == 0:
                    cv2.drawContours(crop_img, convex_hulls_4to10, -1, (0,0,255), 2)
                else:
                    cv2.drawContours(crop_img, convex_hulls_4to10, -1, (0,255,0), 2)

            else:
                pass

        return log_string


predictor = ConeCubeDetector(overrides={})
results = predictor(source="test_vid.mp4", model="runs/detect/MODEL/weights/best.pt", stream=True)

for result in results:
    boxes = result.boxes
    print(boxes)
