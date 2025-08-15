"""
Minimal SORT tracker (Bewley 2016) – single-file version to avoid pip issues.
Requires: numpy, scipy, filterpy
"""

import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # x, y, s, r (centre x/y, scale, ratio)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0,4] = self.kf.F[1,5] = self.kf.F[2,6] = 1
        self.kf.H[:4,:4] = np.eye(4)
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.update(bbox)

    def update(self, bbox):
        x,y,x2,y2 = bbox
        w  = x2 - x
        h  = y2 - y
        x_c = x + w/2
        y_c = y + h/2
        s   = w * h
        r   = w / (h+1e-6)
        z = np.array([x_c, y_c, s, r]).reshape((4,1))
        self.kf.update(z)
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        x_c,y_c,s,r = self.kf.x[:4].reshape((4,))
        w = np.sqrt(s*r)
        h = s / (w+1e-6)
        return np.array([x_c-w/2, y_c-h/2, x_c+w/2, y_c+h/2])

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_th = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets_np):
        """
        Params:
          dets_np – ndarray N×5 (x1,y1,x2,y2,conf)
        Returns:
          ndarray M×5 (x1,y1,x2,y2,track_id)
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [*self.trackers[t].get_state(), 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets = dets_np[:,:4] if len(dets_np) else np.empty((0,4))
        iou_mat = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_mat[d,t] = iou(det, trk)

                # --- matching ----------------------------------------------------
                # --- matching ----------------------------------------------------
        matched_indices = np.empty((0, 2), dtype=int)   # safe default
        if iou_mat.size:
            a, b = (iou_mat > self.iou_th).nonzero()
            if len(a):
                matched_indices = np.stack((a, b), axis=1)

        unmatched_dets = list(set(range(len(dets))) - set(matched_indices[:,0])) if len(dets) else []
        unmatched_trks = list(set(range(len(trks))) - set(matched_indices[:,1]))

        # update matched trackers with assigned detections
        for d,t in matched_indices:
            self.trackers[t].update(dets_np[d,:4])

        # create new trackers for unmatched detections
        for idx in unmatched_dets:
            trk = KalmanBoxTracker(dets_np[idx,:4])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))  # id +1
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return np.concatenate(ret) if len(ret) else np.empty((0,5))
