"""Simple causal track-motion heuristic; uses YOLO tracks only, never GT."""
import numpy as np

def track_records(frame):
    """Return one YOLO detection per currently assigned tracker ID."""
    return {int(tid):(frame['boxes'][i],frame['xy'][i],frame['conf'][i])
            for i,tid in enumerate(frame['ids']) if int(tid)>=0}

def pair_motion(previous,current,confidence=.3,min_joints=4,clip=.5):
    """Dimensionless per-frame motion: 75% articulation + 25% box movement.

    Articulation is the mean clipped displacement of shared reliable body joints
    after subtracting each frame's box center. Normalize by the average box
    diagonal. This is a heuristic, not a physical velocity or action label.
    """
    b0,p0,c0=previous;b1,p1,c1=current
    b0=np.asarray(b0,dtype=float);b1=np.asarray(b1,dtype=float)
    p0=np.asarray(p0,dtype=float)[5:17];p1=np.asarray(p1,dtype=float)[5:17]
    c0=np.asarray(c0,dtype=float)[5:17];c1=np.asarray(c1,dtype=float)[5:17]
    if not np.isfinite(b0).all() or not np.isfinite(b1).all():return None
    w0=b0[2:]-b0[:2];w1=b1[2:]-b1[:2]
    if np.any(w0<=0) or np.any(w1<=0):return None
    scale=(np.linalg.norm(w0)+np.linalg.norm(w1))/2
    if scale<1e-6:return None
    valid=np.isfinite(p0).all(1)&np.isfinite(p1).all(1)&(p0>0).all(1)&(p1>0).all(1)
    valid&=np.isfinite(c0)&np.isfinite(c1)&(c0>=confidence)&(c1>=confidence)
    if int(valid.sum())<min_joints:return None
    center0=(b0[:2]+b0[2:])/2;center1=(b1[:2]+b1[2:])/2
    relative=(p1[valid]-center1)-(p0[valid]-center0)
    articulation=float(np.mean(np.minimum(np.linalg.norm(relative,axis=1)/scale,clip)))
    translation=min(float(np.linalg.norm(center1-center0)/scale),clip)
    return {'score':.75*articulation+.25*translation,'articulation':articulation,
            'translation':translation,'joints':int(valid.sum())}

def movement_scores(cache,index,window=20,confidence=.3,min_pairs=5):
    """Causal per-ID mean motion over up to `window` adjacent frame pairs.

    Only detections with an ID in the *current* frame are candidates. Never
    bridge a missing frame. No GT, action labels, or future frames are read.
    """
    if window<1 or min_pairs<1 or min_pairs>window:raise ValueError('Invalid motion window')
    if not 0<=index<len(cache):raise IndexError('frame not cached')
    current=track_records(cache[index]);history={tid:[] for tid in current}
    for t in range(max(1,index-window+1),index+1):
        left=track_records(cache[t-1]);right=track_records(cache[t])
        for tid in history.keys() & left.keys() & right.keys():
            item=pair_motion(left[tid],right[tid],confidence=confidence)
            if item is not None:history[tid].append(item)
    return {tid:{'pairs':len(values),'score':float(np.mean([v['score'] for v in values])) if len(values)>=min_pairs else None,
                 'articulation':float(np.mean([v['articulation'] for v in values])) if values else None,
                 'translation':float(np.mean([v['translation'] for v in values])) if values else None}
            for tid,values in history.items()}

def choose_most_moving(scores):
    """Return current ID with highest eligible score; stable ID tie-break."""
    valid=[(tid,item['score']) for tid,item in scores.items() if item['score'] is not None and np.isfinite(item['score'])]
    return min(valid,key=lambda pair:(-pair[1],pair[0]))[0] if valid else None
