"""Verify pilot sample, annotation and source frames against the YOLO cache."""
import csv
import hashlib
import json
import numpy as np
import compare_person_selection as base
import gt_crop_pose_experiment as crop


def main():
    out=base.ROOT/'outputs/mediapipe_yolo_pilot_20261103'
    info=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    with base.MANIFEST.open(encoding='utf-8-sig',newline='') as f:
        manifest={r['id']:r for r in csv.DictReader(f)}
    frames=0
    for sid in info['ids']:
        row=manifest[sid];assert row['seed0_split']=='validation'
        assert base.sha(base.LABELS/f'{sid}.mat')==row['label_sha256']
        a=np.load(crop.OUT/'poses'/f'{sid}.npz');b=np.load(out/'poses'/f'{sid}.npz')
        assert str(a['annotation_sha256'])==str(b['annotation_sha256'])==row['label_sha256']
        assert b['xy'].shape==(int(row['nframes']),12,2)
        h=hashlib.sha256()
        for i in range(1,int(row['nframes'])+1):
            raw=(base.FRAMES/sid/f'{i:06d}.jpg').read_bytes()
            h.update(len(raw).to_bytes(8,'big'));h.update(raw);frames+=1
        assert h.hexdigest()==str(a['image_bytes_sha256'])
    assert frames==summary['frames']
    print(f'PASS: {len(info["ids"])} validation IDs, labels, {frames} image bytes and pose shapes verified.')


if __name__=='__main__':main()
