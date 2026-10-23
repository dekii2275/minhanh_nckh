"""
Module: select_sequences.py
Purpose: Deterministically select 3 representative video sequences from Penn Action 
         with distinct actions (e.g. jumping_jacks, squats, tennis_forehand/baseball_pitch),
         sample 50 uniformly spaced frames per sequence (total 150 frames),
         and output data/processed/selected_frames.csv before running YOLO.
"""

import os
import glob
import pandas as pd
import numpy as np
import scipy.io as sio

def select_pilot_dataset(
    penn_dir="data/raw/Penn_Action",
    target_actions=["jumping_jacks", "squat", "tennis_forehand"],
    frames_per_sequence=50,
    output_csv="data/processed/selected_frames.csv"
):
    """
    Deterministically select sequences based on fixed a priori criteria:
    1. Action belongs to target list of diverse dynamic movements.
    2. Sequence has sufficient frames (>= frames_per_sequence).
    3. High GT visibility (> 85% valid visible keypoints).
    4. Deterministic selection: first valid sequence encountered in sorted order.
    5. Uniform temporal sampling across the video duration.
    """
    labels_dir = os.path.join(penn_dir, "labels")
    frames_dir = os.path.join(penn_dir, "frames")
    
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
    mat_files = sorted(glob.glob(os.path.join(labels_dir, "*.mat")))
    print(f"Scanning {len(mat_files)} annotations to select target sequences...")
    
    # Collect candidates by action
    candidates_by_action = {act: [] for act in target_actions}
    
    for mat_path in mat_files:
        try:
            data = sio.loadmat(mat_path)
            action = str(data['action'][0]) if 'action' in data else ''
            if action in candidates_by_action:
                nframes = int(data['nframes'][0, 0])
                vis = np.array(data['visibility'], dtype=int)
                if vis.shape[1] != 13 and vis.shape[0] == 13:
                    vis = vis.T
                vis_rate = float(vis.sum()) / float(vis.size) if vis.size > 0 else 0
                
                seq_id = os.path.splitext(os.path.basename(mat_path))[0]
                seq_frame_dir = os.path.join(frames_dir, seq_id)
                
                if os.path.exists(seq_frame_dir) and nframes >= frames_per_sequence and vis_rate >= 0.85:
                    candidates_by_action[action].append({
                        'seq_id': seq_id,
                        'mat_path': mat_path,
                        'seq_frame_dir': seq_frame_dir,
                        'nframes': nframes,
                        'vis_rate': vis_rate
                    })
        except Exception as e:
            continue
            
    print("Candidates found per target action:")
    for act, cands in candidates_by_action.items():
        print(f"  - {act}: {len(cands)} eligible sequences")
        
    selected_seqs = {}
    for act in target_actions:
        if len(candidates_by_action[act]) == 0:
            # Fallback: take candidate with largest nframes
            all_cands = []
            for mat_path in mat_files:
                data = sio.loadmat(mat_path)
                if str(data['action'][0]) == act:
                    nframes = int(data['nframes'][0, 0])
                    seq_id = os.path.splitext(os.path.basename(mat_path))[0]
                    all_cands.append({'seq_id': seq_id, 'mat_path': mat_path, 'nframes': nframes, 'seq_frame_dir': os.path.join(frames_dir, seq_id)})
            all_cands.sort(key=lambda x: x['nframes'], reverse=True)
            if all_cands:
                selected_seqs[act] = all_cands[0]
            else:
                raise ValueError(f"No sequence found for action: {act}")
        else:
            # Select the first sequence satisfying the criteria
            selected_seqs[act] = candidates_by_action[act][0]
            
    print("\nSelected sequences:")
    for act, info in selected_seqs.items():
        print(f"  - Action: {act} | Seq: {info['seq_id']} | Total Frames: {info['nframes']}")
        
    # Generate selected_frames records with uniform temporal sampling
    records = []
    for act, info in selected_seqs.items():
        seq_id = info['seq_id']
        total_f = info['nframes']
        mat_path = info['mat_path']
        seq_dir = info['seq_frame_dir']
        
        # Uniform sampling of distinct frames
        num_samples = min(frames_per_sequence, total_f)
        sample_indices = np.unique(np.linspace(0, total_f - 1, num_samples, dtype=int))
        for s_idx in sample_indices:
            frame_num = int(s_idx) + 1 # 1-based numbering for file names (000001.jpg)
            img_path = os.path.join(seq_dir, f"{frame_num:06d}.jpg")
            records.append({
                'sequence_id': seq_id,
                'action': act,
                'frame_index': frame_num,
                'sample_order': len(records),
                'image_path': img_path,
                'annotation_path': mat_path
            })
            
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} selected frames to {output_csv}")
    print(f"Summary by sequence:\n{df.groupby(['sequence_id', 'action']).size()}")
    return df

if __name__ == "__main__":
    select_pilot_dataset()
