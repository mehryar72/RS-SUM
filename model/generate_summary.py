import numpy as np
from knapsack_implementation import knapSack
import ruptures as rpt

def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions, new_shots=2, pen=1, all_scores2=[]):
    all_summaries = []
    for video_index in range(len(all_scores)):
        # Get shots' boundaries
        shot_bound = all_shot_bound[
            video_index]  # [number_of_shots, 2] - the boundaries refer to the initial number of frames (before the subsampling)
        frame_init_scores = all_scores[video_index]
        if len(all_scores2):
            score2 = all_scores2[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]
        if new_shots:
            if len(all_scores2):
                temp = np.transpose(np.stack([frame_init_scores, score2]))
            else:
                temp = frame_init_scores
            if new_shots == 4:
                temp = (temp - np.min(temp,0)) / (np.max(temp,0) - np.min(temp,0))
            algo = rpt.Pelt(model="rbf").fit(temp)
            result = algo.predict(pen=pen)
            new_bounds = []
        # Compute the importance scores for the initial frame sequence (not the subsampled one)
        frame_scores = np.zeros((n_frames), dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if new_shots:
                if i in result:
                    new_bounds.append(pos_left - 1)

            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        if new_shots:
            if (new_shots % 2) == 0:
                new_bounds.extend(shot_bound[:, 1])
            new_bounds = np.sort(np.asarray(new_bounds))
            new_bounds = np.delete(new_bounds, np.argwhere(np.ediff1d(new_bounds) <= 15))

            shot_bound = [[0, new_bounds[0]]]
            for new_in in range(len(new_bounds) - 1):
                shot_bound.append([new_bounds[new_in] + 1, new_bounds[new_in + 1]])
            # shot_bound.append([new_bounds[-1]+1,pos_right])
        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

        # Select the best shots using the knapsack implementation
        final_max_length = int((shot[1] + 1) * 0.15)

        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries

