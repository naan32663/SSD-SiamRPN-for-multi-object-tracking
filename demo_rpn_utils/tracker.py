
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:00:05 2020

@author: Anna
"""

import numpy as np
from demo_rpn_utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from scipy.optimize import linear_sum_assignment


class Track(object):
    """Track class for every object to be tracked
    """

    def __init__(self, target_pos, target_sz, trackIdCount, trackNet, image):
        """Initialize variables used by Track class
        """
        self.track_id = trackIdCount                                 # identification of each track object
        self.track_net = trackNet                                    # SimaRPN instance to track this object
        self.target_pos = target_pos                                 # position of track object (xmin,ymin)
        self.target_sz = target_sz                                   # size of track object (w,h)
        self.target_img = image                                      # image of this object
        self.skipped_frames = 0                                      # number of frames skipped undetected


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    """

    def __init__(self, similarity_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.similarity_thresh = similarity_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, targets, tracknet, image, modelname):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            targets: detected objects to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(targets)):
                track = Track(targets[i]['target_pos'], targets[i]['target_sz'], self.trackIdCount, tracknet, image)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks) # template objects
        M = len(targets)  # detection objects
        print("len of tracks = %d , len of detection = %d " %(N, M))
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(targets)):
                state = SiamRPN_init(targets[j]['target_img'], targets[j]['target_pos'], targets[j]['target_sz'], tracknet, modelname)
                state = SiamRPN_track(state, self.tracks[i].target_img)  # track

                score = state['score']
                cost[i][j] = score
#               print("score = " + str(score))

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)

        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(targets)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(targets[un_assigned_detects[i]]['target_pos'], targets[un_assigned_detects[i]]['target_sz'], 
                              self.trackIdCount, tracknet, targets[un_assigned_detects[i]]['target_img'])
                self.trackIdCount += 1
                self.tracks.append(track)

