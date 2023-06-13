import numpy as np
import matching
from basetrack import BaseTrack, TrackState
from collections import deque
import time

class STrack(BaseTrack):
    def __init__(self, tlwh, score, clase, image, hash):

        # wait activate
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False
        self.clase = clase
        self.image = image
        self.hash = hash
        self.hash_memory = deque(maxlen=10)
        self.last_hash_stored = time.time()
        self.store_period = 1.5
        self.score = score
        self.tracklet_len = 0

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_class_id(str(self.clase))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.image = new_track.image

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.tlwh = new_track.tlwh
        self.state = TrackState.Tracked
        self.is_activated = True
        self.image = new_track.image
        self.hash = new_track.hash
        if time.time() - self.last_hash_stored > self.store_period:
            self.hash_memory.append(new_track.hash)
            self.last_hash_stored = time.time()
        self.score = new_track.score

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class HashTracker(object):
    def __init__(self, frame_rate=30, buffer_=15):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.match_thresh = 0.99999999
        self.frame_id = 0
        # self.args = args
        #self.det_thresh = args.track_thresh
        self.track_thresh = 0.2
        self.det_thresh = self.track_thresh + 0.1
        self.track_buffer = buffer_
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size

        # For specific element mode
        self.tracked_element = None
        self.chosen_track = -1

    def update(self, scores, bboxes, clases, images, hash):
        if not self.tracked_stracks and not self.lost_stracks:
            self.chosen_track = -1

        # Cleaning not followed tracks
        if self.chosen_track != -1:
            self.tracked_stracks = [track for track in self.tracked_stracks if track.track_id == self.chosen_track]
            self.lost_stracks = [track for track in self.lost_stracks if track.track_id == self.chosen_track]
            return self.update_element_following(scores, bboxes, clases, images, hash)
        else:
            # print("tracked_stracks", self.tracked_stracks)
            # print("lost_stracks", self.lost_stracks)
            return self.update_original(scores, bboxes, clases, images, hash)

    def update_element_following(self, scores, bboxes, clases, images, hash):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > self.track_thresh

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        clases_keep = clases[remain_inds]
        images_keep = images[remain_inds]
        hash_keep = hash[remain_inds]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clases, images, hash) for
                          (tlbr, s, clases, images, hash) in zip(dets, scores_keep, clases_keep, images_keep, hash_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        if len(detections) > 0 and len(strack_pool) > 0:
            # # Predict the current location with KF
            dists = matching.hash_distance_following(strack_pool, detections)
            pos_match = matching.get_max_similarity_detection(dists)
            # dists = matching.fuse_score(dists, detections)
            # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
            track = strack_pool[0]
            det = detections[pos_match]
            if track.state == TrackState.Tracked:
                track.update(detections[pos_match], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        else:
            for it in strack_pool:
                if not it.state == TrackState.Lost:
                    it.mark_lost()
                    lost_stracks.append(it)

        # ''' Step 3: Second association, with low score detection boxes'''
        # # association the untrack to the low score detections
        # if len(dets_second) > 0:
        #     '''Detections'''
        #     detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clase, roi, hash) for
        #                   (tlbr, s, clase, roi, hash) in zip(dets_second, scores_second, clases_second, images_second, hash_second)]
        # else:
        #     detections_second = []
        # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # dists = matching.hash_distance(r_tracked_stracks, detections_second)
        # matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # for itracked, idet in matches:
        #     track = r_tracked_stracks[itracked]
        #     det = detections_second[idet]
        #     if track.state == TrackState.Tracked:
        #         track.update(det, self.frame_id)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_stracks.append(track)
        #
        # for it in u_track:
        #     track = r_tracked_stracks[it]
        #     if not track.state == TrackState.Lost:
        #         track.mark_lost()
        #         lost_stracks.append(track)

        # '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # detections = [detections[i] for i in u_detection]
        # dists = matching.hash_distance(unconfirmed, detections)
        # dists = matching.fuse_score(dists, detections)
        # matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # for itracked, idet in matches:
        #     unconfirmed[itracked].update(detections[idet], self.frame_id)
        #     activated_starcks.append(unconfirmed[itracked])
        # for it in u_unconfirmed:
        #     track = unconfirmed[it]
        #     track.mark_removed()
        #     removed_stracks.append(track)

        # """ Step 4: Init new stracks"""
        # for inew in u_detection:
        #     track = detections[inew]
        #     if track.score < self.det_thresh:
        #         continue
        #     track.activate(self.frame_id)
        #     activated_starcks.append(track)
        # """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks.extend(self.lost_stracks)
        return output_stracks

    def update_original(self, scores, bboxes, clases, images, hash):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        clases_keep = clases[remain_inds]
        clases_second = clases[inds_second]
        images_keep = images[remain_inds]
        images_second = images[inds_second]
        hash_keep = hash[remain_inds]
        hash_second = hash[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clases, image, hash) for
                          (tlbr, s, clases, image, hash) in zip(dets, scores_keep, clases_keep, images_keep, hash_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        dists = matching.hash_distance(strack_pool, detections)
        #print("DISTS", dists)
        dists = matching.fuse_score(dists, detections)
        #print("DISTS AFTER FUSING", dists)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clase, image, hash) for
                          (tlbr, s, clase, image, hash) in zip(dets_second, scores_second, clases_second, images_second, hash_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.hash_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.hash_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks.extend(self.lost_stracks)
        return output_stracks

    def joint_stracks(self, tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res


    def sub_stracks(self, tlista, tlistb):
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())


    def remove_duplicate_stracks(self, stracksa, stracksb):
        pdist = matching.hash_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if not i in dupa]
        resb = [t for i, t in enumerate(stracksb) if not i in dupb]
        return resa, resb
