import math


class PinchSession:
    def __init__(self, smooth=0.40, pinch_hold_frames=4, base_threshold=0.05, merge_hold_frames=5, max_move_per_frame=5, drop_grace_frames=5):
        self.locks = {0: None, 1: None}
        self.offsets = {0: None, 1: None}
        self.session_active = False
        self.session_ids = set()
        self.merge_triggered = False
        self.smooth = smooth
        self.pinch_hold_frames = pinch_hold_frames
        self.base_threshold = base_threshold
        self.pinch_counters = {0: 0, 1: 0}
        self.pinch_active = {0: False, 1: False}
        self.merge_hold_frames = merge_hold_frames
        self.merge_counter = 0
        self.max_move_per_frame = max_move_per_frame
        self.drop_grace_frames = drop_grace_frames
        self.drop_counter = 0

    def reset(self):
        self.locks = {0: None, 1: None}
        self.offsets = {0: None, 1: None}
        self.session_active = False
        self.session_ids = set()
        self.merge_triggered = False
        self.pinch_counters = {0: 0, 1: 0}
        self.pinch_active = {0: False, 1: False}
        self.merge_counter = 0
        self.drop_counter = 0

    def _pinch_threshold(self, lm):
        wrist = lm[0]
        mid = lm[9]
        hand_size = math.hypot(wrist[0] - mid[0], wrist[1] - mid[1])
        return max(self.base_threshold, hand_size * 0.35)

    def update(self, venv, hand_infos, raw_h, w, h, is_active, was_active=False):
        # If signal not active or tracking drops temporarily, apply grace period before snapping back
        if not is_active or len(hand_infos) != 2:
            if self.session_active:
                self.drop_counter += 1
                if self.drop_counter >= self.drop_grace_frames:
                    for a in venv.atoms:
                        if a.id in self.session_ids:
                            a.snap_back()
                    self.reset()
            else:
                self.reset()
            return False

        self.drop_counter = 0 # Reset grace counter if tracking is good
        pinch_points = {}

        for hand in hand_infos:
            idx = hand['index']
            lm = hand['landmarks']
            t_nx, t_ny = lm[4][0], lm[4][1]
            i_nx, i_ny = lm[8][0], lm[8][1]

            pinch_dist = math.hypot(t_nx - i_nx, t_ny - i_ny)
            threshold = self._pinch_threshold(lm)

            if self.pinch_active[idx]:
                if pinch_dist <= threshold:
                    pinch_points[idx] = ((t_nx + i_nx) / 2, (t_ny + i_ny) / 2)
                else:
                    self.pinch_active[idx] = False
                    self.pinch_counters[idx] = 0
            else:
                if pinch_dist <= threshold:
                    self.pinch_counters[idx] += 1
                    if self.pinch_counters[idx] >= self.pinch_hold_frames:
                        self.pinch_active[idx] = True
                        pinch_points[idx] = ((t_nx + i_nx) / 2, (t_ny + i_ny) / 2)
                else:
                    self.pinch_counters[idx] = 0

        # Unlock hands that stopped pinching
        for idx in list(self.locks.keys()):
            if not self.pinch_active.get(idx, False):
                if self.locks[idx] is not None:
                    self.locks[idx].is_grabbed = False
                    self.locks[idx].grabbed_by_hand = None
                self.locks[idx] = None
                self.offsets[idx] = None

        # Lock atoms
        for hand in hand_infos:
            idx = hand['index']
            if not self.pinch_active.get(idx, False) or self.locks.get(idx) is not None:
                continue

            pinch_point = pinch_points.get(idx)
            if pinch_point is None:
                continue

            candidates = []
            for atom in venv.atoms:
                if atom.is_grabbed and atom.grabbed_by_hand != idx:
                    continue
                wx, wy = venv.project_normalized_to_3d(pinch_point[0], pinch_point[1], atom.z)
                dist = math.hypot(wx - atom.x, wy - atom.y)

                if dist <= atom.base_radius:
                    candidates.append((dist, atom, wx, wy))

            if candidates:
                candidates.sort(key=lambda t: t[0])
                chosen = candidates[0][1]
                grab_wx = candidates[0][2]
                grab_wy = candidates[0][3]

                chosen.is_grabbed = True
                chosen.grabbed_by_hand = idx
                self.locks[idx] = chosen
                self.offsets[idx] = (chosen.x - grab_wx, chosen.y - grab_wy)

        full_pinch = (
            self.locks[0] is not None
            and self.locks[1] is not None
            and self.locks[0].id != self.locks[1].id
        )

        if full_pinch and not self.session_active:
            self.session_active = True
            self.session_ids = {self.locks[0].id, self.locks[1].id}
            self.merge_triggered = False

        if not full_pinch and self.session_active:
            self.drop_counter += 1
            if self.drop_counter >= self.drop_grace_frames:
                for a in venv.atoms:
                    if a.id in self.session_ids:
                        a.snap_back()
                self.session_active = False
                self.session_ids = set()
                self.locks = {0: None, 1: None}
                self.offsets = {0: None, 1: None}
                self.merge_triggered = False

        # Move the locked atoms
        if full_pinch:
            p0 = pinch_points.get(0)
            p1 = pinch_points.get(1)
            atom0 = self.locks[0]
            atom1 = self.locks[1]
            off0 = self.offsets.get(0, (0, 0))
            off1 = self.offsets.get(1, (0, 0))

            if p0 and p1 and atom0 and atom1:
                # Project both pinch points
                wx0, wy0 = venv.project_normalized_to_3d(p0[0], p0[1], atom0.z)
                wx1, wy1 = venv.project_normalized_to_3d(p1[0], p1[1], atom1.z)

                # Check if the tracker accidentally swapped the hand IDs
                dist_straight = math.hypot((wx0 + off0[0]) - atom0.x, (wy0 + off0[1]) - atom0.y) + \
                                math.hypot((wx1 + off1[0]) - atom1.x, (wy1 + off1[1]) - atom1.y)

                dist_crossed = math.hypot((wx1 + off0[0]) - atom0.x, (wy1 + off0[1]) - atom0.y) + \
                               math.hypot((wx0 + off1[0]) - atom1.x, (wy0 + off1[1]) - atom1.y)

                # If crossing them results in a shorter path, Mediapipe swapped the hands! Swap them back.
                if dist_crossed < dist_straight:
                    p0, p1 = p1, p0
                    wx0, wy0 = venv.project_normalized_to_3d(p0[0], p0[1], atom0.z)
                    wx1, wy1 = venv.project_normalized_to_3d(p1[0], p1[1], atom1.z)

                # Apply the movement
                for atom, wx, wy, off in [(atom0, wx0, wy0, off0), (atom1, wx1, wy1, off1)]:
                    target_x = wx + off[0]
                    target_y = wy + off[1]

                    dx = target_x - atom.x
                    dy = target_y - atom.y
                    dist = math.hypot(dx, dy)

                    if dist > self.max_move_per_frame:
                        scale = self.max_move_per_frame / dist
                        dx *= scale
                        dy *= scale

                    atom.x += dx * self.smooth
                    atom.y += dy * self.smooth

        # Merging logic
        if (
            full_pinch
            and not self.merge_triggered
            and self.locks[0] and self.locks[1]
            and self.locks[0].id != self.locks[1].id
        ):
            a1, a2 = self.locks[0], self.locks[1]
            dist = math.hypot(a1.x - a2.x, a1.y - a2.y)
            merge_distance = (a1.base_radius + a2.base_radius) * 0.75

            if dist < merge_distance:
                self.merge_counter += 1
            else:
                self.merge_counter = 0

            if self.merge_counter >= self.merge_hold_frames:
                print(f"Formed Bond between {a1.element} & {a2.element}")
                venv.add_bond(a1, a2)

                a1.snap_back()
                a2.snap_back()
                a1.is_grabbed = False
                a2.is_grabbed = False
                a1.grabbed_by_hand = None
                a2.grabbed_by_hand = None

                self.merge_triggered = True
                self.reset()
        else:
            self.merge_counter = 0

        return full_pinch
