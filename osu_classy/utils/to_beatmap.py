import bisect

import numpy as np
import scipy
import bezier

from osu_classy.osu.hit_objects import TimingPoint
from osu_classy.utils.smooth_hit import smooth_hit, HIT_SD

BEAT_DIVISOR = 4

map_template = """osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
Mode: 0

[Metadata]
Title: {title}
TitleUnicode: {title}
Artist: {artist}
ArtistUnicode: {artist}
Creator: osu!dreamer
Version: {version}
Tags: osu_dreamer

[Difficulty]
HPDrainRate: 0
CircleSize: 3
OverallDifficulty: 0
ApproachRate: 9.5
SliderMultiplier: 1
SliderTickRate: 1

[TimingPoints]
{timing_points}

[HitObjects]
{hit_objects}
"""

F_B = max(2, HIT_SD * 6)
F = smooth_hit(np.arange(-F_B, F_B + 1), 0.0)


def _decode(sig, peak_h, hit_offset):
    corr = scipy.signal.correlate(sig, F, mode="same")
    hit_peaks = scipy.signal.find_peaks(corr, height=peak_h)[0] + hit_offset
    return hit_peaks.astype(np.int32).tolist()


def decode_hit(sig):
    return _decode(sig, 0.5, 0)


def decode_hold(sig):
    sig_grad = np.gradient(sig)
    start_sig = np.maximum(0, sig_grad)
    end_sig = -np.minimum(0, sig_grad)

    start_idxs = _decode(start_sig, 0.25, 1)
    end_idxs = _decode(end_sig, 0.25, -1)

    while len(start_idxs) and len(end_idxs) and start_idxs[0] >= end_idxs[0]:
        end_idxs.pop(0)

    if len(start_idxs) > len(end_idxs):
        start_idxs = start_idxs[: len(end_idxs) - len(start_idxs)]
    elif len(end_idxs) > len(start_idxs):
        end_idxs = end_idxs[: len(start_idxs) - len(end_idxs)]

    return start_idxs, end_idxs


def hodo(p):
    return p.shape[0] * (p[1:] - p[:-1])


def q(p, t):
    """evaluates bezier at t"""
    return bezier.Curve.from_nodes(p.T).evaluate_multi(t).T


def qprime(p, t):
    """evaluates bezier first derivative at t"""
    return bezier.Curve.from_nodes(hodo(p).T).evaluate_multi(t).T


def qprimeprime(p, t):
    """evaluates bezier second derivative at t"""
    return bezier.Curve.from_nodes(hodo(hodo(p)).T).evaluate_multi(t).T


def normalize(v):
    magnitude = np.sqrt(np.dot(v, v))
    if magnitude < np.finfo(float).eps:
        return v
    return v / magnitude


def compute_error(p, points, u):
    errs = ((q(p, u) - points) ** 2).sum(-1)
    split_point = errs.argmax()
    return errs[split_point], split_point


def fit_bezier(points, max_err, left_tangent=None, right_tangent=None):
    """fit one (ore more) Bezier curves to a set of points"""

    assert points.shape[0] > 0

    weights = (
        lambda x, n: (float(x) ** -np.arange(1, n + 1)) / (1 - float(x) ** -n) * (x - 1)
    )(2, min(5, len(points) - 2))

    if left_tangent is None:
        # points[1] - points[0]
        l_vecs = points[2 : 2 + len(weights)] - points[1]
        left_tangent = normalize(np.einsum("np,n->p", l_vecs, weights))

    if right_tangent is None:
        # points[-2] - points[-1]
        r_vecs = points[-3 : -3 - len(weights) : -1] - points[-2]
        right_tangent = normalize(np.einsum("np,n->p", r_vecs, weights))

    # use heuristic if region only has two points in it
    if len(points) == 2:
        dist = np.linalg.norm(points[0] - points[1]) / 3.0
        return [
            [
                points[0],
                points[0] + left_tangent * dist,
                points[1] + right_tangent * dist,
                points[1],
            ]
        ]

    u = None
    bez_curve = None
    for _ in range(32):
        if u is None:
            # parameterize points
            u = [0]
            u[1:] = np.cumsum(np.linalg.norm(points[1:] - points[:-1], axis=1))
            u /= u[-1]
        else:
            # iterate parameterization
            u = newton_raphson_root_find(bez_curve, points, u)

        bez_curve = generate_bezier(points, u, left_tangent, right_tangent)
        err, split_point = compute_error(bez_curve, points, u)

        if err < max_err:
            # check if line is a good fit
            line_err, _ = compute_error(bez_curve[[0, -1]], points, u)
            if line_err < max_err:
                return [bez_curve[[0, -1]]]

            return [bez_curve]

    # Fitting failed -- split at max error point and fit recursively
    center_tangent = normalize(points[split_point - 1] - points[split_point + 1])
    return [
        *fit_bezier(points[: split_point + 1], max_err, left_tangent, center_tangent),
        *fit_bezier(points[split_point:], max_err, -center_tangent, right_tangent),
    ]


def generate_bezier(points, u, left_tangent, right_tangent):
    bez_curve = np.array([points[0], points[0], points[-1], points[-1]])

    # compute the A's
    A = (3 * (1 - u) * u * np.array([1 - u, u])).T[..., None] * np.array(
        [left_tangent, right_tangent]
    )

    # Create the C and X matrices
    C = np.einsum("lix,ljx->ij", A, A)
    X = np.einsum("lix,lx->i", A, points - q(bez_curve, u))

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if abs(det_C0_C1) < 1e-5 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if abs(det_C0_C1) < 1e-5 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text)
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent NewtonRaphsonRootFind() call)
    seg_len = np.linalg.norm(points[0] - points[-1])
    epsilon = 1e-6 * seg_len
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bez_curve[1] += left_tangent * (seg_len / 3.0)
        bez_curve[2] += right_tangent * (seg_len / 3.0)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bez_curve[1] += left_tangent * alpha_l
        bez_curve[2] += right_tangent * alpha_r

    return bez_curve


def newton_raphson_root_find(bez, points, u):
    """
    Newton's root finding algorithm calculates f(x)=0 by reiterating
    x_n+1 = x_n - f(x_n)/f'(x_n)
    We are trying to find curve parameter u for some point p that minimizes
    the distance from that point to the curve. Distance point to curve is d=q(u)-p.
    At minimum distance the point is perpendicular to the curve.
    We are solving
    f = q(u)-p * q'(u) = 0
    with
    f' = q'(u) * q'(u) + q(u)-p * q''(u)
    gives
    u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    """

    d = q(bez, u) - points
    qp = qprime(bez, u)
    num = (d * qp).sum(-1)
    den = (qp**2 + d * qprimeprime(bez, u)).sum(-1)

    return u + np.where(den == 0, 0, num / den)


def to_sorted_hits(hit_signal):
    """
    returns a list of tuples representing each hit object sorted by start:
        `(start_idx, end_idx, object_type, new_combo)`

    `hit_signal`: [4,L] array of [0,1] where:
    - [0] represents hits
    - [1] represents slider holds
    - [2] represents spinner holds
    - [3] represents new combos
    """

    tap_sig, slider_sig, spinner_sig, new_combo_sig = hit_signal

    tap_idxs = decode_hit(tap_sig)
    slider_start_idxs, slider_end_idxs = decode_hold(slider_sig)
    spinner_start_idxs, spinner_end_idxs = decode_hold(spinner_sig)
    new_combo_idxs = decode_hit(new_combo_sig)

    sorted_hits = sorted(
        [
            *[(t, t, 0, False) for t in tap_idxs],
            *[
                (s, e, 1, False)
                for s, e in zip(sorted(slider_start_idxs), sorted(slider_end_idxs))
            ],
            *[
                (s, e, 2, False)
                for s, e in zip(sorted(spinner_start_idxs), sorted(spinner_end_idxs))
            ],
        ]
    )

    # associate hits with new combos
    for new_combo_idx in new_combo_idxs:
        idx = bisect.bisect_left(sorted_hits, (new_combo_idx,))
        if idx == len(sorted_hits):
            idx = idx - 1
        elif idx > 0 and abs(new_combo_idx - sorted_hits[idx][0]) > abs(
            sorted_hits[idx - 1][0] - new_combo_idx
        ):
            idx = idx - 1
        sorted_hits[idx] = (*sorted_hits[idx][:3], True)

    return sorted_hits


def to_playfield_coordinates(cursor_signal):
    """
    transforms the cursor signal to osu!pixel coordinates
    """

    # rescale to fill the entire playfield
    # cs_valid_min = cursor_signal.min(axis=1, keepdims=True)
    # cs_valid_max = cursor_signal.max(axis=1, keepdims=True)
    # cursor_signal = (cursor_signal - cs_valid_min) / (cs_valid_max - cs_valid_min)

    # pad so that the cursor isn't too close to the edges of the screen
    # padding = 0.
    # cursor_signal = padding + cursor_signal * (1 - 2*padding)
    return cursor_signal * np.array([[512], [384]])


def to_slider_decoder(frame_times, cursor_signal, slider_signal):
    """
    returns a function that takes a start and end frame index and returns:
    - slider length
    - number of slides
    - slider control points
    """
    repeat_sig, seg_boundary_sig = slider_signal

    repeat_idxs = np.zeros_like(frame_times)
    repeat_idxs[decode_hit(repeat_sig)] = 1
    seg_boundary_idxs = decode_hit(seg_boundary_sig)

    def decoder(a, b):
        slides = int(sum(repeat_idxs[a : b + 1]) + 1)
        ctrl_pts = []
        length = 0
        sb_idxs = [s for s in seg_boundary_idxs if a < s < b]
        for seg_start, seg_end in zip([a] + sb_idxs, sb_idxs + [b]):
            if len(cursor_signal.T[seg_start : seg_end + 1]) == 0:
                continue
            for b in fit_bezier(cursor_signal.T[seg_start : seg_end + 1], max_err=100):
                b = np.array(b).round().astype(int)
                ctrl_pts.extend(b)
                length += bezier.Curve.from_nodes(b.T).length

        return length, slides, ctrl_pts

    return decoder


def to_beatmap(metadata, sig, frame_times, timing):
    """
    returns the beatmap as the string contents of the beatmap file
    """

    hit_signal, sig = np.split(sig, (4,))
    slider_signal, sig = np.split(sig, (2,))
    cursor_signal, sig = np.split(sig, (2,))
    assert sig.shape[0] == 0

    # process hit signal
    sorted_hits = to_sorted_hits(hit_signal)

    # process slider signal
    slider_decoder = to_slider_decoder(frame_times, cursor_signal, slider_signal)

    # process cursor signal
    cursor_signal = to_playfield_coordinates(cursor_signal)

    # `timing` can be one of:
    # - List[TimingPoint] : timed according to timing points
    # - None : no prior knowledge of audio timing
    # - number : audio is constant BPM
    if isinstance(timing, list) and len(timing) > 0:
        beat_snap, timing_points = True, timing
    elif timing is None:
        # TODO: compute tempo from hit times

        # the following code only works when the whole song is a constant tempo

        # diff_dist = scipy.stats.gaussian_kde([
        #     np.log(frame_times[b[0]] - frame_times[a[0]])
        #     for a,b in zip(sorted_hits[:-1], sorted_hits[1:])
        # ])
        # x = np.linspace(0,20,1000)
        # timing_beat_len = np.exp(x[diff_dist(x).argmax()])

        beat_snap, timing_points = False, [TimingPoint(0, 1000, None, 4, None)]
    elif isinstance(timing, (int, float)):
        timing_beat_len = 60.0 * 1000.0 / float(timing)
        # compute timing offset
        frames = []
        for i, _, _, _ in sorted_hits:
            if i >= len(frame_times):
                continue
            frames.append(frame_times[i] % timing_beat_len)
        offset_dist = scipy.stats.gaussian_kde(frames)
        offset = (
            offset_dist.pdf(np.linspace(0, timing_beat_len, 1000)).argmax()
            / 1000.0
            * timing_beat_len
        )

        beat_snap, timing_points = True, [
            TimingPoint(offset, timing_beat_len, None, 4, None)
        ]

    hos = []  # hit objects
    tps = []  # timing points

    # dur = length / (slider_mult * 100 * SV) * beat_length
    # dur = length / (slider_mult * 100) / SV * beat_length
    # SV  = length / dur / (slider_mult * 100) * beat_length
    # SV  = length / dur / (slider_mult * 100 / beat_length)
    # => base_slider_vel = slider_mult * 100 / beat_length
    beat_length = timing_points[0].beat_length
    base_slider_vel = 100 / beat_length
    beat_offset = timing_points[0].t

    def add_hit_circle(i, j, t, u, new_combo):
        x, y = cursor_signal[:, i].round().astype(int)
        hos.append(f"{x},{y},{t},{1 + new_combo},0,0:0:0:0:")

    def add_spinner(i, j, t, u, new_combo):
        if t == u:
            # start and end time are the same, add a hit circle instead
            return add_hit_circle(i, j, t, u, new_combo)
        hos.append(f"256,192,{t},{8 + new_combo},0,{u}")

    def add_slider(i, j, t, u, new_combo):
        if t == u:
            # start and end time are the same, add a hit circle instead
            return add_hit_circle(i, j, t, u, new_combo)

        length, slides, ctrl_pts = slider_decoder(i, j)

        if length == 0:
            # slider has zero length, add a hit circle instead
            return add_hit_circle(i, j, t, u, new_combo)

        SV = length * slides / (u - t) / base_slider_vel

        x1, y1 = ctrl_pts[0]
        curve_pts = "|".join(f"{x}:{y}" for x, y in ctrl_pts[1:])
        hos.append(f"{x1},{y1},{t},{2 + new_combo},0,B|{curve_pts},{slides},{length}")

        if len(tps) == 0:
            print(
                "warning: inherited timing point added before any uninherited timing points"
            )
        tps.append(f"{t},{-100/SV},4,0,0,50,0,0")

    last_up = None
    for i, j, t_type, new_combo in sorted_hits:
        if any(_i >= len(frame_times) for _i in (i, j)):
            continue
        t, u = frame_times[i], frame_times[j]
        if beat_snap:
            beat_f_len = beat_length / BEAT_DIVISOR
            t = round((t - beat_offset) / beat_f_len) * beat_f_len + beat_offset
            u = round((u - beat_offset) / beat_f_len) * beat_f_len + beat_offset

        t, u = int(t), int(u)

        # add timing points
        if len(timing_points) > 0 and t > timing_points[0].t:
            tp = timing_points.pop(0)
            tps.append(f"{tp.t},{tp.beat_length},{tp.meter},0,0,50,1,0")
            beat_length = tp.beat_length
            base_slider_vel = 100 / beat_length
            beat_offset = tp.t

        # ignore objects that start before the previous one ends
        if last_up is not None and t <= last_up + 1:
            continue

        [add_hit_circle, add_slider, add_spinner][t_type](
            i, j, t, u, 4 if new_combo else 0
        )
        last_up = u

    return map_template.format(
        **metadata, timing_points="\n".join(tps), hit_objects="\n".join(hos)
    )
