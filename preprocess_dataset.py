import time
import random
import librosa
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from multiprocessing import Pool
from audioread.ffdec import FFmpegAudioFile

from osu_classy.osu.beatmap import Beatmap
from osu_classy.osu.hit_objects import Circle, Slider, Spinner
from osu_classy.osu.sliders import Bezier
from osu_classy.utils.smooth_hit import smooth_hit


N_FFT = 2048
N_MFCC = 20
SR = 22050
FRAME_RATE = 6
HOP_LENGTH = int(SR * FRAME_RATE / 1000)
N_MELS = 64


def load_audio(audio_path):
    aro = FFmpegAudioFile(audio_path)
    wave, _ = librosa.load(aro, sr=SR)
    if wave.shape[0] == 0:
        raise ValueError("Empty audio file")

    # spectrogram
    spec = librosa.feature.mfcc(
        y=wave,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return spec


def hit_object_pairs(beatmap, frame_times):
    pairs = zip([None] + beatmap.hit_objects, beatmap.hit_objects + [None])
    a, b = next(pairs)
    for t in frame_times:
        while b is not None and b.t <= t:
            a, b = next(pairs)
        yield a, b


def from_beatmap(beatmap, frame_times):
    hit_signals = np.zeros((4, frame_times.shape[0]))
    for ho in beatmap.hit_objects:
        if isinstance(ho, Circle):
            hit_signals[0] += smooth_hit(frame_times, float(ho.t))
        elif isinstance(ho, Slider):
            hit_signals[1] += smooth_hit(
                frame_times, (float(ho.t), float(ho.end_time()))
            )
        elif isinstance(ho, Spinner):
            hit_signals[2] += smooth_hit(
                frame_times, (float(ho.t), float(ho.end_time()))
            )

        if ho.new_combo:
            hit_signals[3] += smooth_hit(frame_times, float(ho.t))

    slider_signals = np.zeros((2, frame_times.shape[0]))
    for ho in beatmap.hit_objects:
        if not isinstance(ho, Slider):
            continue

        if ho.slides > 1:
            single_slide = ho.slide_duration / ho.slides
            for i in range(1, ho.slides):
                slider_signals[0] += smooth_hit(
                    frame_times, float(ho.t + single_slide * i)
                )

            if isinstance(ho, Bezier):
                seg_len_t = np.cumsum([0] + [c.length for c in ho.path_segments])
                seg_boundaries = seg_len_t / ho.length * ho.slide_duration + ho.t
                for boundary in seg_boundaries:
                    slider_signals[1] += smooth_hit(frame_times, boundary)

    pos = []
    for t, (a, b) in zip(frame_times, hit_object_pairs(beatmap, frame_times)):
        if a is None:
            pos.append(b.start_pos())
        elif t < a.end_time():
            if isinstance(a, Spinner):
                pos.append(a.start_pos())
            else:
                single_slide = a.slide_duration / a.slides
                ts = (t - a.t) % (single_slide * 2) / single_slide
                if ts < 1:
                    pos.append(a.lerp(ts))
                else:
                    pos.append(a.lerp(2 - ts))
        elif b is None:
            pos.append(a.end_pos())
        else:
            f = (t - a.end_time()) / (b.t - a.end_time())
            pos.append((1 - f) * a.end_pos() + f * b.start_pos())
    cursor_signals = np.array(pos).T
    cursor_signals = (cursor_signals - cursor_signals.min(axis=1, keepdims=True)) / (
        cursor_signals.max(axis=1, keepdims=True)
        - cursor_signals.min(axis=1, keepdims=True)
    )

    return np.concatenate([hit_signals, slider_signals, cursor_signals], axis=0)


def prepare_map(data_dir, map_file):
    try:
        bm = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"Failed to load {map_file.name}: {e}")
        return

    if bm.mode != 0:
        print(f"Skipping non-std map {map_file.name}")
        return

    # try:
    #     star_ratings = calculateStarRating(
    #         filepath=map_file,
    #         returnAllDifficultyValues=True,
    #     )
    # except Exception as e:
    #     print(f"Failed to calculate SR for {map_file.name}: {e}")
    #     return

    # if star_ratings["nomod"]["total"] < 4.0:
    #     print(f"Skipping easy map {map_file.name}")
    #     return

    af_dir = "_".join(
        [bm.audio_filename.stem, *(s[1:] for s in bm.audio_filename.suffixes)]
    )
    map_dir = data_dir / map_file.parent.name / af_dir
    spec_path = map_dir / "spec.npz"
    map_path = map_dir / f"{map_file.stem}.map.npz"

    if map_path.exists():
        print(f"Skipping existing map {map_file.name}")
        return

    try:
        bm.parse_map_data()
    except Exception as e:
        print(f"Failed to parse {map_file.name}: {e}")
        return

    if spec_path.exists():
        for i in range(5):
            try:
                spec = np.load(spec_path)["spec"]
                break
            except ValueError:
                time.sleep(0.001 * 2**i)
        else:
            print(f"Failed to load spec for {map_file.name}")
            return
    else:
        try:
            spec = load_audio(bm.audio_filename)
        except Exception as e:
            print(f"Failed to load audio for {map_file.name}: {e}")
            return

        spec_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(spec_path, spec=spec)

    frame_times = (
        librosa.frames_to_time(
            np.arange(spec.shape[1]),
            sr=SR,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
        )
        * 1000
    )

    x = from_beatmap(bm, frame_times)
    np.savez_compressed(map_path, x=x)


if __name__ == "__main__":
    src_path = Path("/mnt/d/Games/osu!/Songs/")
    dst_path = Path("/mnt/d/Dataset/beatmaps_v1/")

    src_maps = list(src_path.glob("**/*.osu"))
    random.shuffle(src_maps)

    with Pool(processes=6) as pool:
        for _ in tqdm(
            pool.imap_unordered(partial(prepare_map, dst_path), src_maps),
            total=len(src_maps),
        ):
            pass
