from moviepy import VideoFileClip
import os

def cut_video_by_timestamps(input_path, timestamps, output_dir, output_prefix):
    video = VideoFileClip(input_path)
    duration = video.duration

    os.makedirs(output_dir, exist_ok=True)

    for idx, (start, end) in enumerate(timestamps):
        # Clamp start and end times to video duration
        start_clamped = max(0, min(start, duration))
        end_clamped = max(0, min(end, duration))

        if end_clamped <= start_clamped:
            print(f"Skipping segment {idx}: invalid or zero-length segment ({start} to {end})")
            continue

        segment = video.subclipped(start_clamped, end_clamped)
        output_path = os.path.join(
            output_dir,
            f"{output_prefix}_{round(start_clamped)}s_to_{round(end_clamped)}s.mp4"
        )
        segment.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Saved segment: {output_path}")

if __name__ == "__main__":
    input_video = "data/1_apexlegends.mp4"
    output_dir = "data/segments"
    output_prefix = "1_apexlegends"

    # Example list of (start, end) times in seconds
    timestamps = [(x, x+3) for x in range(0, 60, 3)]

    cut_video_by_timestamps(input_video, timestamps, output_dir, output_prefix)
