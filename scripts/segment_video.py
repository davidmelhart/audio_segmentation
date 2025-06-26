from moviepy import VideoFileClip

def cut_video_into_segments(input_path, segment_length_sec, output_prefix):
    video = VideoFileClip(input_path)
    duration = int(video.duration)
    start = 0
    segment_idx = 0
    
    while start < duration:
        end = min(start + segment_length_sec, duration)
        segment = video.subclipped(start, end)
        output_path = f"data/segments/{output_prefix}_segment_{segment_idx:03d}.mp4"
        segment.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Saved segment: {output_path}")
        start += segment_length_sec
        segment_idx += 1

if __name__ == "__main__":
    input_video = "data/1_apexlegends.mp4"         # your input video file
    segment_length = 5                        # segment length in seconds
    output_prefix = "1_apexlegends"            # prefix for output files
    
    cut_video_into_segments(input_video, segment_length, output_prefix)
