from pydub import AudioSegment
import os

def cut_audio(input_file, output_folder, segment_length=4000):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Calculate the number of segments
    num_segments = len(audio) // segment_length

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Cut the audio into segments
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        segment = audio[start_time:end_time]

        # Output file name
        output_file = os.path.join(output_folder, f"cut10_{i + 1}.wav")

        # Export the segment as a new WAV file
        segment.export(output_file, format="wav")

    # Handle the last segment
    start_time = num_segments * segment_length
    end_time = len(audio)
    last_segment = audio[start_time:end_time]
    output_file = os.path.join(output_folder, f"cut10_{num_segments + 1}.wav")
    last_segment.export(output_file, format="wav")

if __name__ == "__main__":
    # Replace 'input.wav' with the path to your input WAV file
    input_file_path = r"C:\Users\ksree\Downloads\chainsaw-07.wav"

    # Replace 'output_folder' with the path to the folder where you want to save the segments
    output_folder_path = r"C:\Users\ksree\PycharmProjects\pythonProject1\cut10"

    # Specify the length of each segment in milliseconds (4 seconds in this case)
    segment_length_ms = 4000

    cut_audio(input_file_path, output_folder_path, segment_length_ms)
