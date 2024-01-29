import cv2
import os

def create_video_from_png(folder_path, output_video_path, fps=30):
    # Get the list of PNG files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort()  # Ensure files are in order

    if not png_files:
        print("No PNG files found in the folder.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(folder_path, png_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop through PNG files and write frames to the video
    for png_file in png_files:
        image_path = os.path.join(folder_path, png_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()

    print(f"Video created successfully: {output_video_path}")

# Example usage
folder_path = "image/"
output_video_path = "vid/video_undefended30fps.mp4"
create_video_from_png(folder_path, output_video_path)
