import cairosvg
from PIL import Image
import io

def convert_svg_to_tiff(svg_path, tiff_output_path, width=11956, height=8808):
    # Render the SVG to PNG in memory
    png_data = cairosvg.svg2png(url=svg_path, output_width=width, output_height=height)

    # Convert PNG data to an image object with PIL
    image = Image.open(io.BytesIO(png_data))

    # Save the image as a TIFF file
    image.save(tiff_output_path, format='TIFF')

# Example usage:
convert_svg_to_tiff('output1.svg', 'converted_output.tif', width=11956, height=8808)
