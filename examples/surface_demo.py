import numpy as np
import H5Gizmos as gz
from resample3.extrude import Extruder
from resample3 import matrices

npz_path = "labels_and_image.npz"
data = np.load(npz_path)
input_volume = data["labels"]
print(f"Loaded input volume from {npz_path} with shape {input_volume.shape} and dtype {input_volume.dtype}")


class SurfaceDemo:

    def __init__(self, input_volume, width=1000, rescale=0.7):
        self.input_volume = input_volume
        self.width = width
        self.rescale = rescale
        maxshape = max(input_volume.shape)
        self.output_size = output_size = int(maxshape / rescale)
        self.plane_shape = plane_shape = (output_size,) * 2
        self.extruder = Extruder(input_volume, shape=plane_shape, min_value=0.0)
        self.scales = (10, 1, 1)
        self.matrix = matrices.projection_matrix(
            self.input_volume.shape, plane_shape, self.scales
        )

    def dashboard(self):
        output_size = self.output_size
        self.image = gz.Image(height=output_size, width=output_size)
        self.info = gz.Text("info here")
        title = "Positive surface projection demo"
        self.x_slider = gz.Slider(
            minimum=-np.pi,
            maximum=np.pi,
            step=np.pi / 100,
            value=0.0,
            on_change=self.update,
        )
        self.x_slider.css(width=str(self.width) + "px")
        self.y_slider = gz.Slider(
            minimum=-np.pi,
            maximum=np.pi,
            step=np.pi / 100,
            value=0.0,
            on_change=self.update,
        )
        self.y_slider.css(width=str(self.width) + "px")
        self.z_slider = gz.Slider(
            minimum=-np.pi,
            maximum=np.pi,
            step=np.pi / 100,
            value=0.0,
            on_change=self.update,
        )
        self.z_slider.css(width=str(self.width) + "px")
        dash = gz.Stack([
            title,
            self.image,
            self.x_slider,
            self.y_slider,
            self.z_slider,
            self.info,
        ])
        dash.call_when_started(self.update)
        self.dashboard = dash
        return dash

    def update(self, *ignored):
        rx = self.x_slider.value
        ry = self.y_slider.value
        rz = self.z_slider.value
        self.matrix = matrices.projection_matrix(
            self.input_volume.shape, self.plane_shape, self.scales, rx=rx, ry=ry, rz=rz
        )
        self.extruder.extrude(self.matrix)
        output_depths = self.extruder.output_depths
        display_depths = np.where(np.isfinite(output_depths), output_depths, 0.0)
        self.image.change_array(display_depths, scale=True)
        self.info.text(f"X rotation: {rx:.2f} rad  Y rotation: {ry:.2f} rad  Z rotation: {rz:.2f} rad")


if __name__ == "__main__":
    demo = SurfaceDemo(input_volume, rescale=0.7)
    gz.serve(demo.dashboard().link())
