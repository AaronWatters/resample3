import numpy as np
import H5Gizmos as gz
from resample3.extrude import Extruder
from resample3 import matrices

npz_path = "labels_and_image.npz"
data = np.load(npz_path)
input_volume = data["labels"]
print(f"Loaded input volume from {npz_path} with shape {input_volume.shape} and dtype {input_volume.dtype}")


class ShadeDemo:

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

        unique_labels = np.unique(input_volume.astype(np.intp, copy=False))
        max_label = int(unique_labels.max(initial=0))
        self.random_colors = random_colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
        random_values = np.random.default_rng().integers(
            0,
            256,
            size=(len(unique_labels), 3),
            dtype=np.uint8,
        )
        random_colors[unique_labels] = random_values
        random_colors[0] = (0, 0, 0)

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
        self.mix_slider = gz.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.7,
            on_change=self.update,
        )
        self.mix_slider.css(width=str(self.width) + "px")
        dash = gz.Stack([
            title,
            self.image,
            self.x_slider,
            self.y_slider,
            self.z_slider,
            self.mix_slider,
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
        mask = self.extruder.output_plane > 0
        mask[1:] = mask[1:] & mask[:-1]
        mask[:, 1:] = mask[:, 1:] & mask[:, :-1]
        test_depths = np.where(mask, output_depths, -output_depths.max())
        xchange = np.zeros_like(test_depths)
        xchange[1:] = test_depths[1:] - test_depths[:-1]
        ychange = np.zeros_like(test_depths)
        ychange[:, 1:] = test_depths[:, 1:] - test_depths[:, :-1]
        red_intensity = np.clip(0.5 * (np.atan(xchange) + 1), 0, 1)
        green_intensity = np.clip(0.5 * (np.atan(ychange) + 1), 0, 1)
        blue_intensity = np.clip(0.5 * (np.atan(-xchange - ychange) + 1), 0, 1)
        display_image = np.zeros(test_depths.shape + (3,), dtype=np.uint8)
        display_image[:, :, 0] = (red_intensity * 255).astype(np.uint8)
        display_image[:, :, 1] = (green_intensity * 255).astype(np.uint8)
        display_image[:, :, 2] = (blue_intensity * 255).astype(np.uint8)
        output_plane = self.extruder.output_plane.astype(np.intp, copy=False)
        colorized_plane = self.random_colors[output_plane]
        mix = self.mix_slider.value
        display_image = (
            (1 - mix) * display_image.astype(np.float32)
            + mix * colorized_plane.astype(np.float32)
        ).astype(np.uint8)
        self.image.change_array(display_image, scale=False)
        self.info.text(
            f"X rotation: {rx:.2f} rad  Y rotation: {ry:.2f} rad  Z rotation: {rz:.2f} rad  Mix: {mix:.2f}"
        )


if __name__ == "__main__":
    demo = ShadeDemo(input_volume, rescale=0.7)
    gz.serve(demo.dashboard().link())
