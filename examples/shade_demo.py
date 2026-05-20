import numpy as np
import H5Gizmos as gz
from resample3.extrude import Extruder
from resample3.maximize import Maximizer
from resample3 import matrices

npz_path = "labels_and_image.npz"
data = np.load(npz_path)
input_labels = data["labels"]
input_image = data["img"]
print(f"Loaded input volume from {npz_path} with shape {input_labels.shape} and dtype {input_labels.dtype}")
print(f"Loaded input image from {npz_path} with shape {input_image.shape} and dtype {input_image.dtype}")


class ShadeDemo:

    def __init__(self, input_labels, input_image, width=1000, rescale=0.7):
        self.input_labels = input_labels
        self.input_image = input_image
        self.width = width
        self.rescale = rescale
        maxshape = max(input_labels.shape)
        self.output_size = output_size = int(maxshape / rescale)
        self.plane_shape = plane_shape = (output_size,) * 2
        self.extruder = Extruder(input_labels, shape=plane_shape, min_value=0.0)
        vmin = input_image.min()
        self.maximizer = Maximizer(input_image, shape=plane_shape, min_value=vmin)
        self.scales = (10, 1, 1)
        self.matrix = matrices.projection_matrix(
            self.input_labels.shape, plane_shape, self.scales
        )
        unique_labels = np.unique(input_labels.astype(np.intp, copy=False))
        max_label = int(unique_labels.max(initial=0))
        self.random_colors = random_colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
        '''
        random_values = np.random.default_rng().integers(
            0,
            256,
            size=(len(unique_labels), 3),
            dtype=np.uint8,
        )
        random_colors[unique_labels] = random_values
        random_colors[0] = (0, 0, 0)
        '''
        # random_colors rgb values in range 0..1.
        random_colors = np.random.default_rng().random(
            size=(len(unique_labels), 3),
            dtype=np.float32,
        )
        random_colors[0] = (0, 0, 0)
        self.random_colors = random_colors # combined below with shading to get final display colors

    def dashboard(self):
        output_size = self.output_size
        self.image = gz.Image(height=output_size, width=output_size)
        self.info = gz.Text("info here")
        title = "Positive surface projection demo"
        slider_width = str(self.width//2) + "px"
        self.x_slider = gz.Slider(
            minimum=-np.pi,
            maximum=np.pi,
            step=np.pi / 100,
            value=0.0,
            on_change=self.update,
        )
        self.x_slider.css(width=slider_width)
        self.y_slider = gz.Slider(
            minimum=-np.pi,
            maximum=np.pi,
            step=np.pi / 100,
            value=0.0,
            on_change=self.update,
        )
        self.y_slider.css(width=slider_width)
        self.z_slider = gz.Slider(
            minimum=-np.pi,
            maximum=np.pi,
            step=np.pi / 100,
            value=0.0,
            on_change=self.update,
        )
        self.z_slider.css(width=slider_width)
        self.mix_slider = gz.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.7,
            on_change=self.update,
        )
        self.mix_slider.css(width=slider_width)
        self.image_slider = gz.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.7,
            on_change=self.update,
        )
        self.image_slider.css(width=slider_width)
        dash = gz.Stack([
            title,
            self.image,
            ["x", self.x_slider],
            ["y", self.y_slider],
            ["z", self.z_slider],
            ["shading", self.mix_slider],
            ["image", self.image_slider],
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
            self.input_labels.shape, self.plane_shape, self.scales, rx=rx, ry=ry, rz=rz
        )
        self.extruder.extrude(self.matrix)
        output_image = self.maximizer.maximize(self.matrix).astype(float)
        imax = output_image.max()
        imin = output_image.min()
        nimg = (output_image - imin) / max(imax - imin, 1e-5)
        imix = self.image_slider.value
        imix1 = 1 - imix
        output_depths = self.extruder.output_depths
        mask = self.extruder.output_plane > 0
        mask[1:] = mask[1:] & mask[:-1]
        mask[:, 1:] = mask[:, 1:] & mask[:, :-1]
        test_depths = np.where(mask, output_depths, -output_depths.max())
        xchange = np.zeros_like(test_depths)
        xchange[1:] = test_depths[1:] - test_depths[:-1]
        ychange = np.zeros_like(test_depths)
        ychange[:, 1:] = test_depths[:, 1:] - test_depths[:, :-1]
        rshade = 0.5 * (np.atan(xchange) + 1)
        #red_intensity = np.clip(image_mix1 * red_intensity + image_mix * normalized_img, 0, 1)
        gshade = (0.5 * (np.atan(ychange) + 1))
        #green_intensity = np.clip(image_mix1 * green_intensity + image_mix * normalized_img, 0, 1)  
        bshade = (0.5 * (np.atan(-xchange - ychange) + 1))
        #blue_intensity = np.clip(image_mix1 * blue_intensity + image_mix * normalized_img, 0, 1)
        #display_image = np.zeros(test_depths.shape + (3,), dtype=np.uint8)
        #display_image[:, :, 0] = (red_intensity * 255).astype(np.uint8)
        #display_image[:, :, 1] = (green_intensity * 255).astype(np.uint8)
        #display_image[:, :, 2] = (blue_intensity * 255).astype(np.uint8)
        output_plane = self.extruder.output_plane.astype(np.intp, copy=False)
        cplane = self.random_colors[output_plane] # colors in range 0..1
        mix = self.mix_slider.value
        mix1 = 1 - mix
        # compute channels in range 0..1, then combine and convert to uint8.
        #print(type(imix1), type(nimg), type(imix), type(cplane), type(rshade))
        img_intensity = imix1 * nimg
        red_channel = imix * (mix * cplane[:,:,0] + mix1 * rshade) + img_intensity
        green_channel = imix * (mix * cplane[:,:,1] + mix1 * gshade) + img_intensity
        blue_channel = imix * (mix * cplane[:,:,2] + mix1 * bshade) + img_intensity
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        display_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
        self.image.change_array(display_image, scale=False)
        self.info.text(
            f"X rotation: {rx:.2f} rad  Y rotation: {ry:.2f} rad  Z rotation: {rz:.2f} rad  Mix: {mix:.2f}"
        )


if __name__ == "__main__":
    demo = ShadeDemo(input_labels, input_image, rescale=0.7)
    gz.serve(demo.dashboard().link())
