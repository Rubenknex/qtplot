import numpy as np
import os
import logging

from vispy import gloo, scene
from vispy.util.transforms import ortho, translate

from .colormap import Colormap
from .util import eng_format

logger = logging.getLogger(__name__)

# Vertex and fragment shader used to draw the linecut line
# The red color is hardcoded in the fragment shader
linecut_vert = """
attribute vec2 a_position;

uniform mat4 u_view;
uniform mat4 u_projection;

void main()
{
    gl_Position = u_projection * u_view * vec4(a_position, 0.0, 1.0);
}
"""

linecut_frag = """
void main()
{
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

# Vertex and fragment shader to draw the colorbar next to the plot
colormap_vert = """
attribute vec2 a_position;
attribute float a_texcoord;

uniform mat4 u_view;
uniform mat4 u_projection;

varying float v_texcoord;

void main (void) {
    v_texcoord = a_texcoord;

    gl_Position = u_projection * u_view * vec4(a_position.x, a_position.y, 0.0, 1.0);
}
"""

colormap_frag = """
uniform sampler1D u_colormap;

varying float v_texcoord;

void main()
{
    gl_FragColor = texture1D(u_colormap, v_texcoord);
}
"""

# Vertex and fragment shader to draw the actual data vertices
data_vert = """
attribute vec2 a_position;
attribute float a_value;

uniform mat4 u_view;
uniform mat4 u_projection;

varying float v_value;

void main()
{
    gl_Position = u_projection * u_view * vec4(a_position, 0.0, 1.0);
    v_value = a_value;
}
"""

data_frag = """
uniform float z_min;
uniform float z_max;
uniform sampler1D u_colormap;

varying float v_value;

void main()
{
    float normalized = clamp((v_value-z_min)/(z_max-z_min), 0.01, 0.99);
    gl_FragColor = texture1D(u_colormap, normalized);
}
"""


class Canvas(scene.SceneCanvas):
    """
    Handles the fast drawing of data using OpenGL for real-time editing.

    A data point is drawn using two triangles to form a quad,
    it is colored by using the normalized data value and a
    colormap texture in the fragment shader.
    """

    def __init__(self, parent=None):
        scene.SceneCanvas.__init__(self, parent=parent)

        self.parent = parent
        self.has_redrawn = True

        self.data = None
        self.data_changed = False
        self.data_program = gloo.Program(data_vert, data_frag)

        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'colormaps/transform/Seismic.npy')
        self.colormap = Colormap(path)

        self.colorbar_program = gloo.Program(colormap_vert, colormap_frag)

        #  horizontal / vertical / diagonal
        self.line_type = None
        # x for a vertical line, y for a horizontal line
        self.line_coord = None
        # Start and end points for a linecut
        self.mouse_start = (0, 0)
        self.mouse_end = (0, 0)

        self.linecut_program = gloo.Program(linecut_vert, linecut_frag)
        self.linecut_program['a_position'] = [self.mouse_start, self.mouse_end]

        gloo.set_clear_color((1, 1, 1, 1))

    def set_data(self, data):
        self.data = data
        self.data_changed = True

        vertices = self.generate_vertices(data)

        self.xmin = np.nanmin(vertices['a_position'][:, 0])
        self.xmax = np.nanmax(vertices['a_position'][:, 0])
        self.ymin = np.nanmin(vertices['a_position'][:, 1])
        self.ymax = np.nanmax(vertices['a_position'][:, 1])

        if self.xmin == self.xmax or self.ymin == self.ymax:
            logger.error(('Cannot plot because min and max values of'
                          ' vertices are identical'))

            return

        # Determines the width of the colorbar
        self.cm_dx = (self.xmax - self.xmin) * 0.1

        self.view = translate((0, 0, 0))

        # Orthogonal projection matrix
        self.projection = ortho(self.xmin, self.xmax + self.cm_dx,
                                self.ymin, self.ymax, -1, 1)

        self.data_program['u_view'] = self.view
        self.data_program['u_projection'] = self.projection

        cmap_texture = gloo.Texture1D(self.colormap.get_colors(),
                                      interpolation='linear')
        self.data_program['u_colormap'] = cmap_texture

        self.colorbar_program['u_view'] = self.view
        self.colorbar_program['u_projection'] = self.projection

        self.linecut_program['u_view'] = self.view
        self.linecut_program['u_projection'] = self.projection

        self.vbo = gloo.VertexBuffer(vertices)
        self.data_program.bind(self.vbo)

        self.update()

    def generate_vertices(self, data):
        """ Generate vertices for the dataset quadrilaterals """
        xq, yq = data.get_quadrilaterals(data.x, data.y)

        # Top left
        x1 = xq[0:-1, 0:-1].ravel()
        y1 = yq[0:-1, 0:-1].ravel()
        # Bottom left
        x2 = xq[1:, 0:-1].ravel()
        y2 = yq[1:, 0:-1].ravel()
        # Bottom right
        x3 = xq[1:, 1:].ravel()
        y3 = yq[1:, 1:].ravel()
        # Top right
        x4 = xq[0:-1, 1:].ravel()
        y4 = yq[0:-1, 1:].ravel()

        # Two triangles / six vertices per datapoint
        xy = np.concatenate((x1[:, np.newaxis], y1[:, np.newaxis],
                             x2[:, np.newaxis], y2[:, np.newaxis],
                             x4[:, np.newaxis], y4[:, np.newaxis],
                             x2[:, np.newaxis], y2[:, np.newaxis],
                             x4[:, np.newaxis], y4[:, np.newaxis],
                             x3[:, np.newaxis], y3[:, np.newaxis]), axis=1)

        total_vertices = len(x1) * 6
        vertices = xy.reshape((total_vertices, 2))

        dtype = [('a_position', np.float32, 2), ('a_value', np.float32, 1)]
        vertex_data = np.zeros(total_vertices, dtype=dtype)
        vertex_data['a_position'] = vertices

        # Repeat the values six times for every six vertices required
        # by the two datapoint triangles
        vertex_data['a_value'] = np.repeat(data.z.ravel(), 6, axis=0)

        return vertex_data

    def screen_to_data_coords(self, pos):
        """ Convert mouse position in the plot to data coordinates """
        screen_w, screen_h = self.size
        screen_x, screen_y = pos

        # Calculate in normalized coordinates
        relx = float(screen_x * 1.1) / screen_w
        rely = float(screen_h - screen_y) / screen_h

        # Convert to data coords using data min/max values
        dx = self.xmin + (relx) * (self.xmax - self.xmin)
        dy = self.ymin + (rely) * (self.ymax - self.ymin)

        return dx, dy

    def draw_linecut(self, event, old_position=False, initial_press=False):
        """ Draw the linecut depending on which mouse button was used """
        # We need to check whether the canvas has had time to redraw itself
        # because continuous mouse movement events are too fast and
        # surpress the redrawing.
        if self.data is not None and self.has_redrawn:
            x_name, y_name, data_name = self.parent.get_axis_names()

            # If we need to draw the linecut at a new position
            if not old_position and event.button in [1, 2, 3]:
                x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))

                # Set up the parameters and data for either a horizontal or
                # vertical linecut
                if event.button == 1:
                    self.draw_horizontal_linecut(y)
                elif event.button == 2:
                    self.draw_arbitrary_linecut(x, y, initial_press)
                elif event.button == 3:
                    self.draw_vertical_linecut(x)

                self.has_redrawn = False
            elif old_position:
                # Drawing the linecut at the old position
                if self.line_type == 'horizontal':
                    self.draw_horizontal_linecut(self.line_coord)
                elif self.line_type == 'diagonal':
                    self.draw_arbitrary_linecut(*self.mouse_end,
                                                initial_press=False)
                elif self.line_type == 'vertical':
                    self.draw_vertical_linecut(self.line_coord)

                self.has_redrawn = False

            # Set the line endpoints in the shader program
            self.linecut_program['a_position'] = [self.mouse_start,
                                                  self.mouse_end]

        self.update()

    def draw_horizontal_linecut(self, y):
        self.line_type = 'horizontal'
        self.line_coord = self.data.get_closest_y(y)
        self.mouse_start = (self.xmin, self.line_coord)
        self.mouse_end = (self.xmax, self.line_coord)

        # Get the data row
        x, y, row_numbers, index = self.data.get_row_at(y)
        z = np.nanmean(self.data.y[index, :])

        x_name, y_name, data_name = self.parent.get_axis_names()

        self.parent.linecut.plot_linetrace(x, y, z, row_numbers, self.line_type,
                                           self.line_coord,
                                           self.parent.name,
                                           x_name, data_name,
                                           y_name)

    def draw_vertical_linecut(self, x):
        self.line_type = 'vertical'
        self.line_coord = self.data.get_closest_x(x)
        self.mouse_start = (self.line_coord, self.ymin)
        self.mouse_end = (self.line_coord, self.ymax)

        # Get the data column
        x, y, row_numbers, index = self.data.get_column_at(x)
        z = np.nanmean(self.data.x[:, index])

        x_name, y_name, data_name = self.parent.get_axis_names()

        self.parent.linecut.plot_linetrace(x, y, z, row_numbers, self.line_type,
                                           self.line_coord,
                                           self.parent.name,
                                           y_name, data_name,
                                           x_name)

    def draw_arbitrary_linecut(self, x, y, initial_press):
        self.line_type = 'diagonal'

        if initial_press:
            # Store the initial location as start and end
            self.mouse_start = (x, y)
            self.mouse_end = (x, y)
        else:
            self.mouse_end = (x, y)

            # Create datapoints on the line to interpolate over
            x_start, y_start = self.mouse_start
            x_points = np.linspace(x_start, x, 500)
            y_points = np.linspace(y_start, y, 500)

            if self.data_changed:
                self.data.generate_triangulation()
                self.data_changed = False

            vals = self.data.interpolate(np.column_stack((x_points, y_points)))

            # Create data for the x-axis using hypotenuse
            dist = np.hypot(x_points - x_points[0], y_points - y_points[0])

            x_name, y_name, data_name = self.parent.get_axis_names()

            self.parent.linecut.plot_linetrace(dist, vals, 0, None,
                                               self.line_type,
                                               self.line_coord,
                                               self.parent.name,
                                               'Distance (-)',
                                               data_name, x_name)

            # Display slope and inverse slope in status bar
            dx = x - x_start
            dy = y - y_start
            text = 'Slope: {:.3e}\tInv: {:.3e}'.format(dy / dx, dx / dy)

            self.parent.l_slope.setText(text)

    def on_mouse_press(self, event):
        self.draw_linecut(event, initial_press=True)

    def on_mouse_move(self, event):
        if self.data is not None:
            sw, sh = self.size
            sx, sy = event.pos

            # If we are within the plot window
            if 0 <= sx < sw and 0 <= sy < sh:
                x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))

                if not np.isnan(x) and not np.isnan(y):
                    # Show the coordinates in the statusbar
                    text = 'X: %s\tY: %s' % (eng_format(x, 1),
                                             eng_format(y, 1))
                    self.parent.l_position.setText(text)

                    # If a mouse button was pressed, try to redraw linecut
                    if len(event.buttons) > 0:
                        self.draw_linecut(event)

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear()

        if self.data is not None:
            # Draw first the data, then colormap, and then linecut
            cmap_texture = gloo.Texture1D(self.colormap.get_colors(),
                                          interpolation='linear')

            # Drawing of the plot
            self.data_program['u_colormap'] = cmap_texture
            self.data_program['z_min'] = self.colormap.min
            self.data_program['z_max'] = self.colormap.max
            self.data_program.draw('triangles')

            # Drawing of the colormap bar
            self.colorbar_program['u_colormap'] = cmap_texture
            colorbar_vertices = [(self.xmax + self.cm_dx * .2, self.ymax),
                                 (self.xmax + self.cm_dx * .2, self.ymin),
                                 (self.xmax + self.cm_dx, self.ymax),
                                 (self.xmax + self.cm_dx, self.ymin)]
            self.colorbar_program['a_position'] = colorbar_vertices
            self.colorbar_program['a_texcoord'] = [[1], [0], [1], [0]]
            self.colorbar_program.draw('triangle_strip')

            # Drawing of the linecut
            self.linecut_program.draw('lines')

        self.has_redrawn = True
