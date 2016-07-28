import numpy as np
import os

from vispy import gloo, scene
from vispy.util.transforms import ortho, translate

from .colormap import Colormap
from .util import eng_format

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

# Vertex and fragment shader to draw the colormap bar next to the plot
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

# Vertex and fragment shader to draw the actual data quads
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
    float normalized = clamp((v_value-z_min)/(z_max-z_min), 0.0, 1.0);
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

        self.colormap_program = gloo.Program(colormap_vert, colormap_frag)

        #  horizontal / vertical / diagonal
        self.line_type = None
        # x for a vertical line, y for a horizontal line
        self.line_coord = None
        # start and end points
        self.line_positions = [(0, 0), (0, 0)]

        self.linecut_program = gloo.Program(linecut_vert, linecut_frag)
        self.linecut_program['a_position'] = self.line_positions

        gloo.set_clear_color((1, 1, 1, 1))

    def set_data(self, data):
        self.data = data
        self.data_changed = True

        vertices = self.generate_vertices(data)

        self.xmin = np.nanmin(vertices['a_position'][:,0])
        self.xmax = np.nanmax(vertices['a_position'][:,0])
        self.ymin = np.nanmin(vertices['a_position'][:,1])
        self.ymax = np.nanmax(vertices['a_position'][:,1])

        self.cm_dx = (self.xmax - self.xmin) * 0.1

        self.view = translate((0, 0, 0))
        self.projection = ortho(self.xmin, self.xmax + self.cm_dx,
                                self.ymin, self.ymax, -1, 1)

        self.data_program['u_view'] = self.view
        self.data_program['u_projection'] = self.projection
        self.data_program['u_colormap'] = gloo.Texture1D(self.colormap.get_colors(),
                                                         interpolation='linear')

        self.colormap_program['u_view'] = self.view
        self.colormap_program['u_projection'] = self.projection

        self.linecut_program['u_view'] = self.view
        self.linecut_program['u_projection'] = self.projection

        self.vbo = gloo.VertexBuffer(vertices)
        self.data_program.bind(self.vbo)

        self.update()

    def generate_vertices(self, data):
        xq, yq = data.get_quadrilaterals(data.x, data.y)

        # Top left
        x1 = xq[0:-1, 0:-1].ravel()
        y1 = yq[0:-1, 0:-1].ravel()
        # Bottom left
        x2 = xq[1:,   0:-1].ravel()
        y2 = yq[1:,   0:-1].ravel()
        # Bottom right
        x3 = xq[1:,   1:].ravel()
        y3 = yq[1:,   1:].ravel()
        # Top right
        x4 = xq[0:-1, 1:].ravel()
        y4 = yq[0:-1, 1:].ravel()

        # Two triangles / six vertices per datapoint
        xy = np.concatenate((x1[:,np.newaxis], y1[:,np.newaxis],
                             x2[:,np.newaxis], y2[:,np.newaxis],
                             x4[:,np.newaxis], y4[:,np.newaxis],
                             x2[:,np.newaxis], y2[:,np.newaxis],
                             x4[:,np.newaxis], y4[:,np.newaxis],
                             x3[:,np.newaxis], y3[:,np.newaxis]), axis=1)

        total_vertices = len(x1) * 6
        vertices = xy.reshape((total_vertices, 2))

        vertex_data = np.zeros(total_vertices, dtype=[('a_position', np.float32, 2),
                                                      ('a_value',    np.float32, 1)])
        vertex_data['a_position'] = vertices

        # Repeat the values six times for every two datapoint triangles
        vertex_data['a_value'] = np.repeat(data.z.ravel(), 6, axis=0)

        return vertex_data

    def screen_to_data_coords(self, pos):
        sw, sh = self.size
        sx, sy = pos

        relx, rely = float(sx*1.1) / sw, float(sh - sy) / sh

        dx = self.xmin + (relx) * (self.xmax - self.xmin)
        dy = self.ymin + (rely) * (self.ymax - self.ymin)

        return dx, dy

    def draw_linecut(self, event, old_position=False, initial_press=False):
        # We need to check wether the canvas has had time to redraw itself
        # because continuous mouse movement events surpress the redrawing.
        if self.data is not None and self.has_redrawn:
            x_name, y_name, data_name = self.parent.get_axis_names()

            # If we need to draw the linecut at a new position
            if not old_position and event.button in [1, 2, 3]:
                x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))

                # Set up the parameters and data for either a horizontal or vertical linecut
                if event.button == 1:
                    self.line_type = 'horizontal'
                    self.line_coord = self.data.get_closest_y(y)
                    self.line_positions = [(self.xmin, self.line_coord),
                                           (self.xmax, self.line_coord)]

                    x, y, index = self.data.get_row_at(y)
                    z = np.nanmean(self.data.y[index,:])

                    self.parent.linecut.plot_linetrace(x, y, z, self.line_type,
                                                       self.line_coord,
                                                       self.parent.name,
                                                       x_name, data_name, y_name)
                elif event.button == 2:
                    self.line_type = 'diagonal'

                    if initial_press:
                        x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))
                        self.line_positions = [(x, y), (x, y)]
                    else:
                        x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))
                        self.line_positions[1] = (x, y)

                        x_points = np.linspace(self.line_positions[0][0], self.line_positions[1][0], 500)
                        y_points = np.linspace(self.line_positions[0][1], self.line_positions[1][1], 500)

                        if self.data_changed:
                            self.data.generate_triangulation()
                            self.data_changed = False

                        vals = self.data.interpolate(np.column_stack((x_points, y_points)))

                        dist = np.hypot(x_points - x_points[0], y_points - y_points[0])

                        self.parent.linecut.plot_linetrace(dist, vals, 0, self.line_type,
                                                       self.line_coord,
                                                       self.parent.name,
                                                       'Distance (-)', data_name, x_name)
                elif event.button == 3:
                    self.line_type = 'vertical'
                    self.line_coord = self.data.get_closest_x(x)
                    self.line_positions = [(self.line_coord, self.ymin),
                                           (self.line_coord, self.ymax)]

                    x, y, index = self.data.get_column_at(x)
                    z = np.nanmean(self.data.x[:,index])

                    self.parent.linecut.plot_linetrace(x, y, z, self.line_type,
                                                       self.line_coord,
                                                       self.parent.name,
                                                       y_name, data_name, x_name)

                self.has_redrawn = False

            self.linecut_program['a_position'] = self.line_positions

            self.update()
        else:
            self.update()

    def on_mouse_press(self, event):
        self.draw_linecut(event, initial_press=True)

    def on_mouse_move(self, event):
        if self.data is not None:
            sw, sh = self.size
            sx, sy = event.pos

            if 0 <= sx < sw and 0 <= sy < sh:
                x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))

                if not np.isnan(x) and not np.isnan(y):
                    xstr, ystr = eng_format(x, 1), eng_format(y, 1)
                    self.parent.status_bar.showMessage('X: %s\t\t\tY: %s' % (xstr, ystr))

                    self.draw_linecut(event)

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear()

        if self.data is not None:
            # Draw first the data, then colormap, and then linecut
            self.data_program['u_colormap'] = gloo.Texture1D(self.colormap.get_colors())
            self.data_program['z_min'] = self.colormap.min
            self.data_program['z_max'] = self.colormap.max
            self.data_program.draw('triangles')

            self.colormap_program['u_colormap'] = gloo.Texture1D(
                self.colormap.get_colors(), interpolation='linear')

            self.colormap_program['a_position'] = [(self.xmax + self.cm_dx*.2, self.ymax),
                                             (self.xmax + self.cm_dx*.2, self.ymin),
                                             (self.xmax + self.cm_dx, self.ymax),
                                             (self.xmax + self.cm_dx, self.ymin)]
            self.colormap_program['a_texcoord'] = [[1], [0], [1], [0]]
            self.colormap_program.draw('triangle_strip')

            self.linecut_program.draw('lines')

        self.has_redrawn = True
