import numpy as np
import logging

from vispy import gloo, scene
from vispy.util.transforms import ortho, translate

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

        view = translate((0, 0, 0))

        # The vertex buffer object containing data vertices
        self.vbo = None
        self.vbo_line = None

        self.data_program = gloo.Program(data_vert, data_frag)
        self.data_program['u_view'] = view

        self.colormap = None

        self.colorbar_program = gloo.Program(colormap_vert, colormap_frag)
        self.colorbar_program['u_view'] = view

        #  horizontal / vertical / diagonal
        self.line_type = None
        # x for a vertical line, y for a horizontal line
        self.line_coord = None

        self.linecut_program = gloo.Program(linecut_vert, linecut_frag)
        self.linecut_program['u_view'] = view
        self.linecut_program['a_position'] = [(0, 0), (0, 0)]

        gloo.set_clear_color((1, 1, 1, 1))

    def set_data(self, xq, yq, z):
        vertices = self.generate_vertices(xq, yq, z)

        self.xmin = np.nanmin(vertices['a_position'][:, 0])
        self.xmax = np.nanmax(vertices['a_position'][:, 0])
        self.ymin = np.nanmin(vertices['a_position'][:, 1])
        self.ymax = np.nanmax(vertices['a_position'][:, 1])

        if self.xmin == self.xmax or self.ymin == self.ymax:
            raise ValueError('Cannot plot because min and max values of'
                             ' vertices are identical')

        # Determines the width of the colorbar
        self.cm_dx = (self.xmax - self.xmin) * 0.1

        # Orthogonal projection matrix
        self.projection = ortho(self.xmin, self.xmax + self.cm_dx,
                                self.ymin, self.ymax, -1, 1)

        # Upload the projection matrix
        self.data_program['u_projection'] = self.projection
        self.colorbar_program['u_projection'] = self.projection
        self.linecut_program['u_projection'] = self.projection

        # Bind the VBO
        self.vbo = gloo.VertexBuffer(vertices)
        self.data_program.bind(self.vbo)

        self.update()

    def set_linetrace_data(self, x, y):
        data = np.zeros(len(x), dtype=[('a_position', np.float32, 2)])
        data['a_position'][:,0] = x
        data['a_position'][:,1] = y

        self.vbo_line = gloo.VertexBuffer(data)
        self.linecut_program.bind(self.vbo_line)

        self.update()

    def generate_vertices(self, xq, yq, z):
        """ Generate vertices for the dataset quadrilaterals """
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
        vertex_data['a_value'] = np.repeat(z.ravel(), 6, axis=0)

        return vertex_data

    def screen_to_data_coords(self, pos):
        """ Convert mouse position in the plot to data coordinates """
        if self.vbo is None:
            raise ValueError('No data has been plotted yet')

        screen_w, screen_h = self.size
        screen_x, screen_y = pos

        # Calculate in normalized coordinates
        relx = float(screen_x * 1.1) / screen_w
        rely = float(screen_h - screen_y) / screen_h

        # Convert to data coords using data min/max values
        dx = self.xmin + (relx) * (self.xmax - self.xmin)
        dy = self.ymin + (rely) * (self.ymax - self.ymin)

        return dx, dy

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear()

        if self.vbo is not None:
            logger.info('Redrawing the 2D plot')

            # Draw first the data, then colormap, and then linecut
            cmap_texture = gloo.Texture1D(self.colormap.get_colors(),
                                          interpolation='linear')

            # Draw the data
            self.data_program['u_colormap'] = cmap_texture
            self.data_program['z_min'] = self.colormap.min
            self.data_program['z_max'] = self.colormap.max
            self.data_program.draw('triangles')

            # Draw the colormap bar
            self.colorbar_program['u_colormap'] = cmap_texture
            colorbar_vertices = [(self.xmax + self.cm_dx * .2, self.ymax),
                                 (self.xmax + self.cm_dx * .2, self.ymin),
                                 (self.xmax + self.cm_dx, self.ymax),
                                 (self.xmax + self.cm_dx, self.ymin)]

            self.colorbar_program['a_position'] = colorbar_vertices
            self.colorbar_program['a_texcoord'] = [[1], [0], [1], [0]]
            self.colorbar_program.draw('triangle_strip')

            # Draw the linecut
            self.linecut_program.draw('line_strip')

        self.has_redrawn = True
