import numpy as np
import sys
import os

from PyQt4 import QtGui, QtCore

from vispy import app, gloo, scene, visuals
from vispy.util.transforms import ortho, translate

from colormap import Colormap
from util import FixedOrderFormatter, eng_format

import time

basic_vert = """
attribute vec2 a_position;

uniform mat4 u_view;
uniform mat4 u_projection;

void main()
{
    gl_Position = u_projection * u_view * vec4(a_position, 0.0, 1.0);
}
"""

basic_frag = """
void main()
{
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

cm_vert = """
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

cm_frag = """
uniform sampler1D u_colormap;

varying float v_texcoord;

void main()
{
    gl_FragColor = texture1D(u_colormap, v_texcoord);
    //gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

vert = """
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

frag = """
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

    A data point is drawn using two triangles to form a quad, it is colored
    by using the normalized data value and a colormap texture in the fragment shader.
    """
    def __init__(self, parent=None):
        scene.SceneCanvas.__init__(self, parent=parent)

        self.parent = parent
        self.has_redrawn = True

        self.data = None
        self.program = gloo.Program(vert, frag)

        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'new_colormaps/transform/Seismic.npy')
        self.colormap = Colormap(path)

        self.program_cm = gloo.Program(cm_vert, cm_frag)
        
        self.line_type = None
        self.line_coord = None
        self.line_positions = [(0, 0), (0, 0)]

        self.program_line = gloo.Program(basic_vert, basic_frag)
        self.program_line['a_position'] = self.line_positions

        gloo.set_clear_color((1, 1, 1, 1))

    def set_data(self, data):
        self.data = data

        vertices = self.generate_vertices(data)

        self.xmin, self.xmax = np.nanmin(vertices['a_position'][:,0]), np.nanmax(vertices['a_position'][:,0])
        self.ymin, self.ymax = np.nanmin(vertices['a_position'][:,1]), np.nanmax(vertices['a_position'][:,1])

        #self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = data.get_limits()

        self.cm_dx = (self.xmax-self.xmin)*0.1

        self.view = translate((0, 0, 0))
        self.projection = ortho(self.xmin, self.xmax + self.cm_dx, self.ymin, self.ymax, -1, 1)

        self.program['u_view'] = self.view
        self.program['u_projection'] = self.projection
        self.program['u_colormap'] = gloo.Texture1D(self.colormap.get_colors(), interpolation='linear')
        
        self.program_cm['u_view'] = self.view
        self.program_cm['u_projection'] = self.projection

        self.program_line['u_view'] = self.view
        self.program_line['u_projection'] = self.projection

        t0 = time.clock()
        #print 'generate_vertices: ', time.clock()-t0
        self.vbo = gloo.VertexBuffer(vertices)
        self.program.bind(self.vbo)

        self.update()

    def generate_vertices(self, data):
        #x, y, z = data.get_sorted_by_coordinates()
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
                                                      ('a_value', np.float32, 1)])
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

    def draw_linecut(self, event, old_position=False):
        # We need to check wether the canvas has had time to redraw itself 
        # because continuous mouse movement events surpress the redrawing.
        if self.data != None and self.has_redrawn:
            x_name, y_name, data_name, order_x, order_y = self.parent.get_axis_names()

            if not old_position:
                x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))
                
                if event.button == 1:
                    self.line_type = 'horizontal'
                    self.line_coord = self.data.get_closest_y(y)
                    self.line_positions = [(self.xmin, self.line_coord), (self.xmax, self.line_coord)]

                    x, y, index = self.data.get_row_at(y)
                    z = np.nanmean(self.data.y[index,:])
                    self.parent.linecut.plot_linetrace(x, y, z, self.line_type, self.line_coord, self.parent.name, x_name, data_name, y_name)
                    self.has_redrawn = False
                elif event.button == 3:
                    self.line_type = 'vertical'
                    self.line_coord = self.data.get_closest_x(x)
                    self.line_positions = [(self.line_coord, self.ymin), (self.line_coord, self.ymax)]
                    
                    x, y, index = self.data.get_column_at(x)
                    z = np.nanmean(self.data.x[:,index])
                    self.parent.linecut.plot_linetrace(x, y, z, self.line_type, self.line_coord, self.parent.name, y_name, data_name, x_name)
                    self.has_redrawn = False
            elif self.line_coord != None:
                if self.line_type == 'horizontal':
                    self.line_positions = [(self.xmin, self.line_coord), (self.xmax, self.line_coord)]
                    x, y, index = self.data.get_row_at(self.line_coord)
                    z = np.nanmean(self.data.y[:,index])
                    self.parent.linecut.plot_linetrace(x, y, z, self.line_type, self.line_coord, self.parent.name, x_name, data_name, y_name)
                else:
                    self.line_positions = [(self.line_coord, self.ymin), (self.line_coord, self.ymax)]
                    x, y, index = self.data.get_column_at(self.line_coord)
                    z = np.nanmean(self.data.x[index,:])
                    self.parent.linecut.plot_linetrace(x, y, z, self.line_type, self.line_coord, self.parent.name, y_name, data_name, x_name)
                
                self.has_redrawn = False

            self.program_line['a_position'] = self.line_positions

            self.update()
        else:
            self.update()

    def on_mouse_press(self, event):
        self.draw_linecut(event)

    def on_mouse_move(self, event):
        if self.data != None:
            sw, sh = self.size
            sx, sy = event.pos

            if 0 <= sx < sw and 0 <= sy < sh:
                x, y = self.screen_to_data_coords((event.pos[0], event.pos[1]))
                xstr, ystr = eng_format(x, 1), eng_format(y, 1)
                self.parent.status_bar.showMessage('X: %s\t\t\tY: %s' % (xstr, ystr))

                self.draw_linecut(event)

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear()

        if self.data != None:
            self.program['u_colormap'] = gloo.Texture1D(self.colormap.get_colors())
            self.program['z_min'] = self.colormap.min
            self.program['z_max'] = self.colormap.max

            self.program_cm['u_colormap'] = gloo.Texture1D(self.colormap.get_colors())
            self.program_cm['a_position'] = [(self.xmax + self.cm_dx*.2, self.ymax), 
                                             (self.xmax + self.cm_dx*.2, self.ymin), 
                                             (self.xmax + self.cm_dx, self.ymax), 
                                             (self.xmax + self.cm_dx, self.ymin)]
            self.program_cm['a_texcoord'] = [[1], [0], [1], [0]]

            self.program.draw('triangles')
            self.program_cm.draw('triangle_strip')
            self.program_line.draw('lines')

        self.has_redrawn = True