from PIL import Image
from object import Object
from point import Point

import numpy as np

class Plane(Object):
    def __init__(self, origin, normal, diffuse, reflection, shiny = 0.5, k = 8,
                 color = np.array([0,0,0]), texture = None, scale = 1.0):
        self.origin = np.array(origin)
        self.normal_v = np.array(normal)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k
        self.color = np.array(color)
        self.scale = scale
        self.texture = None

        if(texture is not None):
            self.texture = Image.open(texture)

        # uv mapping based on https://gamedev.stackexchange.com/questions/172352/finding-texture-coordinates-for-plane
        self.e1 = Point(np.cross(self.normal_v, np.array([1,0,0])), None)
        self.e1.normalize()

        if(self.e1.x() == 0 and self.e1.y() == 0 and self.e1.z() == 0):
            self.e1 = Point(np.cross(self.normal_v, np.array([0,0,1])), None)
            self.e1.normalize()

            # if(self.e1.x() == 0 and self.e1.y() == 0 and self.e1.z() == 0):
            #     self.e1 = Point(np.cross(self.normal_v, np.array([0,1,0])), None)
            #     self.e1.normalize()

        self.e2 = Point(np.cross(self.normal_v, self.e1.coords), None)
        self.e2.normalize()

    def normal(self):
        return(self.normal_v)

    def normal_at(self, origin):
        return(self.normal_v)

    def get_uv(self, point):
        u = np.dot(self.e1.coords, point.coords)
        v = np.dot(self.e2.coords, point.coords)
        return(u, v)

    def color_at(self, point):
        if(self.texture is None):
            return(self.color)

        u,v = self.get_uv(point)
        x = int(abs(u) * self.scale) % self.texture.width
        y = int(abs(v) * self.scale) % self.texture.height

        return(self.texture.getpixel((int(x),int(y)))[:3])

    # ray-plane intersection
    # adapted from https://stackoverflow.com/questions/23975555/how-to-do-ray-plane-intersection
    def geo_intersect(self, ray):
        denom = np.dot(self.normal_v, ray.direction)

        if(abs(denom) > 0.0001):
            t = np.dot(self.origin - ray.origin, self.normal_v) / denom
            if(t > 0):
                return(Point(ray.origin + (ray.direction * t), None), t)

        return(None, None)
