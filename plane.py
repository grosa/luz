from object import Object
from point import Point

import numpy as np

class Plane(Object):
    def __init__(self, origin, normal, diffuse, reflection, shiny = 0.5, k = 8, color1 = np.array([0,0,0]), color2 = np.array([255, 255, 255])):
        self.origin = np.array(origin)
        self.normal_v = np.array(normal)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k
        self.color1 = np.array(color1)
        self.color2 = np.array(color2)

    def normal(self):
        return(self.normal_v)

    def normal_at(self, origin):
        return(self.normal_v)

    # simple checkerboard hack
    def color_at(self, origin):
        if((int(origin.x()) % 2 == 0) and (int(origin.z()) % 2 == 0)):
            return(self.color1)
        if((int(origin.x()) % 2 == 1) and (int(origin.z()) % 2 == 1)):
            return(self.color1)
        return(self.color2)

    # ray-plane intersection
    # adapted from https://stackoverflow.com/questions/23975555/how-to-do-ray-plane-intersection
    def geo_intersect(self, ray):
        denom = np.dot(self.normal_v, ray.direction)

        if(abs(denom) > 0.0001):
            t = np.dot(self.origin - ray.origin, self.normal_v) / denom
            if(t > 0):
                return(Point(ray.origin + (ray.direction * t), None), t)

        return(None, None)
