from object import Object
from point import Point

import numpy as np


class Sphere(Object):
    def __init__(self, origin, radius = 1, diffuse = 0.5, reflection = 0.5, shiny = 0.5, k = 8, color = [255, 255, 255]):
        self.origin = np.array(origin)
        self.radius = radius
        self.color = np.array(color)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k

    # adapted from http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
    def geo_intersect(self, ray):

        oc = ray.origin - self.origin
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - (self.radius * self.radius)
        discriminant = (b * b) - (4 * a * c)

        if(discriminant < 0.0):
            return(None, None)
        else:
            numerator = -b - np.sqrt(discriminant)
            if(numerator > 0.0):
                distance = numerator / (2.0 * a)
                origin = ray.origin + (ray.direction * distance)
                return(Point(origin, None), distance)

            numerator = -b + np.sqrt(discriminant)
            if(numerator > 0.0):
                distance = numerator / (2.0 * a)
                origin = ray.origin + (ray.direction * distance)
                return(Point(origin, None), distance)
            else:
                return(None, None)

    def normal_at(self, origin):
        dir = origin - self.origin
        norm = np.linalg.norm(dir)
        if(norm != 0):
            dir /= norm
        return(dir)

    def color_at(self, origin):
        return(self.color)
