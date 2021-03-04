from geometry.Object import Object
from vectors.Point import Point

import numpy as np
import math

from PIL import Image

class Sphere(Object):
    def __init__(self, origin, radius = 1, diffuse = 0.5, reflection = 0.5, shiny = 0.5, k = 8, color = [255, 255, 255], texture = None, uv = (0, 0)):

        self.origin = np.array(origin)
        self.radius = radius
        self.color = np.array(color)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k
        self.texture = None
        self.uv = uv

        if(texture is not None):
            self.texture = Image.open(texture)

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

    def normal_at(self, point):
        dir = point - self.origin
        norm = np.linalg.norm(dir)
        if(norm != 0):
            dir /= norm
        return(dir)

    # based on https://viclw17.github.io/2019/04/12/raytracing-uv-mapping-and-texturing/
    # this might be faster https://gamedev.stackexchange.com/questions/114412/how-to-get-uv-coordinates-for-sphere-cylindrical-projection
    def get_uv(self, point):
        phi = np.arctan2(point.z(), point.x())
        theta = math.asin(point.y())

        u = (phi + np.pi) / (2 * np.pi);
        v = (theta + np.pi / 2) / np.pi;
        return(self.uv[0] - u, v - self.uv[1])

        # faster?
        # u = np.arctan2(point.x(), point.z()) / (2 * np.pi) + 0.5;
        # v = point.y() * 0.5 + 0.5;
        # return(1.0 - u, v - 1.0)

    def color_at(self, point):
        if(self.texture is None):
            return(self.color)

        p = Point(self.origin - point.coords, None)
        p.normalize()

        u,v = self.get_uv(p)
        x = self.texture.width * u
        y = self.texture.height * v
        return(self.texture.getpixel((x,y))[:3])

class Skybox(Sphere):

    def normal_at(self, origin):
        dir = self.origin - origin
        norm = np.linalg.norm(dir)
        if(norm != 0):
            dir /= norm
        return(dir)
