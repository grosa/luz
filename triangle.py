from object import Object
from point import Point

import numpy as np

class Triangle(Object):

    def __init__(self, v0, v1, v2, diffuse = 0.5, reflection = 0.5, shiny = 0.5, k = 8,
                 color0 = [255, 255, 255], color1 = [255, 255, 255], color2 = [255, 255, 255]):
        self.v0 = np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k
        self.color0 = np.array(color0)
        self.color1 = np.array(color1)
        self.color2 = np.array(color2)

        self.A = Point(self.v1 - self.v0, None)
        # self.A.normalize()

        self.B = Point(self.v2 - self.v0, None)
        # self.B.normalize()

        self.C = Point(np.cross(self.A.coords, self.B.coords), None)
        self.C.normalize()

        # print("Trig normal:", self.C.coords)

    # Möller-Trumbore, adapted from https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    def geo_intersect(self, ray):

        pvec = np.cross(ray.direction, self.B.coords)
        det = np.dot(self.A.coords, pvec)

        if (det < 0.001):
            return(None, None)
        invDet = 1.0 / det;

        tvec = ray.origin - self.v0;
        u = np.dot(tvec, pvec) * invDet

        if (u < 0.0 or u > 1.0):
            return(None, None)

        qvec = np.cross(tvec, self.A.coords);
        v = np.dot(ray.direction, qvec) * invDet

        if (v < 0.0 or (u + v) > 1.0):
            return(None, None)

        t = np.dot(self.B.coords, qvec) * invDet;
        # point_parametric = ray.origin + t * ray.direction
        point = Point( (u * self.v1) + (v * self.v2) + ((1 - u - v) * self.v0), (u,v,(1 - u - v)))

        distance = np.sqrt(sum(point.coords * point.coords))
        return(point, distance)
        # return(point_parametric, t)

    def normal_at(self, point):
        return(self.C.coords)

    def color_at(self, point):
        u, v, w = point.barycentric()
        return(u * self.color1 + v * self.color2 + w * self.color0)
